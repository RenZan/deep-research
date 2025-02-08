import FirecrawlApp from '@mendable/firecrawl-js';
import { generateObject } from 'ai';
import { compact } from 'lodash-es';
import pLimit from 'p-limit';
import { z } from 'zod';
import fetch from 'node-fetch';
import TurndownService from 'turndown';

import { o3MiniModel, trimPrompt } from './ai/providers';
import { systemPrompt } from './prompt';

export type ResearchResult = {
  learnings: string[];
  visitedUrls: string[];
};

const ConcurrencyLimit = 2;

// Initialisation de Firecrawl
const firecrawl = new FirecrawlApp({
  apiKey: process.env.FIRECRAWL_KEY ?? '',
  apiUrl: process.env.FIRECRAWL_BASE_URL,
});
console.info(`[init] Firecrawl configuré avec apiUrl=${process.env.FIRECRAWL_BASE_URL}`);

// Instanciation du convertisseur HTML -> Markdown
const turndownService = new TurndownService();

const SEARXNG_URL = process.env.SEARXNG_URL || 'http://10.13.0.5:8081';

/**
 * Effectue une recherche sur SearXNG et retourne les URLs associées.
 */
async function searchSearxng(
  query: string,
  { limit = 5, timeout = 15000 } = {}
): Promise<{ data: Array<{ url: string; snippet?: string }> }> {
  const params = new URLSearchParams({
    q: query,
    format: 'json',
  });
  const url = `${SEARXNG_URL}/search?${params.toString()}`;
  console.info(`[searchSearxng] Lancement de la recherche SERP pour "${query}" via ${url}`);
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) {
      throw new Error(`Erreur HTTP: ${response.status}`);
    }
    const json = await response.json();
    clearTimeout(timeoutId);
    const data = (json.results || [])
      .slice(0, limit)
      .map((item: any) => ({
        url: item.url,
        snippet: item.content || item.snippet || '',
      }));
    console.info(`[searchSearxng] ${data.length} URL(s) trouvée(s) pour "${query}"`);
    return { data };
  } catch (error) {
    clearTimeout(timeoutId);
    console.error(`[searchSearxng] Erreur lors de la recherche pour "${query}":`, error);
    throw error;
  }
}

/**
 * Scrape le contenu d'une URL donnée en privilégiant le format Markdown.
 * Si aucun Markdown n'est retourné, essaie de convertir le HTML en Markdown.
 */
async function scrapeUrl(url: string, timeout = 15000): Promise<string> {
  console.info(`[scrapeUrl] Début du scraping de l'URL: ${url}`);
  try {
    // On demande à Firecrawl de renvoyer les formats markdown et html
    const result = await firecrawl.scrapeUrl(url, {
      timeout,
      formats: ['markdown', 'html'],
    });

    if (result.success) {
      // Si le Markdown est disponible, on l'utilise
      if (result.data?.markdown) {
        console.info(
          `[scrapeUrl] URL ${url} scrappée en Markdown avec succès (taille: ${result.data.markdown.length} caractères)`
        );
        return result.data.markdown;
      }
      // Sinon, si le HTML est disponible, on le convertit en Markdown
      else if (result.data?.html) {
        const converted = turndownService.turndown(result.data.html);
        console.info(
          `[scrapeUrl] URL ${url} n'a pas fourni de Markdown directement, conversion depuis HTML réussie (taille: ${converted.length} caractères)`
        );
        return converted;
      } else {
        console.warn(`[scrapeUrl] Aucun contenu utile (ni Markdown ni HTML) retourné pour l'URL: ${url}`);
        return '';
      }
    } else {
      console.warn(
        `[scrapeUrl] Le scraping de l'URL ${url} n'a pas été marqué comme un succès. Détails: ${JSON.stringify(result)}`
      );
      return '';
    }
  } catch (error: any) {
    console.error(`[scrapeUrl] Erreur lors du scraping de l'URL ${url}: ${error.message}`, error);
    return '';
  }
}

/**
 * Génère une liste de requêtes SERP à partir du prompt utilisateur.
 */
async function generateSerpQueries({
  query,
  numQueries = 3,
  learnings,
}: {
  query: string;
  numQueries?: number;
  learnings?: string[];
}) {
  console.info(
    `[generateSerpQueries] Génération de jusqu'à ${numQueries} requêtes SERP pour le prompt: "${query}"`
  );
  const res = await generateObject({
    model: o3MiniModel,
    system: systemPrompt(),
    prompt: `Given the following prompt from the user, generate a list of SERP queries to research the topic. Return a maximum of ${numQueries} queries, but feel free to return less if the original prompt is clear. Make sure each query is unique and not similar to each other: <prompt>${query}</prompt>\n\n${
      learnings
        ? `Here are some learnings from previous research, use them to generate more specific queries: ${learnings.join('\n')}`
        : ''
    }`,
    schema: z.object({
      queries: z
        .array(
          z.object({
            query: z.string().describe('The SERP query'),
            researchGoal: z.string().describe(
              'First talk about the goal of the research that this query is meant to accomplish, then go deeper into how to advance the research once the results are found, mention additional research directions. Be as specific as possible, especially for additional research directions.'
            ),
          }),
        )
        .describe(`List of SERP queries, max of ${numQueries}`),
    }),
  });
  console.info(`[generateSerpQueries] ${res.object.queries.length} requête(s) générée(s):`, res.object.queries);
  return res.object.queries.slice(0, numQueries);
}

/**
 * Utilise le LLM pour extraire des apprentissages à partir des contenus scrappés.
 */
async function processSerpResult({
  query,
  result,
  numLearnings = 3,
  numFollowUpQuestions = 3,
}: {
  query: string;
  result: { data: Array<{ url: string; markdown?: string }> };
  numLearnings?: number;
  numFollowUpQuestions?: number;
}) {
  const contents = compact(result.data.map(item => item.markdown)).map(content =>
    trimPrompt(content, 25000)
  );
  console.info(
    `[processSerpResult] Pour la requête "${query}", ${contents.length} contenu(s) scrappé(s) seront analysé(s) par le LLM.`
  );
  const res = await generateObject({
    model: o3MiniModel,
    abortSignal: AbortSignal.timeout(60000),
    system: systemPrompt(),
    prompt: `Given the following contents from a SERP search for the query <query>${query}</query>, generate a list of learnings from the contents. Return a maximum of ${numLearnings} learnings, but feel free to return less if the contents are clear. Make sure each learning is unique and not similar to each other. The learnings should be concise and to the point, as detailed and information dense as possible. Make sure to include any entities like people, places, companies, products, things, etc in the learnings, as well as any exact metrics, numbers, or dates. The learnings will be used to research the topic further.\n\n<contents>${contents
      .map(content => `<content>\n${content}\n</content>`)
      .join('\n')}</contents>`,
    schema: z.object({
      learnings: z.array(z.string()).describe(`List of learnings, max of ${numLearnings}`),
      followUpQuestions: z.array(z.string()).describe(
        `List of follow-up questions to research the topic further, max of ${numFollowUpQuestions}`
      ),
    }),
  });
  console.info(
    `[processSerpResult] Analyse LLM terminée pour "${query}". ${res.object.learnings.length} apprentissage(s) généré(s).`
  );
  return res.object;
}

/**
 * Génère le rapport final en combinant les apprentissages et les URLs visitées.
 */
export async function writeFinalReport({
  prompt,
  learnings,
  visitedUrls,
}: {
  prompt: string;
  learnings: string[];
  visitedUrls: string[];
}) {
  const learningsString = trimPrompt(
    learnings.map(learning => `<learning>\n${learning}\n</learning>`).join('\n'),
    150000
  );
  const res = await generateObject({
    model: o3MiniModel,
    system: systemPrompt(),
    prompt: `Given the following prompt from the user, write a final report on the topic using the learnings from research. Make it as as detailed as possible, aim for 3 or more pages, include ALL the learnings from research:\n\n<prompt>${prompt}</prompt>\n\nHere are all the learnings from previous research:\n\n<learnings>\n${learningsString}\n</learnings>`,
    schema: z.object({
      reportMarkdown: z.string().describe('Final report on the topic in Markdown'),
    }),
  });
  const urlsSection = `\n\n## Sources\n\n${visitedUrls.map(url => `- ${url}`).join('\n')}`;
  console.info(
    `[writeFinalReport] Rapport final généré avec ${learnings.length} apprentissage(s) et ${visitedUrls.length} source(s).`
  );
  return res.object.reportMarkdown + urlsSection;
}

/**
 * Réalise une recherche approfondie en combinant plusieurs requêtes SERP.
 * Pour chaque URL trouvée, on scrape le contenu et on l'envoie au LLM pour analyse.
 */
export async function deepResearch({
  query,
  breadth,
  depth,
  learnings = [],
  visitedUrls = [],
}: {
  query: string;
  breadth: number;
  depth: number;
  learnings?: string[];
  visitedUrls?: string[];
}): Promise<ResearchResult> {
  console.info(
    `[deepResearch] Début de la recherche approfondie pour "${query}" avec breadth ${breadth} et depth ${depth}`
  );
  const serpQueries = await generateSerpQueries({ query, learnings, numQueries: breadth });
  const limit = pLimit(ConcurrencyLimit);

  const results = await Promise.all(
    serpQueries.map(serpQuery =>
      limit(async () => {
        try {
          console.info(`[deepResearch] Requête SERP: "${serpQuery.query}"`);
          const searxResult = await searchSearxng(serpQuery.query, { timeout: 15000, limit: 5 });
          const scrapedData = await Promise.all(
            searxResult.data.map(async (item) => {
              console.info(`[deepResearch] Visite de l'URL: ${item.url}`);
              const markdown = await scrapeUrl(item.url, 15000);
              if (markdown) {
                console.info(
                  `[deepResearch] URL ${item.url} scrappée (taille: ${markdown.length} caractères). Contenu envoyé au LLM.`
                );
              } else {
                console.warn(`[deepResearch] URL ${item.url} n'a retourné aucun contenu utile.`);
              }
              return { url: item.url, markdown };
            })
          );
          const newResult = { data: scrapedData };
          const newUrls = compact(newResult.data.map(item => item.url));
          const newBreadth = Math.ceil(breadth / 2);
          const newDepth = depth - 1;
          const newLearnings = await processSerpResult({
            query: serpQuery.query,
            result: newResult,
            numFollowUpQuestions: newBreadth,
          });
          const allLearnings = [...learnings, ...newLearnings.learnings];
          const allUrls = [...visitedUrls, ...newUrls];

          if (newDepth > 0) {
            console.info(
              `[deepResearch] Approfondissement pour "${serpQuery.query}" - nouvelle breadth: ${newBreadth}, nouveau depth: ${newDepth}`
            );
            const nextQuery = `
              Previous research goal: ${serpQuery.researchGoal}
              Follow-up research directions: ${newLearnings.followUpQuestions.map(q => `\n${q}`).join('')}
            `.trim();

            return deepResearch({
              query: nextQuery,
              breadth: newBreadth,
              depth: newDepth,
              learnings: allLearnings,
              visitedUrls: allUrls,
            });
          } else {
            return { learnings: allLearnings, visitedUrls: allUrls };
          }
        } catch (e: any) {
          if (e.message && e.message.includes('Timeout')) {
            console.error(`[deepResearch] Timeout lors de l'exécution de la requête "${serpQuery.query}": `, e);
          } else {
            console.error(`[deepResearch] Erreur lors de l'exécution de la requête "${serpQuery.query}": `, e);
          }
          return { learnings: [], visitedUrls: [] };
        }
      })
    )
  );

  const finalLearnings = [...new Set(results.flatMap(r => r.learnings))];
  const finalUrls = [...new Set(results.flatMap(r => r.visitedUrls))];
  console.info(
    `[deepResearch] Recherche approfondie terminée. Total apprentissages: ${finalLearnings.length}. Total URLs visitées: ${finalUrls.length}`
  );
  return { learnings: finalLearnings, visitedUrls: finalUrls };
}
