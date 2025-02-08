import { generateObject } from 'ai';
import { compact } from 'lodash-es';
import pLimit from 'p-limit';
import { z } from 'zod';
import fetch from 'node-fetch';
import TurndownService from 'turndown';

import { o3MiniModel, trimPrompt } from './ai/providers';
import { systemPrompt } from './prompt';

// DEBUG: Indication de chargement
console.info("=== LOADED deep-research.ts WITH CHUNKING & NO 60s TIMEOUT ===");

export type ResearchResult = {
  learnings: string[];
  visitedUrls: string[];
};

const ConcurrencyLimit = 2;
const turndownService = new TurndownService();

const SEARXNG_URL = process.env.SEARXNG_URL || 'http://10.13.0.5:8081';

/**
 * Effectue une recherche sur SearXNG et retourne les URLs associées.
 */
async function searchSearxng(
  query: string,
  { limit = 5, timeout = 15000 } = {}
): Promise<{ data: Array<{ url: string; snippet?: string }> }> {
  console.info(`[searchSearxng] SERP for "${query}" (limit=${limit}, timeout=${timeout})`);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const params = new URLSearchParams({ q: query, format: 'json' });
    const url = `${SEARXNG_URL}/search?${params.toString()}`;
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

    console.info(`[searchSearxng] Found ${data.length} result(s) for "${query}".`);
    console.debug(`[searchSearxng] URLs: ${data.map(d => d.url).join(', ')}`);

    return { data };
  } catch (error) {
    clearTimeout(timeoutId);
    console.error(`[searchSearxng] Error on "${query}":`, error);
    throw error;
  }
}

/**
 * Nettoie le Markdown pour le rendre plus exploitable par un LLM
 */
function cleanMarkdown(md: string): string {
  return md
    .replace(/\n{3,}/g, '\n\n')
    .replace(/^[\s-]+$/gm, '')
    .trim();
}

/**
 * Découpe une longue chaîne en plusieurs chunks de taille max (par ex. 12k chars).
 */
function chunkText(text: string, chunkSize = 50000): string[] {
  const chunks: string[] = [];
  let startIndex = 0;
  while (startIndex < text.length) {
    const endIndex = startIndex + chunkSize;
    chunks.push(text.slice(startIndex, endIndex));
    startIndex = endIndex;
  }
  return chunks;
}

/**
 * Scrape le contenu d'une URL avec Crawl4AI (Markdown le plus propre possible).
 */
async function scrapeUrl(url: string, timeout = 20000): Promise<string> {
  console.info(`[scrapeUrl] Scraping: ${url}`);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (process.env.CRAWL4AI_API_TOKEN) {
      headers['Authorization'] = `Bearer ${process.env.CRAWL4AI_API_TOKEN}`;
    }

    const crawlRequest = {
      urls: url,
      priority: 10,
      options: {
        ignore_links: true,
        ignore_images: true,
        escape_html: true,
        body_width: 80,
      },
    };

    const crawlRes = await fetch(`${process.env.CRAWL4AI_API_URL}/crawl`, {
      method: 'POST',
      headers,
      body: JSON.stringify(crawlRequest),
      signal: controller.signal,
    });
    if (!crawlRes.ok) {
      throw new Error(`Erreur HTTP: ${crawlRes.status}`);
    }

    const crawlJson = await crawlRes.json();
    const taskId = crawlJson.task_id;
    if (!taskId) {
      throw new Error(`Aucun task_id retourné pour ${url}.`);
    }

    // Polling du résultat
    const startTime = Date.now();
    let taskResult: any = null;
    while (Date.now() - startTime < timeout) {
      const taskRes = await fetch(`${process.env.CRAWL4AI_API_URL}/task/${taskId}`, { headers, signal: controller.signal });
      if (!taskRes.ok) {
        throw new Error(`Erreur HTTP lors du polling: ${taskRes.status}`);
      }
      const taskJson = await taskRes.json();
      if (taskJson.status === "completed") {
        taskResult = taskJson;
        break;
      }
      await new Promise(resolve => setTimeout(resolve, 2000));
    }

    if (!taskResult) {
      throw new Error(`Timeout scraping: ${url}`);
    }

    let markdown = taskResult.result?.fit_markdown || taskResult.result?.markdown || '';
    if (!markdown && taskResult.result?.html) {
      markdown = turndownService.turndown(taskResult.result.html);
    }
    if (!markdown) {
      console.warn(`[scrapeUrl] No exploitable Markdown for ${url}`);
      return '';
    }

    const cleaned = cleanMarkdown(markdown);
    console.info(`[scrapeUrl] Got Markdown for ${url}, length=${cleaned.length}`);
    return cleaned;
  } finally {
    clearTimeout(timeoutId);
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
  console.info(`[generateSerpQueries] Generating up to ${numQueries} SERP queries for: "${query}"`);
  console.debug(`[generateSerpQueries] Current learnings: ${JSON.stringify(learnings, null, 2)}`);

  const promptToLLM = `Given the following prompt from the user, generate a list of SERP queries to research the topic. 
Return a maximum of ${numQueries} queries, but feel free to return less if the original prompt is clear.
Each query must be unique and not similar to each other.

<prompt>${query}</prompt>

${
  learnings?.length
    ? `Here are some learnings from previous research:\n${learnings.join('\n')}`
    : ''
}`;

  const truncatedPrompt = promptToLLM.length > 1500
    ? promptToLLM.substring(0, 1500) + '...\n[TRUNCATED]'
    : promptToLLM;
  console.debug(`[generateSerpQueries] PromptToLLM:\n${truncatedPrompt}`);

  const res = await generateObject({
    model: o3MiniModel,
    system: systemPrompt(),
    prompt: promptToLLM,
    schema: z.object({
      queries: z.array(z.object({
        query: z.string(),
        researchGoal: z.string(),
      })),
    }),
  });

  console.info(`[generateSerpQueries] Got ${res.object.queries.length} queries.`);
  console.debug(`[generateSerpQueries] Queries: ${JSON.stringify(res.object.queries, null, 2)}`);

  return res.object.queries.slice(0, numQueries);
}

/**
 * Appelle le LLM sur un seul chunk de texte, renvoie { learnings, followUpQuestions } partiels.
 * Pas de timeout => on laisse le LLM répondre aussi longtemps qu'il le faut.
 */
async function analyzeChunk({
  query,
  chunk,
  numLearnings,
  numFollowUpQuestions,
}: {
  query: string;
  chunk: string;
  numLearnings: number;
  numFollowUpQuestions: number;
}): Promise<{ learnings: string[]; followUpQuestions: string[] }> {
  console.debug(`[analyzeChunk] Starting chunk analysis (length=${chunk.length}) for query="${query}"...`);
  
  const promptToLLM = `Given this chunk of text from a SERP search for <query>${query}</query>,
generate up to ${numLearnings} distinct learnings and up to ${numFollowUpQuestions} follow-up questions.
The learnings should be factual, concise, and mention any key names or numbers found in the text.
If there's uncertainty, disclaim it.

<content length=${chunk.length}>
${chunk}
</content>`;

  const truncatedPrompt = promptToLLM.length > 1500
    ? promptToLLM.substring(0, 1500) + '...\n[TRUNCATED]'
    : promptToLLM;
  console.debug(`[analyzeChunk] Prompt:\n${truncatedPrompt}`);

  try {
    const res = await generateObject({
      model: o3MiniModel,
      system: systemPrompt(),
      prompt: promptToLLM,
      schema: z.object({
        learnings: z.array(z.string()),
        followUpQuestions: z.array(z.string()),
      }),
      // Pas de timeout => on laisse la requête se terminer.
    });

    console.debug(`[analyzeChunk] LLM response => learnings=${res.object.learnings.length}, followUps=${res.object.followUpQuestions.length}`);
    return res.object;
  } catch (error) {
    console.error(`[analyzeChunk] LLM error for chunk => `, error);
    return { learnings: [], followUpQuestions: [] };
  }
}

/**
 * Analyse le contenu scrappé en le découpant en chunks si nécessaire,
 * puis combine tous les learnings et followUpQuestions.
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
  console.info(`[processSerpResult] Analyzing SERP for "${query}"...`);
  console.debug(`[processSerpResult] raw data => ${JSON.stringify(result.data.map(d => ({url:d.url, length:(d.markdown||'').length})), null, 2)}`);

  // On récupère tous les markdown non vides
  const contents = compact(
    result.data.map(item => item.markdown)
  );

  console.info(`[processSerpResult] Non-empty contents: ${contents.length}`);
  if (!contents.length) {
    console.warn(`[processSerpResult] No non-empty contents => no learnings`);
    return { learnings: [], followUpQuestions: [] };
  }

  let aggregatedLearnings: string[] = [];
  let aggregatedFollowUps: string[] = [];

  // On parcourt chaque "content" (un par URL scrappée)
  for (const content of contents) {
    // Tronque globalement à 50k (pour éviter d'analyser 200k d'un coup)
    // puis on chunk par 12k si encore trop grand
    const truncated = trimPrompt(content, 50000);
    const splitted = chunkText(truncated, 12000);

    for (const chunk of splitted) {
      // Pour chaque chunk, on appelle analyzeChunk
      const partialRes = await analyzeChunk({
        query,
        chunk,
        numLearnings,
        numFollowUpQuestions,
      });
      // On additionne
      aggregatedLearnings.push(...partialRes.learnings);
      aggregatedFollowUps.push(...partialRes.followUpQuestions);
    }
  }

  // On peut déduire un ensemble unique
  const uniqueLearnings = [...new Set(aggregatedLearnings)];
  const uniqueFollowUps = [...new Set(aggregatedFollowUps)];

  console.info(`[processSerpResult] Final aggregated => learnings=${uniqueLearnings.length}, followUps=${uniqueFollowUps.length}`);
  return { learnings: uniqueLearnings, followUpQuestions: uniqueFollowUps };
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
  console.info(`[writeFinalReport] Generating final report for "${prompt}"`);
  console.debug(`[writeFinalReport] nbLearnings=${learnings.length}, nbURLs=${visitedUrls.length}`);

  const learningsString = trimPrompt(
    learnings.map(learning => `<learning>\n${learning}\n</learning>`).join('\n'),
    150000
  );

  const promptToLLM = `Given the user prompt, write a final report including ALL learnings.
Aim for at least 3 pages of text. Keep it well structured:

<prompt>${prompt}</prompt>

<learnings>
${learningsString}
</learnings>`;

  const truncatedPrompt = promptToLLM.length > 2000
    ? promptToLLM.substring(0, 2000) + '...\n[TRUNCATED]'
    : promptToLLM;
  console.debug(`[writeFinalReport] Prompt:\n${truncatedPrompt}`);

  try {
    // Pas de timeout => on laisse l'appel se finir
    const res = await generateObject({
      model: o3MiniModel,
      system: systemPrompt(),
      prompt: promptToLLM,
      schema: z.object({
        reportMarkdown: z.string(),
      }),
    });

    console.info(`[writeFinalReport] Final report length=${res.object.reportMarkdown.length}`);

    const urlsSection = `\n\n## Sources\n\n${visitedUrls.map(url => `- ${url}`).join('\n')}`;
    return res.object.reportMarkdown + urlsSection;
  } catch (error) {
    console.error(`[writeFinalReport] LLM error => `, error);
    return "An error occurred while generating the final report.\n";
  }
}

/**
 * Réalise une recherche approfondie en combinant plusieurs requêtes SERP.
 * Pour chaque URL trouvée, on scrape le contenu et on l'envoie au LLM pour analyse.
 */
import pLimit from 'p-limit';

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
  console.info(`[deepResearch] Start => query="${query}", breadth=${breadth}, depth=${depth}`);

  let serpQueries;
  try {
    serpQueries = await generateSerpQueries({ query, learnings, numQueries: breadth });
  } catch (error) {
    console.error(`[deepResearch] Error in generateSerpQueries:`, error);
    throw error;
  }

  const limit = pLimit(ConcurrencyLimit);

  const results = await Promise.all(
    serpQueries.map(serpQuery =>
      limit(async () => {
        try {
          console.info(`[deepResearch] SERP Query => "${serpQuery.query}"`);
          const searxResult = await searchSearxng(serpQuery.query, { timeout: 15000, limit: 5 });
          const scrapedData = await Promise.all(
            searxResult.data.map(async (item) => {
              console.info(`[deepResearch] --> Scraping: ${item.url}`);
              const markdown = await scrapeUrl(item.url, 15000);
              return { url: item.url, markdown };
            })
          );
          const newResult = { data: scrapedData };
          const newUrls = compact(scrapedData.map(s => s.url));

          console.debug(`[deepResearch] Combined => newUrls count=${newUrls.length}`);

          const newLearnings = await processSerpResult({
            query: serpQuery.query,
            result: newResult,
            numFollowUpQuestions: Math.ceil(breadth / 2),
          });

          const allLearnings = [...learnings, ...newLearnings.learnings];
          const allUrls = [...visitedUrls, ...newUrls];

          console.debug(`[deepResearch] SERP "${serpQuery.query}" => +${newLearnings.learnings.length} learnings, now total=${allLearnings.length}`);

          const newDepth = depth - 1;
          if (newDepth > 0) {
            const newBreadth = Math.ceil(breadth / 2);
            const nextQuery = `
              Previous research goal: ${serpQuery.researchGoal}
              Follow-up research directions: ${newLearnings.followUpQuestions.join('\n')}
            `.trim();

            console.info(`[deepResearch] Going deeper => newDepth=${newDepth}, newBreadth=${newBreadth}`);

            return deepResearch({
              query: nextQuery,
              breadth: newBreadth,
              depth: newDepth,
              learnings: allLearnings,
              visitedUrls: allUrls,
            });
          } else {
            console.info(`[deepResearch] Depth=0 => returning final partial result for "${serpQuery.query}"`);
            return { learnings: allLearnings, visitedUrls: allUrls };
          }
        } catch (e: any) {
          if (e.message && e.message.includes('Timeout')) {
            console.error(`[deepResearch] Timeout for "${serpQuery.query}"`, e);
          } else {
            console.error(`[deepResearch] Error for "${serpQuery.query}"`, e);
          }
          return { learnings: [], visitedUrls: [] };
        }
      })
    )
  );

  // Flatten the results
  const finalLearnings = [...new Set(results.flatMap(r => r.learnings))];
  const finalUrls = [...new Set(results.flatMap(r => r.visitedUrls))];

  console.info(`[deepResearch] Done => totalLearnings=${finalLearnings.length}, totalURLs=${finalUrls.length}`);
  console.debug(`[deepResearch] Full finalLearnings => ${JSON.stringify(finalLearnings, null, 2)}`);

  return { learnings: finalLearnings, visitedUrls: finalUrls };
}
