import { generateObject } from 'ai';
import { compact } from 'lodash-es';
import pLimit from 'p-limit';
import { z } from 'zod';
import fetch from 'node-fetch';
import TurndownService from 'turndown';
import http from 'http';
import https from 'https';

import { o3MiniModel, trimPrompt } from './ai/providers';
import { systemPrompt } from './prompt';

// DEBUG: Indication de chargement
console.info("=== LOADED deep-research.ts WITH CHUNKING & NO 60s TIMEOUT ===");

export type ResearchResult = {
  learnings: string[];
  visitedUrls: string[];
};

const ConcurrencyLimit = 2;
const AnalyzeConcurrencyLimit = 2; // Limite pour l'analyse parallèle des chunks
const turndownService = new TurndownService();

const SEARXNG_URL = process.env.SEARXNG_URL || 'http://10.13.0.5:8081';

// Agents HTTP/HTTPS pour réutiliser les connexions
const httpAgent = new http.Agent({ keepAlive: true });
const httpsAgent = new https.Agent({ keepAlive: true });

function getAgent(url: string): http.Agent | https.Agent {
  return url.startsWith('https') ? httpsAgent : httpAgent;
}

/**
 * Fetch avec timeout et mécanisme de retry.
 */
async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeout: number = 15000,
  retries = 2
): Promise<fetch.Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  try {
    // Ajout de l'agent keep-alive
    const agent = getAgent(url);
    const response = await fetch(url, { ...options, agent, signal: controller.signal });
    clearTimeout(id);
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    return response;
  } catch (error) {
    clearTimeout(id);
    if (retries > 0) {
      // Petite pause avant retry
      await new Promise(resolve => setTimeout(resolve, 500));
      return fetchWithTimeout(url, options, timeout, retries - 1);
    }
    throw error;
  }
}

// Cache pour éviter de scraper plusieurs fois la même URL
const scrapeCache = new Map<string, Promise<string>>();

/**
 * Effectue une recherche sur SearXNG et retourne les URLs associées.
 */
async function searchSearxng(
  query: string,
  { limit = 5, timeout = 15000 } = {}
): Promise<{ data: Array<{ url: string; snippet?: string }> }> {
  console.info(`[searchSearxng] SERP for "${query}" (limit=${limit}, timeout=${timeout})`);

  try {
    const params = new URLSearchParams({ q: query, format: 'json' });
    const url = `${SEARXNG_URL}/search?${params.toString()}`;
    const response = await fetchWithTimeout(url, {}, timeout);
    const json = await response.json();

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
    console.error(`[searchSearxng] Error on "${query}":`, error);
    throw error;
  }
}

/**
 * Nettoie le Markdown pour le rendre plus exploitable par un LLM.
 */
function cleanMarkdown(md: string): string {
  return md
    .replace(/\n{3,}/g, '\n\n')
    .replace(/^[\s-]+$/gm, '')
    .trim();
}

/**
 * Découpe une longue chaîne en plusieurs chunks en essayant de ne pas couper au milieu d'une phrase.
 */
function chunkText(text: string, chunkSize = 10000): string[] {
  const chunks: string[] = [];
  let startIndex = 0;
  while (startIndex < text.length) {
    let endIndex = startIndex + chunkSize;
    if (endIndex < text.length) {
      // Cherche le dernier saut de ligne ou espace avant la limite du chunk
      const breakIndexNewline = text.lastIndexOf('\n', endIndex);
      const breakIndexSpace = text.lastIndexOf(' ', endIndex);
      const breakIndex = Math.max(breakIndexNewline, breakIndexSpace);
      if (breakIndex > startIndex) {
        endIndex = breakIndex;
      }
    }
    chunks.push(text.slice(startIndex, endIndex));
    startIndex = endIndex;
  }
  return chunks;
}

/**
 * Scrape le contenu d'une URL avec Crawl4AI (Markdown le plus propre possible).
 * Utilise un cache pour éviter les appels redondants.
 */
async function scrapeUrl(url: string, timeout = 20000): Promise<string> {
  if (scrapeCache.has(url)) {
    return scrapeCache.get(url)!;
  }
  const scrapingPromise = (async () => {
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

      const crawlUrl = `${process.env.CRAWL4AI_API_URL}/crawl`;
      const crawlRes = await fetchWithTimeout(
        crawlUrl,
        {
          method: 'POST',
          headers,
          body: JSON.stringify(crawlRequest),
        },
        timeout
      );
      const crawlJson = await crawlRes.json();
      const taskId = crawlJson.task_id;
      if (!taskId) {
        throw new Error(`Aucun task_id retourné pour ${url}.`);
      }

      // Polling du résultat
      const startTime = Date.now();
      let taskResult: any = null;
      while (Date.now() - startTime < timeout) {
        const taskRes = await fetchWithTimeout(
          `${process.env.CRAWL4AI_API_URL}/task/${taskId}`,
          { headers },
          timeout
        );
        const taskJson = await taskRes.json();
        if (taskJson.status === "completed") {
          taskResult = taskJson;
          break;
        }
        await new Promise(resolve => setTimeout(resolve, 2000));
      }

      if (!taskResult) {
        console.warn(`[scrapeUrl] error scraping for ${url}`);
        return '';
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
  })();

  scrapeCache.set(url, scrapingPromise);
  return scrapingPromise;
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

  console.debug(`[generateSerpQueries] PromptToLLM:\n${promptToLLM}`);

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
 * Aucun timeout n'est imposé afin de laisser le LLM répondre sans contrainte.
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

  try {
    const res = await generateObject({
      model: o3MiniModel,
      system: systemPrompt(),
      prompt: promptToLLM,
      schema: z.object({
        learnings: z.array(z.string()),
        followUpQuestions: z.array(z.string()),
      }),
    });

    console.debug(`[analyzeChunk] LLM response => learnings=${res.object.learnings.length}, followUps=${res.object.followUpQuestions.length}`);
    return res.object;
  } catch (error) {
    console.error(`[analyzeChunk] LLM error for chunk => `, error);
    return { learnings: [], followUpQuestions: [] };
  }
}

/**
 * Analyse le contenu scrappé en le découpant en chunks (en parallèle avec une limite de concurrence),
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
  console.debug(
    `[processSerpResult] raw data => ${JSON.stringify(
      result.data.map(d => ({ url: d.url, length: (d.markdown || '').length })),
      null,
      2
    )}`
  );

  const contents = compact(result.data.map(item => item.markdown));

  console.info(`[processSerpResult] Non-empty contents: ${contents.length}`);
  if (!contents.length) {
    console.warn(`[processSerpResult] No non-empty contents => no learnings`);
    return { learnings: [], followUpQuestions: [] };
  }

  let aggregatedLearnings: string[] = [];
  let aggregatedFollowUps: string[] = [];

  const analyzeLimit = pLimit(AnalyzeConcurrencyLimit);

  for (const content of contents) {
    const limitedContent = content.slice(0, 8000);
    const chunks = chunkText(limitedContent, 8000);
    const partialResults = await Promise.all(
      chunks.map(chunk =>
        analyzeLimit(() =>
          analyzeChunk({
            query,
            chunk,
            numLearnings,
            numFollowUpQuestions,
          })
        )
      )
    );
    for (const partialRes of partialResults) {
      aggregatedLearnings.push(...partialRes.learnings);
      aggregatedFollowUps.push(...partialRes.followUpQuestions);
    }
  }

  const uniqueLearnings = [...new Set(aggregatedLearnings)];
  const uniqueFollowUps = [...new Set(aggregatedFollowUps)];

  console.info(
    `[processSerpResult] Final aggregated => learnings=${uniqueLearnings.length}, followUps=${uniqueFollowUps.length}`
  );
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

  const learningsString = learnings
    .map(learning => `<learning>\n${learning}\n</learning>`)
    .join('\n');

  const promptToLLM = `Given the user prompt, write a final report, IN FRENCH, including ALL USEFUL learnings.
Aim for at least 3 pages of text. Keep it well structured and easy to read.

<prompt>${prompt}</prompt>

<learnings>
${learningsString}
</learnings>`;

  console.debug(`[writeFinalReport] Prompt:\n${promptToLLM}`);

  try {
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

          console.debug(
            `[deepResearch] SERP "${serpQuery.query}" => +${newLearnings.learnings.length} learnings, now total=${allLearnings.length}`
          );

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

  const finalLearnings = [...new Set(results.flatMap(r => r.learnings))];
  const finalUrls = [...new Set(results.flatMap(r => r.visitedUrls))];

  console.info(`[deepResearch] Done => totalLearnings=${finalLearnings.length}, totalURLs=${finalUrls.length}`);
  console.debug(`[deepResearch] Full finalLearnings => ${JSON.stringify(finalLearnings, null, 2)}`);

  return { learnings: finalLearnings, visitedUrls: finalUrls };
}
