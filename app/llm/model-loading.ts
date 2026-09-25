/** Stream one file at a time into the same cache used by inference. No tensors
 * or JS copies of the complete model are allocated by this preparation step. */
export async function prefetchModelFiles(
  cacheName: string,
  urls: readonly string[],
  signal: AbortSignal,
  storage: Pick<CacheStorage, "open"> | undefined = globalThis.caches,
  fetchFile: typeof fetch = globalThis.fetch,
) {
  const startedAt = performance.now();
  let downloadedFiles = 0;
  let cachedFiles = 0;
  try {
    if (!storage) return { downloadedFiles, cachedFiles, ms: 0, skipped: true };
    const cache = await storage.open(cacheName);
    for (const url of urls) {
      signal.throwIfAborted();
      if (await cache.match(url)) {
        cachedFiles++;
        continue;
      }
      const response = await fetchFile(url, { signal });
      if (response.status !== 200) {
        await response.body?.cancel();
        throw new Error(`Model prefetch failed: ${response.status}`);
      }
      await cache.put(url, response);
      downloadedFiles++;
    }
    return { downloadedFiles, cachedFiles, ms: performance.now() - startedAt, skipped: false };
  } catch (error) {
    // Preparation is optional. Inference's ordinary loader remains the fallback.
    return { downloadedFiles, cachedFiles, ms: performance.now() - startedAt, skipped: true,
      error: error instanceof Error ? error.message : String(error) };
  }
}

/** File reads and session construction can overlap. Report the initialization
 * tail after the final file, rather than claiming these are disjoint CPU costs. */
export function createModelLoadProfile(
  weightFiles: readonly string[],
  onWeightsReady: () => void,
  now = () => performance.now(),
) {
  const startedAt = now();
  let lastFileAt = startedAt;
  let initializedAt = startedAt;
  const pending = new Set(weightFiles);
  let notified = false;
  return {
    progress(event: unknown) {
      const data = event as { status?: string; file?: string };
      if (data?.status !== "done" || !data.file) return;
      lastFileAt = now();
      pending.delete(data.file);
      if (!notified && pending.size === 0) {
        notified = true;
        onWeightsReady();
      }
    },
    initialized() {
      initializedAt = now();
      // Unknown model layouts still get overlap during warmup.
      if (!notified) { notified = true; onWeightsReady(); }
    },
    finish() {
      const finishedAt = now();
      return {
        filePhaseMs: lastFileAt - startedAt,
        initializationTailMs: Math.max(0, initializedAt - lastFileAt),
        warmupMs: finishedAt - initializedAt,
        totalMs: finishedAt - startedAt,
      };
    },
  };
}
