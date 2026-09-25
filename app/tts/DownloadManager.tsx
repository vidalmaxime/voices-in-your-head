"use client";

import { useState, useCallback, forwardRef, useImperativeHandle } from "react";

const CACHE_NAME = "tts-weights-cache-v1";

export interface DownloadManagerHandle {
  fetch: (label: string, url: string, headers?: Record<string, string>) => Promise<ArrayBuffer>;
}

interface DownloadState {
  label: string;
  progress: number;
  total: number;
  cached?: boolean;
}

const DownloadManager = forwardRef<DownloadManagerHandle>(function DownloadManager(_, ref) {
  const [downloads, setDownloads] = useState<Map<string, DownloadState>>(new Map());

  const fetchWithProgress = useCallback(async (label: string, url: string, headers?: Record<string, string>): Promise<ArrayBuffer> => {
    // Check cache first
    try {
      const cache = await caches.open(CACHE_NAME);
      const cachedResponse = await cache.match(url);
      if (cachedResponse) {
        setDownloads(prev => {
          const next = new Map(prev);
          next.set(url, { label, progress: 0, total: 0, cached: true });
          return next;
        });
        const buffer = await cachedResponse.arrayBuffer();
        setDownloads(prev => {
          const next = new Map(prev);
          next.delete(url);
          return next;
        });
        return buffer;
      }
    } catch (e) {
      console.warn("Cache API not available, downloading without cache:", e);
    }

    setDownloads(prev => {
      const next = new Map(prev);
      next.set(url, { label, progress: 0, total: 0 });
      return next;
    });

    try {
      const response = await fetch(url, headers ? { headers } : undefined);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const contentLength = response.headers.get("content-length");
      const total = contentLength ? parseInt(contentLength, 10) : 0;

      setDownloads(prev => {
        const next = new Map(prev);
        next.set(url, { label, progress: 0, total });
        return next;
      });

      if (!response.body) {
        const buffer = await response.arrayBuffer();
        setDownloads(prev => {
          const next = new Map(prev);
          next.delete(url);
          return next;
        });
        return buffer;
      }

      const reader = response.body.getReader();
      const chunks: Uint8Array[] = [];
      let received = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        chunks.push(value);
        received += value.length;

        setDownloads(prev => {
          const next = new Map(prev);
          next.set(url, { label, progress: received, total });
          return next;
        });
      }

      const buffer = new Uint8Array(received);
      let offset = 0;
      for (const chunk of chunks) {
        buffer.set(chunk, offset);
        offset += chunk.length;
      }

      // Cache the downloaded data
      try {
        const cache = await caches.open(CACHE_NAME);
        await cache.put(url, new Response(buffer.buffer.slice(0)));
      } catch (e) {
        console.warn("Failed to cache:", e);
      }

      setDownloads(prev => {
        const next = new Map(prev);
        next.delete(url);
        return next;
      });

      return buffer.buffer;
    } catch (error) {
      setDownloads(prev => {
        const next = new Map(prev);
        next.delete(url);
        return next;
      });
      throw error;
    }
  }, []);

  useImperativeHandle(ref, () => ({
    fetch: fetchWithProgress,
  }), [fetchWithProgress]);

  if (downloads.size === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded-lg shadow-lg p-4 min-w-64">
      <h3 className="text-sm font-medium mb-2">Downloads</h3>
      {Array.from(downloads.entries()).map(([url, { label, progress, total, cached }]) => (
        <div key={url} className="mb-2">
          <div className="text-xs text-zinc-600 dark:text-zinc-400 mb-1">
            {label} {cached && "(cached)"}
          </div>
          {cached ? (
            <div className="text-xs text-green-600 dark:text-green-400">
              Loading from cache...
            </div>
          ) : (
            <>
              <div className="w-full bg-zinc-200 dark:bg-zinc-700 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all"
                  style={{ width: total > 0 ? `${(progress / total) * 100}%` : "0%" }}
                />
              </div>
              <div className="text-xs text-zinc-500 mt-1">
                {total > 0
                  ? `${(progress / 1024 / 1024).toFixed(1)} / ${(total / 1024 / 1024).toFixed(1)} MB`
                  : `${(progress / 1024 / 1024).toFixed(1)} MB`}
              </div>
            </>
          )}
        </div>
      ))}
    </div>
  );
});

export default DownloadManager;
