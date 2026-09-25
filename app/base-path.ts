/**
 * Next.js serves this app under `basePath`, so anything shipped in `public/`
 * (workers, worklets, WASM, ONNX weights) lives behind that prefix at runtime.
 */
export const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

/** Absolute URL for a file served from `public/`. */
export function assetUrl(path: string): string {
  return `${BASE_PATH}${path.startsWith("/") ? path : `/${path}`}`;
}
