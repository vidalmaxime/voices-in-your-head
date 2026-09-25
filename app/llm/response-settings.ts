export function normalizeResponseTokenLimit(value: number = 16): number {
  return Number.isFinite(value) ? Math.max(1, Math.min(256, Math.floor(value))) : 16;
}
