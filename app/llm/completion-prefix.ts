import type { ContextTurn } from "./conversation-context.ts";

/** Find a token prefix shared by prompts with and without session history. */
export function getCompletionCachePrefix(
  tokenize: (fragment: string, context?: ContextTurn[]) => readonly bigint[],
): bigint[] {
  // History precedes "Text before cursor", so fragment-only probes would
  // cache part of that label and miss every request containing history.
  const probes = [
    tokenize("alpha begins here"),
    tokenize("Zulu starts elsewhere"),
    tokenize("alpha begins here", [{ transcript: "A prior thought", completions: [] }]),
  ];
  let commonLength = 0;
  while (
    probes.every(ids => commonLength < ids.length && ids[commonLength] === probes[0][commonLength])
  ) {
    commonLength++;
  }

  // Leave room for tokenizer merges at the variable-content boundary.
  const prefixLength = commonLength - 2;
  if (prefixLength < 8) {
    throw new Error("Completion prompt has no safely cacheable prefix");
  }
  return probes[0].slice(0, prefixLength);
}
