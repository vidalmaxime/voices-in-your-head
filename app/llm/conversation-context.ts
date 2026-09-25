export interface ContextTurn {
  transcript: string;
  completions: string[];
}

export const MAX_CONTEXT_TURNS = 8;
export const MAX_CONTEXT_CHARACTERS = 6000;

/** Keep recent turns bounded so session history cannot overwhelm a small model. */
export function appendContextTurn(history: readonly ContextTurn[], turn: ContextTurn): ContextTurn[] {
  if (!turn.transcript.trim()) return [...history];
  const transcript = turn.transcript.trim().slice(-2000);
  const completions = turn.completions.map(text => text.trim()).filter(Boolean).slice(0, 20);
  const perCompletionLimit = Math.min(500, Math.floor((MAX_CONTEXT_CHARACTERS - transcript.length) / Math.max(1, completions.length)));
  const next = [...history, {
    transcript,
    completions: completions.map(text => text.slice(0, perCompletionLimit)),
  }].slice(-MAX_CONTEXT_TURNS);
  const size = () => next.reduce((sum, item) => sum + item.transcript.length + item.completions.join("").length, 0);
  while (next.length > 1 && size() > MAX_CONTEXT_CHARACTERS) next.shift();
  return next;
}

export function formatConversationContext(history: readonly ContextTurn[]): string {
  if (!history.length) return "";
  return `Earlier session context (background only; complete only the current cursor fragment). Generated alternatives are possible thoughts, not facts confirmed by the speaker.\n${history.map(turn =>
    `Speaker said: ${JSON.stringify(turn.transcript)}${turn.completions.length ? `\nGenerated ${turn.completions.length > 1 ? "alternatives" : "continuation"}: ${JSON.stringify(turn.completions)}` : ""}`
  ).join("\n\n")}\n\nCurrent fragment:\n`;
}
