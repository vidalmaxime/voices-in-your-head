import { formatConversationContext, type ContextTurn } from "./conversation-context.ts";

export const COMPLETION_SYSTEM_PROMPT = `You complete a speaker's thought at the cursor.
Output only the words that naturally come next, without repeating the original fragment.
Write in the first person, in the speaker's own voice, tense, perspective, and rhythm.
Never address the speaker or use "you"; the speaker is talking to themselves, not to you.
The examples show the format only. Never reuse their words or ideas; answer the current fragment freshly.
Infer a plausible intention from the available details and carry it somewhere specific or surprising.
When context permits, introduce one vivid detail, image, contrast, consequence, or connection.
Avoid bland endings such as generic agreement, importance, positivity, or "making things better."
Do not answer, explain, summarize, give advice, address the speaker, or behave like a chatbot.
Never mention AI, language processing, the model, or system limitations unless the speaker did.
Return one natural clause or compact sentence fragment, usually 6 to 14 words, and stop when the thought feels complete.`;

// The fragments are deliberately unlike everyday spoken openings so a small
// model cannot pattern-match a live utterance onto an example and echo its
// continuation. matchesCompletionExample() is a hard backstop if one still does.
export const COMPLETION_EXAMPLES = [
  {
    fragment: "the plan only started to make sense once I",
    continuation: "admitted I had been solving the wrong problem the whole time",
  },
  {
    fragment: "what surprised me halfway through was how quickly I",
    continuation: "stopped caring about being right and just wanted to be finished",
  },
  {
    fragment: "the part I never say out loud is that I",
    continuation: "would rather be doubted than asked to explain myself again",
  },
  {
    fragment: "it took me years to notice that I",
    continuation: "measure a good day by how little I had to pretend",
  },
];

export function buildCompletionMessages(
  partialText: string,
  options: { includeExamples?: boolean; context?: ContextTurn[] } = {},
) {
  // A model finetuned on this exact task (scripts/finetune) needs no few-shot
  // examples; it was trained on the bare system prompt + fragment format.
  const includeExamples = options.includeExamples ?? true;
  return [
    { role: "system", content: COMPLETION_SYSTEM_PROMPT },
    ...(includeExamples
      ? COMPLETION_EXAMPLES.flatMap(({ fragment, continuation }) => [
          {
            role: "user",
            content: `Text before cursor: ${fragment}\nText after cursor:`,
          },
          { role: "assistant", content: continuation },
        ])
      : []),
    {
      role: "user",
      content: `${formatConversationContext(options.context ?? [])}Text before cursor: ${partialText}\nText after cursor:`,
    },
  ];
}

function normalizeForPrefix(text: string) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9' ]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function stripLeadingRepeatedFragment(text: string, fragment: string) {
  let cleaned = text;
  const normalizedFragment = normalizeForPrefix(fragment);

  for (let i = 0; i < 3; i++) {
    const normalizedCleaned = normalizeForPrefix(cleaned);
    if (!normalizedCleaned || !normalizedFragment) break;

    if (normalizedCleaned.startsWith(normalizedFragment)) {
      const wordsToStrip = normalizedFragment.split(" ").filter(Boolean).length;
      cleaned = cleaned.split(/\s+/).slice(wordsToStrip).join(" ");
      continue;
    }

    const fragmentWords = normalizedFragment.split(" ").filter(Boolean);
    let strippedSuffix = false;
    for (let len = Math.min(8, fragmentWords.length); len >= 2; len--) {
      const suffix = fragmentWords.slice(-len).join(" ");
      if (normalizedCleaned.startsWith(suffix)) {
        cleaned = cleaned.split(/\s+/).slice(len).join(" ");
        strippedSuffix = true;
        break;
      }
    }
    if (!strippedSuffix) break;
  }

  return cleaned;
}

function stripRepeatedBoundaryWord(text: string, fragment: string) {
  const fragmentWords = normalizeForPrefix(fragment).split(" ").filter(Boolean);
  const outputWords = text.split(/\s+/).filter(Boolean);
  if (fragmentWords.length === 0 || outputWords.length === 0) return text;

  const firstOutputWord = normalizeForPrefix(outputWords[0]);
  if (fragmentWords.at(-1) !== firstOutputWord) return text;
  return outputWords.slice(1).join(" ");
}

const DANGLING_WORDS = new Set([
  "a",
  "an",
  "and",
  "as",
  "at",
  "because",
  "but",
  "by",
  "for",
  "from",
  "in",
  "of",
  "on",
  "or",
  "the",
  "that",
  "to",
  "which",
  "with",
]);

function stripDanglingTail(text: string) {
  const words = text.split(/\s+/).filter(Boolean);
  while (words.length > 3) {
    const finalWord = words.at(-1) ?? "";
    if (/[.!]$/.test(finalWord)) break;
    if (!DANGLING_WORDS.has(normalizeForPrefix(finalWord))) break;
    words.pop();
  }
  return words.join(" ");
}

const EXAMPLE_CONTINUATION_KEYS = COMPLETION_EXAMPLES.map((example) =>
  normalizeForPrefix(example.continuation),
);

/**
 * True when the completion echoes one of the few-shot examples. Small models
 * sometimes copy an example when the live fragment resembles its fragment. A
 * run of six or more words shared with an example is far past coincidence, so
 * the completion is dropped rather than spoken back as a canned line.
 */
// Phrases that only occur when the model has slipped into talking to a listener
// instead of finishing the speaker's own thought. A self-directed fragment
// never contains these, so their first appearance ends the completion.
const CHATBOT_ASIDE =
  /\b(please\b|let me know|feel free to|as an ai|an? ai\b|would you like|do you want me to|i can help|i'?m happy to)/i;

function stripChatbotAside(text: string) {
  const match = text.match(CHATBOT_ASIDE);
  if (match?.index === undefined) return text;
  return text.slice(0, match.index).replace(/[\s,;:.\-"'"]+$/, "").trim();
}

export function matchesCompletionExample(cleaned: string) {
  const words = normalizeForPrefix(cleaned).split(" ").filter(Boolean);
  if (words.length === 0) return false;
  for (const key of EXAMPLE_CONTINUATION_KEYS) {
    if (!key) continue;
    if (key === words.join(" ")) return true;
    const keyWords = key.split(" ");
    const run = Math.min(6, keyWords.length, words.length);
    if (run < 6) continue;
    // Any six-word window of the output that appears in an example continuation.
    for (let i = 0; i + run <= words.length; i++) {
      const window = words.slice(i, i + run).join(" ");
      if (key.includes(window)) return true;
    }
  }
  return false;
}

export type ThoughtCompletionAnalysis = {
  /** The completion as it will be shown and spoken. */
  text: string;
  /**
   * True once the raw text already holds everything that will be kept: the
   * first sentence has ended, a role label or chatbot aside began, the word
   * cap was hit, or the output copies an example. Generating further tokens
   * could only add text that is discarded.
   */
  finished: boolean;
};

const STOP_LABEL_RE = /\b(User|Assistant|Fragment|Continuation|Completion)\s*:/i;
// Above the notebooks model's 24-word training ceiling; the 16-token default
// cap keeps every other model well under it.
const MAX_COMPLETION_WORDS = 26;

export function analyzeThoughtCompletion(text: string, fragment: string): ThoughtCompletionAnalysis {
  let finished = false;
  let cleaned = text
    .replace(/\s+/g, " ")
    .replace(/^(assistant|user|fragment|continuation|completion)\s*[:\-]\s*/i, "")
    .replace(/^[\s"'`.,:;!?()[\]{}-]+/, "")
    .trim();

  cleaned = stripLeadingRepeatedFragment(cleaned, fragment)
    .replace(/^(assistant|user|fragment|continuation|completion)\s*[:\-]\s*/i, "")
    .replace(/^[\s"'`.,:;!?()[\]{}-]+/, "")
    .trim();
  cleaned = stripRepeatedBoundaryWord(cleaned, fragment);
  const withoutAside = stripChatbotAside(cleaned);
  if (withoutAside !== cleaned) {
    cleaned = withoutAside;
    finished = true;
  }

  const stopMatch = cleaned.match(STOP_LABEL_RE);
  if (stopMatch?.index !== undefined && stopMatch.index > 0) {
    cleaned = cleaned.slice(0, stopMatch.index).trim();
    finished = true;
  }

  const sentenceEnd = cleaned.search(/[.!?]\s+/);
  if (sentenceEnd > 0) {
    cleaned = cleaned.slice(0, sentenceEnd + 1).trim();
    finished = true;
  } else if (/[a-z][.!?]$/i.test(cleaned)) {
    // A terminal mark after a word closes the thought; only a digit before the
    // mark (a decimal, a numbered list) could still continue.
    finished = true;
  }

  const words = cleaned.split(/\s+/).filter(Boolean);
  if (words.length > MAX_COMPLETION_WORDS) {
    cleaned = words.slice(0, MAX_COMPLETION_WORDS).join(" ");
    finished = true;
  }

  // Small models sometimes trail off into an open quote or an ellipsis.
  cleaned = cleaned.replace(/[\s"'"\u2026]+$/, "").replace(/\.\.\.$/, "").trim();
  cleaned = stripDanglingTail(cleaned.replace(/\?+$/, "").trim());
  // A dangling connector may leave a trailing comma behind it.
  cleaned = cleaned.replace(/[\s,;:]+$/, "").trim();
  // A copied example is worse than saying nothing; drop it so the turn stays silent.
  if (matchesCompletionExample(cleaned)) return { text: "", finished: true };
  return { text: cleaned, finished };
}

export function sanitizeThoughtCompletion(text: string, fragment: string) {
  return analyzeThoughtCompletion(text, fragment).text;
}
