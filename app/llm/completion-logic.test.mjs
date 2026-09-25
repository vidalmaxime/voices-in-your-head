import assert from "node:assert/strict";
import test from "node:test";

import {
  analyzeThoughtCompletion,
  buildCompletionMessages,
  COMPLETION_EXAMPLES,
  COMPLETION_SYSTEM_PROMPT,
  matchesCompletionExample,
  sanitizeThoughtCompletion,
} from "./completion-logic.ts";

test("builds a diverse completion prompt around the exact cursor fragment", () => {
  const messages = buildCompletionMessages("I wonder if tomorrow I should");

  assert.match(COMPLETION_SYSTEM_PROMPT, /specific or surprising/);
  assert.ok(COMPLETION_EXAMPLES.length >= 4);
  assert.equal(messages[0].role, "system");
  assert.equal(
    messages.at(-1).content,
    "Text before cursor: I wonder if tomorrow I should\nText after cursor:",
  );
});

test("removes a repeated source fragment", () => {
  assert.equal(
    sanitizeThoughtCompletion(
      "I think the reason this keeps happening is a lack of focus",
      "I think the reason this keeps happening is",
    ),
    "a lack of focus",
  );
});

test("removes a duplicated word at the fragment boundary", () => {
  assert.equal(
    sanitizeThoughtCompletion("to synchronize smoothly with my words", "I want the voice to"),
    "synchronize smoothly with my words",
  );
});

test("removes labels and text after the first sentence", () => {
  assert.equal(
    sanitizeThoughtCompletion("Assistant: feel more natural. User: next", "I want it to"),
    "feel more natural.",
  );
});

test("turns question-shaped output into an appendable clause", () => {
  assert.equal(
    sanitizeThoughtCompletion("the hesitation to respond?", "The reason is"),
    "the hesitation to respond",
  );
});

test("drops a connector left dangling by the token cap", () => {
  assert.equal(
    sanitizeThoughtCompletion(
      "need to be patient with pauses and understanding In",
      "when I pause I usually",
    ),
    "need to be patient with pauses and understanding",
  );
});

test("preserves a complete natural clause", () => {
  assert.equal(
    sanitizeThoughtCompletion("the pause before the voice catches up", "what feels wrong is"),
    "the pause before the voice catches up",
  );
});

test("drops a completion that copies a few-shot example", () => {
  const copied = COMPLETION_EXAMPLES[0].continuation;
  assert.equal(matchesCompletionExample(copied), true);
  assert.equal(sanitizeThoughtCompletion(copied, "the numbers only lined up after I"), "");
});

test("drops a completion that echoes a long run from an example", () => {
  const tail = COMPLETION_EXAMPLES[1].continuation.split(" ").slice(-7).join(" ");
  assert.equal(matchesCompletionExample(tail), true);
  assert.equal(sanitizeThoughtCompletion(tail, "we suddenly noticed the beam"), "");
});

test("keeps an original completion that merely shares a couple of words", () => {
  assert.equal(matchesCompletionExample("stopped worrying about the deadline"), false);
  assert.equal(
    sanitizeThoughtCompletion("stopped worrying about the deadline", "the plan came together once I"),
    "stopped worrying about the deadline",
  );
});

test("no example continuation is itself rejected by the format guard", () => {
  for (const example of COMPLETION_EXAMPLES) {
    assert.equal(matchesCompletionExample(example.continuation), true);
  }
});

test("trims a chatbot aside back to the genuine thought", () => {
  assert.equal(
    sanitizeThoughtCompletion("be more convincing, please respond with a question", "what I want is for the voice to"),
    "be more convincing",
  );
  assert.equal(
    sanitizeThoughtCompletion("keep circling the same worry, let me know if that helps", "the thing I avoid is"),
    "keep circling the same worry",
  );
});

test("cuts an AI mention or a bare please out of the thought", () => {
  assert.equal(
    sanitizeThoughtCompletion("be more specific, like a real person or an AI", "what I want is for the voice to"),
    "be more specific, like a real person",
  );
  assert.equal(
    sanitizeThoughtCompletion("I'm not following the agenda, please help", "the meeting fell apart because"),
    "I'm not following the agenda",
  );
});

test("trims a trailing open quote or ellipsis", () => {
  assert.equal(
    sanitizeThoughtCompletion('just refer to it as "', "I forgot the name so"),
    "just refer to it",
  );
  assert.equal(
    sanitizeThoughtCompletion("I'm not sure about it, but", "the honest answer is"),
    "I'm not sure about it",
  );
});

test("analyzeThoughtCompletion reports when nothing more can be kept", () => {
  const fragment = "the strange thing about hearing my voice is";
  const finished = (text) => analyzeThoughtCompletion(text, fragment).finished;
  assert.equal(finished(""), false);
  assert.equal(finished("how often"), false);
  assert.equal(finished("how often the answer arrives"), false);
  // A sentence end followed by more text: the sanitizer cuts there.
  assert.equal(finished("how often the answer arrives. Then"), true);
  assert.equal(
    analyzeThoughtCompletion("how often the answer arrives. Then", fragment).text,
    "how often the answer arrives.",
  );
  // A terminal mark with nothing after it yet closes the thought as well.
  assert.equal(finished("how often the answer arrives."), true);
  assert.equal(finished("how often the answer arrives!"), true);
  // A decimal or a numbered item may still continue.
  assert.equal(finished("about 3."), false);
  assert.equal(finished("about 3.5 times"), false);
  // A role label means the model moved on to the next turn.
  assert.equal(finished("how often it arrives User:"), true);
  assert.equal(finished("how often it arrives\nAssistant: more"), true);
  // A repeated boundary word with a period is cleaned, not treated as an end.
  assert.equal(finished("is. how often"), false);
  assert.equal(analyzeThoughtCompletion("is. how often", fragment).text, "how often");
  // Past the word cap (26, above the notebooks model's 24-word targets).
  const under = Array.from({ length: 24 }, (_, i) => `w${i}`).join(" ");
  assert.equal(finished(under), false);
  const long = Array.from({ length: 27 }, (_, i) => `w${i}`).join(" ");
  assert.equal(finished(long), true);
});
