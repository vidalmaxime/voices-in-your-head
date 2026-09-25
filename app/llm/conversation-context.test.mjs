import assert from "node:assert/strict";
import test from "node:test";
import { appendContextTurn, MAX_CONTEXT_CHARACTERS, MAX_CONTEXT_TURNS } from "./conversation-context.ts";
import { buildCompletionMessages } from "./completion-logic.ts";

test("includes earlier speech and labels generated branches as alternatives", () => {
  const context = appendContextTurn([], { transcript: "I am moving to Paris", completions: ["next spring", "when my lease ends", "with my sister"] });
  const messages = buildCompletionMessages("I should start looking for", { context });
  assert.match(messages.at(-1).content, /Speaker said: "I am moving to Paris"/);
  assert.match(messages.at(-1).content, /Generated alternatives: \["next spring","when my lease ends","with my sister"\]/);
  assert.ok(messages.at(-1).content.endsWith("Text before cursor: I should start looking for\nText after cursor:"));
  assert.equal(messages.filter(message => message.content.includes("moving to Paris")).length, 1);
});

test("context also works for models without examples; absent context preserves original prompt", () => {
  const context = [{ transcript: "my sister lives in Paris", completions: [] }];
  const messages = buildCompletionMessages("I miss", { includeExamples: false, context });
  assert.equal(messages.length, 2);
  assert.match(messages[1].content, /my sister lives in Paris/);
  assert.equal(buildCompletionMessages("I miss").at(-1).content, "Text before cursor: I miss\nText after cursor:");
});

test("retains recent turns without mutating an in-flight request snapshot", () => {
  const original = [{ transcript: "first", completions: [] }];
  let history = original;
  for (let i = 0; i < 20; i++) history = appendContextTurn(history, { transcript: `turn ${i}`, completions: ["thought"] });
  assert.equal(history.length, MAX_CONTEXT_TURNS);
  assert.equal(history.at(-1).transcript, "turn 19");
  assert.deepEqual(original, [{ transcript: "first", completions: [] }]);
});

test("bounds long speech and discards empty transcripts", () => {
  let history = [];
  for (let i = 0; i < 10; i++) history = appendContextTurn(history, { transcript: "x".repeat(10000), completions: Array(3).fill("y".repeat(3000)) });
  const characters = history.reduce((sum, turn) => sum + turn.transcript.length + turn.completions.join("").length, 0);
  assert.ok(characters <= MAX_CONTEXT_CHARACTERS);
  assert.deepEqual(appendContextTurn(history, { transcript: "  ", completions: [] }), history);
});
