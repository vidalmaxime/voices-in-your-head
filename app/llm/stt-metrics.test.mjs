import assert from "node:assert/strict";
import test from "node:test";

import { cleanTranscript, normalizeTranscript, wordErrorRate } from "./stt-metrics.ts";

test("normalizes case, punctuation, and repeated whitespace", () => {
  assert.equal(normalizeTranscript("  Hello,   MAX!  "), "hello max");
});

test("removes adjacent CTC word repetitions", () => {
  assert.equal(
    cleanTranscript("I like the sea I I love my friends"),
    "I like the sea I love my friends",
  );
});

test("preserves repeated words when they are not adjacent", () => {
  assert.equal(cleanTranscript("I like what I like"), "I like what I like");
});

test("scores an exact transcript as zero error", () => {
  assert.equal(wordErrorRate("My name is Max.", "my name is max"), 0);
});

test("scores substitutions against reference word count", () => {
  assert.equal(wordErrorRate("I like sperm whales", "I like whispering whales"), 0.25);
});

test("scores insertions and deletions", () => {
  assert.equal(wordErrorRate("one two three", "one extra two"), 2 / 3);
});

test("handles empty reference text", () => {
  assert.equal(wordErrorRate("", ""), 0);
  assert.equal(wordErrorRate("", "unexpected"), 1);
});
