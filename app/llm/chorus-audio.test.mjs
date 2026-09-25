import assert from "node:assert/strict";
import test from "node:test";
import { mixSpeechTracks } from "./chorus-audio.ts";

const pcm = values => new Float32Array(values);

test("aligns all three voices at sample zero across chunk boundaries and preserves tails", () => {
  const mixed = mixSpeechTracks([
    [pcm([0.3]), pcm([0.6, 0.9])],
    [pcm([0.6, 0.3])],
    [pcm([0.9]), pcm([0.9, 0.3, 0.6])],
  ]);
  const expected = [0.6, 0.6, 0.4, 0.2];
  assert.equal(mixed.length, expected.length);
  expected.forEach((value, i) => assert.ok(Math.abs(mixed[i] - value) < 1e-6));
});

test("three full-scale voices mix without clipping", () => {
  assert.deepEqual([...mixSpeechTracks(Array.from({ length: 3 }, () => [pcm([1, -1])]))], [1, -1]);
});

test("handles missing audio without attenuating the remaining voice", () => {
  assert.deepEqual([...mixSpeechTracks([[], [pcm([0.5])], []])], [0.5]);
  assert.equal(mixSpeechTracks([]).length, 0);
});
