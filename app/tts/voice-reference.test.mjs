import assert from "node:assert/strict";
import test from "node:test";

import {
  MAX_VOICE_REFERENCE_SECONDS,
  prepareVoiceReference,
} from "./voice-reference.ts";

const SAMPLE_RATE = 1000;

function sine(seconds, amplitude, dcOffset = 0) {
  return Float32Array.from(
    { length: seconds * SAMPLE_RATE },
    (_, i) => dcOffset + amplitude * Math.sin((2 * Math.PI * 7 * i) / SAMPLE_RATE),
  );
}

test("preserves natural reference amplitude instead of peak normalizing", () => {
  const prepared = prepareVoiceReference(sine(2, 0.1), SAMPLE_RATE);
  const peak = prepared.reduce((value, sample) => Math.max(value, Math.abs(sample)), 0);

  assert.ok(peak > 0.09 && peak < 0.11, `unexpected peak ${peak}`);
});

test("removes DC offset", () => {
  const prepared = prepareVoiceReference(sine(2, 0.1, 0.2), SAMPLE_RATE);
  const mean = prepared.reduce((sum, sample) => sum + sample, 0) / prepared.length;

  assert.ok(Math.abs(mean) < 1e-5, `unexpected mean ${mean}`);
});

test("trims leading and trailing silence while retaining padding", () => {
  const samples = new Float32Array(4 * SAMPLE_RATE);
  samples.set(sine(2, 0.1), SAMPLE_RATE);

  const prepared = prepareVoiceReference(samples, SAMPLE_RATE);
  assert.ok(prepared.length >= 2.4 * SAMPLE_RATE);
  assert.ok(prepared.length <= 2.6 * SAMPLE_RATE);
});

test("caps long references before encoding", () => {
  const prepared = prepareVoiceReference(
    sine(MAX_VOICE_REFERENCE_SECONDS + 5, 0.1),
    SAMPLE_RATE,
  );

  assert.equal(prepared.length, MAX_VOICE_REFERENCE_SECONDS * SAMPLE_RATE);
});

test("finds speech after long leading silence before applying the encoder cap", () => {
  const samples = new Float32Array(25 * SAMPLE_RATE);
  samples.set(sine(5, 0.1), 20 * SAMPLE_RATE);

  const prepared = prepareVoiceReference(samples, SAMPLE_RATE);
  assert.ok(prepared.length >= 5 * SAMPLE_RATE);
  assert.ok(prepared.length <= 5.3 * SAMPLE_RATE);
});

test("rejects silent references", () => {
  assert.throws(
    () => prepareVoiceReference(new Float32Array(2 * SAMPLE_RATE), SAMPLE_RATE),
    /silent or too quiet/,
  );
});
