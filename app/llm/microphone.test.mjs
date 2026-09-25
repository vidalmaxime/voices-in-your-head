import assert from "node:assert/strict";
import test from "node:test";

import { requestMicrophoneStream } from "./microphone.ts";

test("returns a microphone stream before the deadline", async () => {
  const stream = { getTracks: () => [] };
  assert.equal(await requestMicrophoneStream(async () => stream, 20), stream);
});

test("times out and stops a stream that resolves late", async () => {
  let resolveStream;
  let stopped = false;
  const pending = new Promise((resolve) => {
    resolveStream = resolve;
  });

  await assert.rejects(
    requestMicrophoneStream(() => pending, 1),
    /Microphone permission timed out/,
  );

  resolveStream({
    getTracks: () => [{ stop: () => { stopped = true; } }],
  });
  await Promise.resolve();
  assert.equal(stopped, true);
});
