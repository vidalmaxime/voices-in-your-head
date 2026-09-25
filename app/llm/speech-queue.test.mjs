import assert from "node:assert/strict";
import test from "node:test";

import { SpeechQueue } from "./speech-queue.ts";

test("speaks branch tasks sequentially in enqueue order", async () => {
  const queue = new SpeechQueue();
  const events = [];
  let releaseFirst;
  let markFirstStarted;
  const firstGate = new Promise((resolve) => {
    releaseFirst = resolve;
  });
  const firstStarted = new Promise((resolve) => {
    markFirstStarted = resolve;
  });

  const first = queue.enqueue(async () => {
    events.push("first:start");
    markFirstStarted();
    await firstGate;
    events.push("first:end");
  });
  const second = queue.enqueue(async () => {
    events.push("second:start");
    events.push("second:end");
  });

  await firstStarted;
  assert.deepEqual(events, ["first:start"]);
  releaseFirst();
  await Promise.all([first, second]);
  assert.deepEqual(events, [
    "first:start",
    "first:end",
    "second:start",
    "second:end",
  ]);
});

test("continues with later branches after one task fails", async () => {
  const queue = new SpeechQueue();
  const events = [];

  const failed = queue.enqueue(async () => {
    events.push("failed");
    throw new Error("synthesis failed");
  });
  const next = queue.enqueue(async () => {
    events.push("next");
  });

  await assert.rejects(failed, /synthesis failed/);
  await next;
  assert.deepEqual(events, ["failed", "next"]);
});
