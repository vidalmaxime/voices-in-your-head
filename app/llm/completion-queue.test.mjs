import assert from "node:assert/strict";
import test from "node:test";
import { CompletionQueue } from "./completion-queue.ts";

function gate() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

test("cancelled speculation does no inference and acknowledges before its replacement", async () => {
  const queue = new CompletionQueue();
  const running = gate();
  const started = gate();
  const events = [];
  const warmup = queue.enqueueWarmup(async () => {
    started.resolve();
    await running.promise;
    events.push("warmup disposed");
  });
  await started.promise;
  const stale = queue.enqueue(async () => {
    assert.fail("cancelled inference must not start");
  }, () => events.push("stale complete"));
  queue.interrupt();
  const replacement = queue.enqueue(async () => events.push("replacement"));
  assert.deepEqual(events, []);
  running.resolve();
  await Promise.all([warmup, stale, replacement]);
  assert.deepEqual(events, ["warmup disposed", "stale complete", "replacement"]);
});

test("interrupting active inference preserves serialization through disposal", async () => {
  const queue = new CompletionQueue();
  const running = gate();
  const started = gate();
  const events = [];
  const first = queue.enqueue(async () => {
    started.resolve();
    await running.promise;
    events.push("disposed");
  }, () => assert.fail("active inference owns its terminal message"));
  await started.promise;
  queue.interrupt();
  const next = queue.enqueue(async () => events.push("next"));
  await Promise.resolve();
  assert.deepEqual(events, []);
  running.resolve();
  await Promise.all([first, next]);
  assert.deepEqual(events, ["disposed", "next"]);
});

test("warmups do not accumulate while the queue is occupied and resume when idle", async () => {
  const queue = new CompletionQueue();
  const running = gate();
  let warmups = 0;
  const first = queue.enqueue(async () => running.promise);
  await queue.enqueueWarmup(async () => { warmups++; });
  assert.equal(warmups, 0);
  running.resolve();
  await first;
  const warmup = queue.enqueueWarmup(async () => { warmups++; });
  await queue.enqueueWarmup(async () => { warmups++; });
  await warmup;
  assert.equal(warmups, 1);
});

test("inference failures do not block subsequent work or idle warmup", async () => {
  const queue = new CompletionQueue();
  const failed = queue.enqueue(async () => { throw new Error("inference failed"); });
  let completed = false;
  const next = queue.enqueue(async () => { completed = true; });
  await assert.rejects(failed, /inference failed/);
  await next;
  assert.equal(completed, true);
  let warmed = false;
  await queue.enqueueWarmup(async () => { warmed = true; });
  assert.equal(warmed, true);
});

test("repeated interrupts acknowledge each queued request exactly once", async () => {
  const queue = new CompletionQueue();
  const cancelled = [];
  const first = queue.enqueue(async () => assert.fail("stale"), () => cancelled.push(1));
  queue.interrupt();
  const second = queue.enqueue(async () => assert.fail("stale"), () => cancelled.push(2));
  queue.interrupt();
  await Promise.all([first, second]);
  assert.deepEqual(cancelled, [1, 2]);
});
