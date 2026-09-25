import assert from "node:assert/strict";
import test from "node:test";
import { createStaggeredChorus } from "./staggered-chorus.ts";
import { ChorusFeed } from "./chorus-feed.ts";

function harness() {
  const tracks = [];
  const ended = [];
  const player = createStaggeredChorus(index => {
    let end;
    const done = new Promise(resolve => { end = resolve; });
    const track = {
      index, chunks: [], flushed: false, aborted: false, closing: false,
      end, startedAt: index + 1, underrunCount: 0, underrunMs: 0,
      playChunk(chunk) { this.chunks.push(chunk); },
      flush() { this.flushed = true; },
      close() { this.closing = true; return done; },
      abort() { this.aborted = true; end(); },
      async resume() {},
    };
    tracks.push(track);
    return track;
  }, index => ended.push(index));
  return { player, tracks, ended };
}

test("starts each voice immediately while prior voices are still draining", async () => {
  const { player, tracks, ended } = harness();
  for (let i = 0; i < 3; i++) {
    player.playChunk(new Float32Array([i + 1]));
    assert.equal(tracks.length, i + 1);
    assert.deepEqual([...tracks[i].chunks[0]], [i + 1]);
    player.finishTrack();
    assert.equal(tracks[i].flushed, true);
    assert.equal(tracks[i].closing, true);
  }
  assert.deepEqual(ended, [], "all three voices can play concurrently");
  let closed = false;
  const done = player.close().then(() => { closed = true; });
  tracks[1].end();
  tracks[0].end();
  await Promise.resolve();
  assert.equal(closed, false, "microphone must stay paused until the final voice ends");
  tracks[2].end();
  await done;
  assert.deepEqual(ended, [1, 0, 2]);
});

test("abort stops every overlapping voice and drops subsequent chunks", async () => {
  const { player, tracks } = harness();
  player.playChunk(new Float32Array([1]));
  player.finishTrack();
  player.playChunk(new Float32Array([2]));
  player.abort();
  player.playChunk(new Float32Array([3]));
  await player.close();
  assert.ok(tracks.every(track => track.aborted));
  assert.equal(tracks.length, 2);
  assert.equal(tracks[1].chunks.length, 1);
});

test("a silent branch does not change the next branch's pan or highlight index", async () => {
  const { player, tracks } = harness();
  player.finishTrack();
  player.playChunk(new Float32Array([1]));
  assert.equal(tracks[0].index, 1);
  player.abort();
  await player.close();
});

test("branch feed delivers the first branch without waiting for later generation", async () => {
  const feed = new ChorusFeed();
  const reader = feed.read();
  feed.push("first");
  assert.deepEqual(await reader.next(), { value: "first", done: false });
  const next = reader.next();
  feed.push("second");
  assert.deepEqual(await next, { value: "second", done: false });
  feed.push("third");
  feed.finish();
  assert.deepEqual(await reader.next(), { value: "third", done: false });
  assert.equal((await reader.next()).done, true);
});

test("aborting a feed wakes a chorus waiting for another branch", async () => {
  const feed = new ChorusFeed();
  const controller = new AbortController();
  const reader = feed.read(controller.signal);
  const pending = reader.next();
  controller.abort();
  assert.equal((await pending).done, true);
});
