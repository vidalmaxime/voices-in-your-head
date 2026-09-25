import assert from 'node:assert/strict';
import test from 'node:test';
import { createModelLoadProfile, prefetchModelFiles } from './model-loading.ts';
import { pocketModelUrls, pocketPrefetchUrls } from '../tts/model-assets.ts';

const signal = () => new AbortController().signal;

test('cached files never download; misses use exactly the loader cache keys', async () => {
  const requests = [], saved = [];
  const cache = {
    match: async url => url === 'https://model/cached' ? new Response('cached') : undefined,
    put: async (url, response) => saved.push([url, await response.text()]),
  };
  const result = await prefetchModelFiles('test-cache', ['https://model/cached', 'https://model/missing'], signal(), {
    open: async name => { assert.equal(name, 'test-cache'); return cache; },
  }, async url => { requests.push(url); return new Response('weights'); });
  assert.deepEqual(requests, ['https://model/missing']);
  assert.deepEqual(saved, [['https://model/missing', 'weights']]);
  assert.equal(result.cachedFiles, 1);
  assert.equal(result.downloadedFiles, 1);
  assert.equal(result.skipped, false);
});

test('next download and initialization wait for the current cache write', async () => {
  const events = [];
  let finishWrite;
  const writeGate = new Promise(resolve => { finishWrite = resolve; });
  let writeStarted;
  const started = new Promise(resolve => { writeStarted = resolve; });
  const prefetch = prefetchModelFiles('test', ['first', 'second'], signal(), {
    open: async () => ({ match: async () => undefined, put: async url => {
      events.push(`write:${url}`);
      if (url === 'first') { writeStarted(); await writeGate; }
    } }),
  }, async url => { events.push(`fetch:${url}`); return new Response('weights'); });
  const load = prefetch.then(() => events.push('initialize'));
  await started;
  assert.deepEqual(events, ['fetch:first', 'write:first']);
  finishWrite();
  await load;
  assert.deepEqual(events, ['fetch:first', 'write:first', 'fetch:second', 'write:second', 'initialize']);
});

test('unavailable cache skips downloading and lets the normal loader take over', async () => {
  let fetches = 0;
  const result = await prefetchModelFiles('test', ['model'], signal(), {
    open: async () => { throw new Error('cache blocked'); },
  }, async () => { fetches++; return new Response('weights'); });
  assert.equal(result.skipped, true);
  assert.equal(fetches, 0);
});

test('HTTP errors and quota failures are recoverable, and stop the prefetch chain', async () => {
  for (const failure of ['http', 'quota']) {
    let fetches = 0;
    const result = await prefetchModelFiles('test', ['first', 'second'], signal(), {
      open: async () => ({ match: async () => undefined, put: async () => { throw new Error('quota'); } }),
    }, async () => { fetches++; return new Response('', { status: failure === 'http' ? 503 : 200 }); });
    assert.equal(result.skipped, true);
    assert.equal(fetches, 1);
  }
});

test('cancellation prevents later files from downloading', async () => {
  const controller = new AbortController();
  let fetches = 0;
  const result = await prefetchModelFiles('test', ['first', 'second'], controller.signal, {
    open: async () => ({ match: async () => undefined, put: async () => controller.abort() }),
  }, async () => { fetches++; return new Response('weights'); });
  assert.equal(result.skipped, true);
  assert.equal(fetches, 1);
});

test('overlap starts only after both graph and external weights finish, once', () => {
  let time = 0, starts = 0;
  const profile = createModelLoadProfile(['model.onnx', 'model.onnx_data'], () => starts++, () => time);
  time = 10; profile.progress({ status: 'done', file: 'model.onnx' });
  time = 15; profile.progress({ status: 'progress', file: 'model.onnx_data' });
  assert.equal(starts, 0);
  time = 20; profile.progress({ status: 'done', file: 'model.onnx_data' });
  assert.equal(starts, 1);
  time = 25; profile.initialized();
  time = 30;
  assert.equal(starts, 1);
  assert.deepEqual(profile.finish(), { filePhaseMs: 20, initializationTailMs: 5, warmupMs: 5, totalMs: 30 });
});

test('unknown model layouts defer prefetch until initialization finishes', () => {
  let starts = 0;
  const profile = createModelLoadProfile(['unknown'], () => starts++);
  profile.progress({ status: 'done', file: 'different' });
  assert.equal(starts, 0);
  profile.initialized(); profile.initialized();
  assert.equal(starts, 1);
});

test('Pocket prefetch shares exact loader URLs and leaves voice assets lazy', () => {
  const urls = pocketModelUrls();
  const prefetched = pocketPrefetchUrls();
  for (const key of ['text_conditioner', 'flow_lm_main', 'flow_lm_flow', 'mimi_decoder', 'tokenizer', 'bundle']) {
    assert.ok(prefetched.includes(urls[key]));
  }
  assert.ok(!prefetched.includes(urls.mimi_encoder));
  assert.ok(!prefetched.includes(urls.voices));
});
