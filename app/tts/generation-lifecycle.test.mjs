import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";
import { mixSpeechTracks } from "../llm/chorus-audio.ts";

// Load the actual browser wrapper with its local imports resolved for Node.
const sourceUrl = new URL("./onnx.ts", import.meta.url);
const source = (await readFile(sourceUrl, "utf8")).replace(
  /from "(\.[^"]+)"/g,
  (_, path) => `from "${new URL(`${path}.ts`, sourceUrl).href}"`,
);
const compiled = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 },
}).outputText;
const { PocketTTSOnnx } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`);
const tick = () => new Promise(resolve => setImmediate(resolve));

function harness(onAudio = () => {}) {
  const requests = [];
  const tts = new PocketTTSOnnx({ onAudio });
  tts.worker = { postMessage: message => requests.push(message) };
  tts.modelsLoaded = true;
  tts.currentVoice = "test";
  return { tts, requests, emit: data => tts.handleMessage({ data }) };
}

test("Chorus waits for all three actual audio tracks despite trailing finish messages", async () => {
  let captured = [];
  const tracks = [];
  const { tts, requests, emit } = harness(audio => captured.push(new Float32Array(audio)));
  const run = (async () => {
    for (const text of ["one", "two", "three"]) {
      captured = [];
      await tts.generate(text);
      tracks.push(captured);
    }
    return mixSpeechTracks(tracks);
  })();

  for (let i = 0; i < 3; i++) {
    const request = requests.filter(message => message.type === "generate")[i];
    assert.ok(request, `track ${i + 1} was requested`);
    // These are the notifications the real worker emits before its final ack.
    emit({ type: "status", state: "idle", status: "Finished (RTFx: 2)" });
    emit({ type: "stream_ended" });
    emit({ type: "status", state: "idle", status: "Finished" });
    if (i > 0) emit({ type: "generation_complete", requestId: request.data.requestId - 1 });
    await tick();
    assert.equal(tracks.length, i, "status messages must not complete a track");
    emit({ type: "audio_chunk", data: new Float32Array([0, (i + 1) * 0.15, 0]) });
    emit({ type: "generation_complete", requestId: request.data.requestId });
    await tick();
    assert.equal(tracks.length, i + 1);
  }
  const mixed = await run;
  assert.equal(tracks.length, 3);
  assert.ok(tracks.every(track => track.length === 1));
  assert.ok(Math.abs(mixed[1] - 0.3) < 1e-6, "all voices contribute to the same sample");
});

test("cancellation waits for worker cleanup before allowing the next generation", async () => {
  const { tts, requests, emit } = harness();
  const controller = new AbortController();
  let done = false;
  const run = tts.generate("first", controller.signal).then(() => { done = true; });
  controller.abort();
  assert.equal(requests.at(-1).type, "stop");
  emit({ type: "status", state: "idle", status: "Stopped" });
  await tick();
  assert.equal(done, false);
  await assert.rejects(tts.generate("too soon"), /already running/);
  emit({ type: "generation_complete", requestId: requests[0].data.requestId });
  await run;
});

test("a failed track rejects its request", async () => {
  const { tts, requests, emit } = harness();
  const run = tts.generate("first");
  const rejection = assert.rejects(run, /synthesis failed/);
  emit({ type: "generation_complete", requestId: requests[0].data.requestId, error: "synthesis failed" });
  await rejection;
});

const workerSource = await readFile(new URL("../../public/tts-onnx/inference-worker.js", import.meta.url), "utf8");
const workerFunction = workerSource.slice(workerSource.indexOf("async function startGeneration("), workerSource.indexOf("function setFlowCacheFrames"));
for (const outcome of ["success", "cancel", "error"]) {
  test(`worker acknowledges ${outcome} only after the generation pipeline exits`, async () => {
    const events = [];
    let release;
    const gate = new Promise(resolve => { release = resolve; });
    const context = vm.createContext({
      isGenerating: false, generationTimings: {}, generationStartedAt: 0,
      performance, console: { log() {}, error() {} },
      postMessage: message => events.push(message),
      splitIntoBestSentences: text => [text],
      currentVoiceEmbedding: {}, currentVoiceName: "test", markTiming() {},
      runGenerationPipeline: async () => {
        await gate;
        if (outcome === "error") throw new Error("failed");
      },
    });
    vm.runInContext(workerFunction, context);
    const run = context.startGeneration("hello", "test", 42);
    assert.equal(events.some(event => event.type === "generation_complete"), false);
    if (outcome === "cancel") context.isGenerating = false;
    release();
    await run;
    assert.equal(context.isGenerating, false);
    assert.equal(events.filter(event => event.type === "generation_complete").length, 1);
    assert.equal(events.at(-1).type, "generation_complete");
    assert.equal(events.at(-1).requestId, 42);
    assert.equal(Boolean(events.at(-1).error), outcome === "error");
  });
}
