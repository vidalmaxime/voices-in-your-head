import { createModelLoadProfile, prefetchModelFiles } from "./model-loading";
import { pipeline, env } from "@huggingface/transformers";

import { cleanTranscript } from "./stt-metrics";

const MODEL_ID = "onnx-community/parakeet-ctc-0.6b-ONNX";
const SAMPLE_RATE = 16000;
const WARMUP_SAMPLES = SAMPLE_RATE / 2;
const PREFERRED_DEVICE = "webgpu";
const FALLBACK_DEVICE = "wasm";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyASRPipeline = any;
type ASRDevice = typeof PREFERRED_DEVICE | typeof FALLBACK_DEVICE;

type ASROutput = {
  text?: string;
};

class AutomaticSpeechRecognitionPipeline {
  static transcriber: Promise<AnyASRPipeline> | null = null;

  static async getInstance(progress_callback?: (x: unknown) => void) {
    this.transcriber ??= this.loadWithFallback(progress_callback).catch((err) => {
      this.transcriber = null;
      throw err;
    });

    return this.transcriber;
  }

  private static async loadWithFallback(progress_callback?: (x: unknown) => void) {
    try {
      return await this.load(PREFERRED_DEVICE, progress_callback);
    } catch (err) {
      console.warn("Speech WebGPU load failed; retrying on WASM.", err);
      self.postMessage({
        status: "loading",
        data: "Speech WebGPU failed; retrying speech model on CPU...",
      });
      return this.load(FALLBACK_DEVICE, progress_callback);
    }
  }

  private static load(device: ASRDevice, progress_callback?: (x: unknown) => void) {
    return pipeline(
      "automatic-speech-recognition",
      MODEL_ID,
      {
        dtype: "q4",
        device,
        progress_callback,
      } as Record<string, unknown>,
    );
  }
}

const SPEECH_FILES = [
  "config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
  "onnx/model_q4.onnx", "onnx/model_q4.onnx_data",
];
let prefetchPromise: ReturnType<typeof prefetchModelFiles> | null = null;
function prefetch() {
  if (loadPromise || prefetchPromise || !env.useBrowserCache || env.useCustomCache || env.experimental_useCrossOriginStorage) return;
  const base = `${env.remoteHost}${env.remotePathTemplate.replace("{model}", MODEL_ID).replace("{revision}", "main")}`;
  prefetchPromise = prefetchModelFiles(env.cacheKey, SPEECH_FILES.map(file => `${base}${file}`), AbortSignal.timeout(300_000));
}

let processing = false;
let loadPromise: Promise<void> | null = null;

function readText(output: ASROutput | ASROutput[]) {
  if (Array.isArray(output)) {
    return output[0]?.text ?? "";
  }

  return output.text ?? "";
}

async function transcribe(audio: Float32Array, progress_callback?: (x: unknown) => void) {
  const transcriber = await AutomaticSpeechRecognitionPipeline.getInstance(progress_callback);
  const output = await transcriber(audio);

  return cleanTranscript(readText(output));
}

async function generate({
  audio,
}: {
  audio: Float32Array;
  language: string;
}) {
  if (processing) return;
  processing = true;

  try {
    self.postMessage({ status: "start" });

    const output = await transcribe(audio);

    self.postMessage({
      status: "complete",
      output,
    });
  } catch (err) {
    self.postMessage({
      status: "error",
      data: err instanceof Error ? err.message : String(err),
    });
  } finally {
    processing = false;
  }
}

// The speculative pipeline sends a pass per transcript refresh. Only the
// newest one is worth running once the current pass finishes, so a single
// pending slot replaces queueing.
let pendingPartial: { audio: Float32Array; requestId?: number } | null = null;

async function generatePartial({
  audio,
  requestId,
}: {
  audio: Float32Array;
  language: string;
  requestId?: number;
}) {
  if (processing) {
    if (pendingPartial?.requestId !== undefined) {
      self.postMessage({ status: "partialSuperseded", requestId: pendingPartial.requestId });
    }
    pendingPartial = { audio, requestId };
    return;
  }
  processing = true;

  try {
    const output = await transcribe(audio);

    self.postMessage({
      status: "partialComplete",
      output,
      requestId,
    });
  } catch (err) {
    self.postMessage({
      status: "error",
      data: err instanceof Error ? err.message : String(err),
      requestId,
    });
  } finally {
    processing = false;
  }

  const next = pendingPartial;
  pendingPartial = null;
  if (next) {
    void generatePartial({ audio: next.audio, language: "en", requestId: next.requestId });
  }
}

// A short silent pass so the recognizer stays hot while the app is idle; it
// yields to any transcription that arrives meanwhile.
async function keepWarm() {
  if (processing || !loadPromise) return;
  processing = true;
  try {
    const transcriber = await AutomaticSpeechRecognitionPipeline.getInstance();
    await transcriber(new Float32Array(WARMUP_SAMPLES));
  } catch {
    // Warmth is best-effort.
  } finally {
    processing = false;
  }
  const next = pendingPartial;
  pendingPartial = null;
  if (next) {
    void generatePartial({ audio: next.audio, language: "en", requestId: next.requestId });
  }
}

async function loadOnce() {
  if (loadPromise) return loadPromise;

  loadPromise = (async () => {
    self.postMessage({
      status: "loading",
      data: "Loading speech model...",
    });

    const prefetchWaitStartedAt = performance.now();
    const prefetchResult = await prefetchPromise;
    const prefetchWaitMs = performance.now() - prefetchWaitStartedAt;
    const profile = createModelLoadProfile(
      ["onnx/model_q4.onnx", "onnx/model_q4.onnx_data"],
      () => self.postMessage({ status: "weights-ready" }),
    );
    const transcriber = await AutomaticSpeechRecognitionPipeline.getInstance((x) => {
      profile.progress(x);
      self.postMessage(x);
    });
    profile.initialized();

    self.postMessage({
      status: "loading",
      data: "Warming speech recognition...",
    });
    await transcriber(new Float32Array(WARMUP_SAMPLES));

    self.postMessage({ status: "ready", loadProfile: { ...profile.finish(), prefetchWaitMs, prefetch: prefetchResult } });
  })().catch((err) => {
    loadPromise = null;
    self.postMessage({
      status: "error",
      data: err instanceof Error ? err.message : String(err),
    });
  });

  return loadPromise;
}

export interface WhisperWorkerMessage {
  type: "prefetch" | "load" | "generate" | "generatePartial" | "keepWarm";
  data?: {
    audio: Float32Array;
    language: string;
    /** Echoed on the result so the page can route speculative passes. */
    requestId?: number;
  };
}

self.addEventListener("message", async (e: MessageEvent<WhisperWorkerMessage>) => {
  const { type, data } = e.data;

  switch (type) {
    case "prefetch":
      prefetch();
      break;
    case "load":
      loadOnce();
      break;

    case "generate":
      if (data) {
        generate(data);
      }
      break;

    case "generatePartial":
      if (data) {
        generatePartial(data);
      }
      break;

    case "keepWarm":
      void keepWarm();
      break;
  }
});
