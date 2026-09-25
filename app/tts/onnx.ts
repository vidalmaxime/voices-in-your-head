import { POCKET_MODEL_BASE, pocketModelUrls } from "./model-assets";
// ONNX-based Pocket TTS wrapper (ported from the pocket-tts-web demo).

import { assetUrl } from "../base-path";
import type { AudioPlayer } from "./audio";
import { parseWav, resampleAudio, SAMPLE_RATE } from "./audio";
import {
  MAX_VOICE_UPLOAD_SECONDS,
  prepareVoiceReference,
} from "./voice-reference";

type StatusHandler = (status: string) => void;
type ProgressHandler = (progress: number) => void;
type AudioHandler = (audio: ArrayBuffer | Float32Array) => void;
type ErrorHandler = (error: Error) => void;
/** Worker-side profile of a generation: timing marks with the first chunk, totals when finished. */
type MetricsHandler = (metrics: Record<string, unknown>, phase: "firstAudio" | "finished") => void;

type Deferred<T> = {
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
  promise: Promise<T>;
};

function createDeferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { resolve, reject, promise };
}

export type PocketTTSOnnxOptions = {
  baseUrl?: string;
  workerUrl?: string;
  onStatus?: StatusHandler;
  onProgress?: ProgressHandler;
  onAudio?: AudioHandler;
  onError?: ErrorHandler;
  onMetrics?: MetricsHandler;
  onLoadProfile?: (profile: Record<string, unknown>) => void;
};

export type PocketTTSOnnxSettings = {
  seed?: number;
  temperature?: number;
  speakerGain?: number;
  lsd?: number;
  /** Keep the conditioned voice state resident on the GPU between utterances (default true). */
  residentState?: boolean;
  /** Disable compact flow-state storage for a baseline comparison. */
  flowCacheFrames?: 512 | 1000;
};

export class PocketTTSOnnx {
  private baseUrl: string;
  private workerUrl: string;
  private onStatus?: StatusHandler;
  private onProgress?: ProgressHandler;
  private onAudio?: AudioHandler;
  private onError?: ErrorHandler;
  private onMetrics?: MetricsHandler;
  private onLoadProfile?: (profile: Record<string, unknown>) => void;

  private worker: Worker | null = null;
  private loadDeferred: Deferred<void> | null = null;
  private generateDeferred: Deferred<void> | null = null;
  private generationRequestId = 0;
  private encodeDeferred: Deferred<void> | null = null;
  private setVoiceDeferred: Deferred<void> | null = null;

  private currentVoice: string | Float32Array | null = null;
  private workerVoiceName: string | null = null;
  private modelsLoaded = false;
  private needsReload = false;

  private lsd = 1;
  private residentState = true;
  private flowCacheFrames: 512 | 1000 = 512;
  private seed = 0;
  private temperature = 0.9;
  private speakerGain = 1;

  private addPunctuations = true;
  private voiceList: string[] = [];
  private voicesLoaded = false;
  private voicesDeferred: Deferred<string[]> | null = null;

  // Dynamic LSD optimization state
  private edgeOptimizationApplied = false;
  private lastChunkTime = 0;
  private generationStartTime = 0;

  private modelUrls() {
    return pocketModelUrls(this.baseUrl);
  }

  constructor(options: PocketTTSOnnxOptions = {}) {
    this.baseUrl =
      options.baseUrl ?? POCKET_MODEL_BASE;
    this.workerUrl =
      options.workerUrl ?? assetUrl("/tts-onnx/inference-worker.js");
    this.onStatus = options.onStatus;
    this.onProgress = options.onProgress;
    this.onAudio = options.onAudio;
    this.onError = options.onError;
    this.onMetrics = options.onMetrics;
    this.onLoadProfile = options.onLoadProfile;
  }

  private handleMessage = (event: MessageEvent) => {
    const data = event.data ?? {};
    const type = data.type;
    switch (type) {
      case "load_profile":
        this.onLoadProfile?.(data.profile);
        break;
      case "status":
        if (typeof data.status === "string") this.onStatus?.(data.status);
        if (data.metrics && typeof data.metrics === "object") {
          this.onMetrics?.(data.metrics, "finished");
        }
        break;
      case "progress":
        if (typeof data.progress === "number") this.onProgress?.(data.progress);
        break;
      case "loaded":
        this.modelsLoaded = true;
        this.loadDeferred?.resolve();
        this.loadDeferred = null;
        break;
      case "voices_loaded":
        if (Array.isArray(data.voices)) {
          this.voiceList = data.voices.slice();
          this.voicesLoaded = true;
          if (this.voicesDeferred) {
            this.voicesDeferred.resolve(this.voiceList);
            this.voicesDeferred = null;
          }
        }
        break;
      case "audio_chunk":
        if (data.data) {
          if (data.metrics?.isFirst && data.metrics.timings) {
            this.onMetrics?.(data.metrics.timings, "firstAudio");
          }
          this.onAudio?.(data.data);

          // Dynamic LSD optimization: reduce LSD when running slower than realtime
          const metrics = data.metrics;
          if (metrics && !this.edgeOptimizationApplied) {
            const now = performance.now();
            let rtf = 0;

            // Calculate RTF from arrival time between chunks
            if (metrics.isFirst) {
              this.lastChunkTime = now;
            } else if (this.lastChunkTime > 0) {
              const timeSinceLastChunk = (now - this.lastChunkTime) / 1000;
              this.lastChunkTime = now;
              if (timeSinceLastChunk > 0 && metrics.chunkDuration) {
                rtf = metrics.chunkDuration / timeSinceLastChunk;
              }
            }

            // Also use genTimeSec if available (more accurate)
            if (
              metrics.genTimeSec &&
              metrics.genTimeSec > 0 &&
              metrics.chunkDuration
            ) {
              rtf = metrics.chunkDuration / metrics.genTimeSec;
            }

            // Apply edge optimization when RTF < 1.0 (slower than realtime)
            if (rtf > 0 && rtf < 1.0) {
              this.edgeOptimizationApplied = true;
              this.worker?.postMessage({ type: "set_lsd", data: { lsd: 1 } });
            }
          }
        }
        break;
      case "generation_complete":
        // Status/stream notifications are not request boundaries. A previous
        // track's trailing notifications can arrive after the next has started.
        if (data.requestId !== this.generationRequestId) break;
        if (data.error) {
          const error = new Error(String(data.error));
          this.generateDeferred?.reject(error);
          this.onError?.(error);
        } else {
          this.generateDeferred?.resolve();
        }
        this.generateDeferred = null;
        break;
      case "voice_encoded":
        this.encodeDeferred?.resolve();
        this.encodeDeferred = null;
        break;
      case "voice_set":
        this.workerVoiceName =
          typeof data.voiceName === "string" ? data.voiceName : this.workerVoiceName;
        this.setVoiceDeferred?.resolve();
        this.setVoiceDeferred = null;
        break;
      case "error": {
        const message =
          typeof data.error === "string" ? data.error : "Unknown worker error";
        const err = new Error(message);
        this.loadDeferred?.reject(err);
        this.generateDeferred?.reject(err);
        this.encodeDeferred?.reject(err);
        this.setVoiceDeferred?.reject(err);
        this.loadDeferred = null;
        this.generateDeferred = null;
        this.encodeDeferred = null;
        this.setVoiceDeferred = null;
        this.onError?.(err);
        break;
      }
      default:
        break;
    }
  };

  private async initWorker(): Promise<void> {
    if (this.worker && !this.needsReload) return;
    if (this.worker) {
      this.worker.removeEventListener("message", this.handleMessage);
      this.worker.terminate();
    }
    try {
      this.worker = new Worker(this.workerUrl, { type: "module" });
      this.worker.addEventListener("message", this.handleMessage);
      this.worker.addEventListener("error", (event) => {
        const detail =
          event.message || `Failed to load worker from ${this.workerUrl}`;
        const err = new Error(`TTS worker error: ${detail}`);
        console.error(err.message, {
          filename: event.filename,
          lineno: event.lineno,
        });
        this.loadDeferred?.reject(err);
        this.generateDeferred?.reject(err);
        this.encodeDeferred?.reject(err);
        this.setVoiceDeferred?.reject(err);
        this.loadDeferred = null;
        this.generateDeferred = null;
        this.encodeDeferred = null;
        this.setVoiceDeferred = null;
        this.needsReload = true;
        this.onError?.(err);
      });
      this.modelsLoaded = false;
      this.workerVoiceName = null;
      this.needsReload = false;
    } catch (err) {
      console.error("Failed to create TTS worker:", err);
      throw err;
    }
  }

  private async loadModels(): Promise<void> {
    if (!this.worker) throw new Error("Worker not initialized");
    if (this.modelsLoaded) return;
    if (this.loadDeferred) return this.loadDeferred.promise;

    this.loadDeferred = createDeferred<void>();
    const urls = this.modelUrls();

    this.worker.postMessage({
      type: "load",
      data: {
        urls,
        seed: this.seed,
        temperature: this.temperature,
        speakerGain: this.speakerGain,
        residentState: this.residentState,
        flowCacheFrames: this.flowCacheFrames,
      },
    });

    return this.loadDeferred.promise;
  }

  async initialize(settings: PocketTTSOnnxSettings = {}): Promise<void> {
    if (typeof settings.seed === "number") {
      if (settings.seed !== this.seed) {
        this.seed = settings.seed;
        this.needsReload = true;
      }
    }
    if (typeof settings.temperature === "number") {
      if (settings.temperature !== this.temperature) {
        this.temperature = settings.temperature;
        this.needsReload = true;
      }
    }
    if (typeof settings.speakerGain === "number") {
      if (settings.speakerGain !== this.speakerGain) {
        this.speakerGain = settings.speakerGain;
        this.needsReload = true;
      }
    }
    if (typeof settings.lsd === "number") {
      this.lsd = settings.lsd;
    }
    if (typeof settings.residentState === "boolean") {
      this.residentState = settings.residentState;
    }
    if (settings.flowCacheFrames && settings.flowCacheFrames !== this.flowCacheFrames) {
      this.flowCacheFrames = settings.flowCacheFrames;
      this.needsReload = true;
    }

    await this.initWorker();
    await this.loadModels();
    if (typeof settings.lsd === "number") {
      this.setLsd(settings.lsd);
    }
  }

  async loadVoices(): Promise<string[]> {
    if (this.voicesLoaded) {
      return this.voiceList.slice();
    }
    if (!this.voicesDeferred) {
      this.voicesDeferred = createDeferred<string[]>();
    }
    return this.voicesDeferred.promise;
  }

  setLsd(lsd: number): void {
    this.lsd = lsd;
    if (this.worker) {
      this.worker.postMessage({ type: "set_lsd", data: { lsd } });
    }
  }

  reportStatus(status: string): void {
    this.onStatus?.(status);
  }

  async setVoice(voice: string | Float32Array): Promise<void> {
    if (!this.worker) throw new Error("Worker not initialized");
    this.currentVoice = voice;
    if (typeof voice === "string") {
      if (this.workerVoiceName === voice) return;
      this.reportStatus(`Using ${voice} voice...`);
      this.setVoiceDeferred = createDeferred<void>();
      this.worker.postMessage({ type: "set_voice", data: { voiceName: voice } });
      return this.setVoiceDeferred.promise;
    }

    this.reportStatus("Using custom voice...");
    this.setVoiceDeferred = createDeferred<void>();
    this.worker.postMessage({ type: "set_voice", data: { voiceName: "custom" } });
    return this.setVoiceDeferred.promise;
  }

  async encodeVoice(blob: Blob): Promise<void> {
    if (!this.worker) throw new Error("Worker not initialized");
    const buffer = await blob.arrayBuffer();
    const { samples, sampleRate } = parseWav(buffer);
    const maxSourceSamples = Math.floor(
      MAX_VOICE_UPLOAD_SECONDS * sampleRate,
    );
    const source = samples.subarray(0, maxSourceSamples);
    const resampled = resampleAudio(source, sampleRate, SAMPLE_RATE);
    const audio = prepareVoiceReference(resampled, SAMPLE_RATE);

    this.encodeDeferred = createDeferred<void>();
    this.worker.postMessage({ type: "encode_voice", data: { audio } }, [
      audio.buffer,
    ]);
    return this.encodeDeferred.promise;
  }

  async generate(text: string, signal?: AbortSignal): Promise<void> {
    if (!this.worker) throw new Error("Worker not initialized");
    if (!this.modelsLoaded) throw new Error("Models not loaded");
    if (!this.currentVoice) throw new Error("Voice not set");

    if (this.generateDeferred) throw new Error("Speech generation is already running");

    // Reset dynamic LSD optimization state for new generation
    this.edgeOptimizationApplied = false;
    this.lastChunkTime = 0;
    this.generationStartTime = performance.now();

    if (this.addPunctuations) {
      let cleaned = text.trim();
      if (cleaned === "") throw new Error("Prompt cannot be empty");
      cleaned = cleaned.replace(/\s+/g, " ");
      const wordCount = cleaned.split(" ").length;
      let framesAfterEosGuess = 3;
      if (wordCount <= 4) {
        framesAfterEosGuess = 5;
      }
      cleaned = cleaned.replace(/^(\p{Ll})/u, (c) => c.toLocaleUpperCase());
      if (/[\p{L}\p{N}]$/u.test(cleaned)) {
        cleaned = cleaned + ".";
      }
      if (cleaned.split(" ").length < 5) {
        cleaned = " ".repeat(8) + cleaned;
      }
      if (framesAfterEosGuess > 0) {
        // Keep parity with the reference text normalization without changing logic.
      }
      text = cleaned;
    }

    if (signal?.aborted) return;

    const deferred = createDeferred<void>();
    this.generateDeferred = deferred;
    const requestId = ++this.generationRequestId;
    const voiceName =
      typeof this.currentVoice === "string" ? this.currentVoice : "custom";
    this.reportStatus("Starting worker generation...");
    this.worker.postMessage({ type: "generate", data: { text, voiceName, requestId } });

    const abortHandler = () => {
      this.stop();
    };
    signal?.addEventListener("abort", abortHandler);

    try {
      await deferred.promise;
    } finally {
      signal?.removeEventListener("abort", abortHandler);
      this.generateDeferred = null;
    }
  }

  stop(): void {
    if (!this.worker) return;
    this.worker.postMessage({ type: "stop" });
  }

  listVoices(): string[] {
    return this.voiceList.slice();
  }

  async close(): Promise<void> {
    if (this.worker) {
      this.worker.removeEventListener("message", this.handleMessage);
      this.worker.terminate();
      this.worker = null;
    }
    this.workerVoiceName = null;
  }
}

export async function playTTSOnnx(
  player: AudioPlayer,
  tts: PocketTTSOnnx,
  text: string,
  options: {
    voice: string | Float32Array;
    lsdDecodeSteps?: number;
    signal?: AbortSignal;
  },
): Promise<void> {
  const { voice, lsdDecodeSteps, signal } = options;

  if (typeof lsdDecodeSteps === "number") {
    tts.setLsd(lsdDecodeSteps);
  }

  tts.reportStatus("Starting audio output...");
  await tts.setVoice(voice);
  // The worker takes a few hundred milliseconds to reach the first chunk, so
  // the output device resumes alongside it rather than ahead of it.
  const resumed = player.resume();
  try {
    await tts.generate(text, signal);
  } finally {
    await resumed;
  }
}
