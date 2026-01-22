"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { cachedFetch, safetensors, tokenizers } from "@jax-js/loaders";
import { Loader2, Mic, MicOff, Volume2, VolumeX } from "lucide-react";
import { MicVAD } from "@ricky0123/vad-web";

import DownloadManager, { DownloadManagerHandle } from "../tts/DownloadManager";
import { createStreamingPlayer, parseWav, resampleAudio, SAMPLE_RATE } from "../tts/audio";
import { playTTS } from "../tts/inference";
import { fromSafetensors, runMimiEncode, type PocketTTS } from "../tts/pocket-tts";

interface WorkerResponse {
  status: string;
  data?: string;
  output?: string;
  tps?: number;
  numTokens?: number;
  state?: "thinking" | "answering";
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
}

interface ProgressItem {
  file: string;
  progress: number;
  total: number;
}

// TTS caching
let _weights: safetensors.File | null = null;
let _model: PocketTTS | null = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _tokenizer: any | null = null;

const HF_URL_PREFIX =
  "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/fbf8280";

const predefinedVoices: Record<string, string> = {
  alba: HF_URL_PREFIX + `/embeddings/alba.safetensors`,
  azelma: HF_URL_PREFIX + `/embeddings/azelma.safetensors`,
  cosette: HF_URL_PREFIX + `/embeddings/cosette.safetensors`,
  eponine: HF_URL_PREFIX + `/embeddings/eponine.safetensors`,
  fantine: HF_URL_PREFIX + `/embeddings/fantine.safetensors`,
  javert: HF_URL_PREFIX + `/embeddings/javert.safetensors`,
  jean: HF_URL_PREFIX + `/embeddings/jean.safetensors`,
  marius: HF_URL_PREFIX + `/embeddings/marius.safetensors`,
};

function parseSafetensorsWithBF16(data: ArrayBuffer): safetensors.File {
  const view = new DataView(data);
  const headerLen = Number(view.getBigUint64(0, true));
  const headerBytes = new Uint8Array(data, 8, headerLen);
  const headerStr = new TextDecoder().decode(headerBytes);
  const header = JSON.parse(headerStr) as Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>;

  const dataOffset = 8 + headerLen;
  const tensors: Record<string, { dtype: string; shape: number[]; data: Uint8Array | Float16Array }> = {};

  for (const [name, meta] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const [start, end] = meta.data_offsets;
    const tensorData = new Uint8Array(data, dataOffset + start, end - start);

    if (meta.dtype === "BF16") {
      const bf16 = new Uint16Array(tensorData.buffer, tensorData.byteOffset, tensorData.byteLength / 2);
      const fp16 = new Float16Array(bf16.length);
      const f32 = new Float32Array(1);
      const u32 = new Uint32Array(f32.buffer);
      for (let i = 0; i < bf16.length; i++) {
        u32[0] = bf16[i] << 16;
        fp16[i] = f32[0];
      }
      tensors[name] = { dtype: "F16", shape: meta.shape, data: fp16 };
    } else if (meta.dtype === "F16") {
      tensors[name] = { dtype: "F16", shape: meta.shape, data: new Float16Array(tensorData.buffer, tensorData.byteOffset, tensorData.byteLength / 2) };
    } else {
      tensors[name] = { dtype: meta.dtype, shape: meta.shape, data: tensorData };
    }
  }

  return { tensors } as safetensors.File;
}

function prepareTextPrompt(text: string): [string, number] {
  text = text.trim();
  if (text === "") throw new Error("Prompt cannot be empty");
  text = text.replace(/\s+/g, " ");
  const numberOfWords = text.split(" ").length;
  let framesAfterEosGuess = 3;
  if (numberOfWords <= 4) {
    framesAfterEosGuess = 5;
  }
  text = text.replace(/^(\p{Ll})/u, (c) => c.toLocaleUpperCase());
  if (/[\p{L}\p{N}]$/u.test(text)) {
    text = text + ".";
  }
  if (text.split(" ").length < 5) {
    text = " ".repeat(8) + text;
  }
  return [text, framesAfterEosGuess];
}

export default function LLMPage() {
  const downloadManagerRef = useRef<DownloadManagerHandle>(null);
  const workerRef = useRef<Worker | null>(null);
  const whisperWorkerRef = useRef<Worker | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Status
  const [llmStatus, setLlmStatus] = useState<"idle" | "loading" | "ready">("idle");
  const [sttStatus, setSttStatus] = useState<"idle" | "loading" | "ready">("idle");
  const [loadingMessage, setLoadingMessage] = useState("");
  const [progressItems, setProgressItems] = useState<ProgressItem[]>([]);
  const [sttProgressItems, setSttProgressItems] = useState<ProgressItem[]>([]);

  // Display state - the main text shown
  const [transcribedText, setTranscribedText] = useState<string | null>(null);
  const [completionText, setCompletionText] = useState<string | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);

  // TTS state
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [ttsLoading, setTtsLoading] = useState(false);
  const [selectedVoice, setSelectedVoice] = useState("jean");
  const [customVoiceFile, setCustomVoiceFile] = useState<File | null>(null);
  const selectedVoiceRef = useRef(selectedVoice);
  const customVoiceFileRef = useRef(customVoiceFile);

  // Keep refs in sync
  useEffect(() => {
    selectedVoiceRef.current = selectedVoice;
  }, [selectedVoice]);
  useEffect(() => {
    customVoiceFileRef.current = customVoiceFile;
  }, [customVoiceFile]);

  // VAD state
  const vadRef = useRef<MicVAD | null>(null);
  const [vadLoading, setVadLoading] = useState(false);
  const [vadError, setVadError] = useState<string | null>(null);
  const [micEnabled, setMicEnabled] = useState(false);
  const [userSpeaking, setUserSpeaking] = useState(false);

  // Thought completion state
  const pauseFrameCountRef = useRef(0);
  const audioBufferRef = useRef<Float32Array[]>([]);
  const isCompletingRef = useRef(false);
  const speechStartTimeRef = useRef<number | null>(null);
  const lastSpeechTimeRef = useRef<number | null>(null);
  const currentPlayerRef = useRef<ReturnType<typeof createStreamingPlayer> | null>(null);
  const completionAbortControllerRef = useRef<AbortController | null>(null);
  const tokenBufferRef = useRef<string>("");
  const interruptCompletionRef = useRef<() => void>(() => {});
  const triggerThoughtCompletionRef = useRef<() => void>(() => {});

  // Thresholds
  const PAUSE_THRESHOLD = 0.3;
  const PAUSE_DURATION_MS = 400;
  const MIN_SPEECH_BEFORE_COMPLETION_MS = 500;
  const FRAME_DURATION_MS = 96;

  const [isWebGPUAvailable, setIsWebGPUAvailable] = useState<boolean | null>(null);

  useEffect(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    setIsWebGPUAvailable(!!(navigator as any).gpu);
  }, []);

  async function downloadTTSWeights(): Promise<safetensors.File> {
    if (_weights) return _weights;
    const weightsUrl =
      "https://huggingface.co/kyutai/pocket-tts/resolve/main/tts_b6369a24.safetensors";

    if (!downloadManagerRef.current) {
      throw new Error("Download manager not initialized");
    }

    const data = await downloadManagerRef.current.fetch("TTS model", weightsUrl, {
      Authorization: "Bearer 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",
    });
    const result = parseSafetensorsWithBF16(data);
    _weights = result;
    return result;
  }

  async function getTTSModel(): Promise<PocketTTS> {
    if (_model) return _model;
    const weights = await downloadTTSWeights();
    _model = fromSafetensors(weights);
    return _model;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async function getTTSTokenizer(): Promise<any> {
    if (!_tokenizer) {
      _tokenizer = await tokenizers.loadSentencePiece(
        "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/fbf8280/tokenizer.model"
      );
    }
    return _tokenizer;
  }

  async function speakText(text: string, signal?: AbortSignal) {
    if (!ttsEnabled || !text.trim()) return;
    const voice = selectedVoiceRef.current;
    const customFile = customVoiceFileRef.current;
    if (voice === "custom" && !customFile) return;

    setIsSpeaking(true);
    try {
      const devices = await init();
      if (devices.includes("webgpu")) {
        defaultDevice("webgpu");
      } else {
        console.warn("WebGPU not supported for TTS");
        return;
      }

      setTtsLoading(true);
      const model = await getTTSModel();
      const tokenizer = await getTTSTokenizer();
      setTtsLoading(false);

      if (signal?.aborted) return;

      const [preparedText, framesAfterEos] = prepareTextPrompt(text);
      const tokens = tokenizer.encode(preparedText);

      let voiceEmbed: np.Array;

      if (voice === "custom" && customFile) {
        console.log("Processing custom voice file...");
        const arrayBuffer = await customFile.arrayBuffer();
        const { samples, sampleRate } = parseWav(arrayBuffer);
        const resampled = resampleAudio(samples, sampleRate, SAMPLE_RATE);
        let maxAbs = 0;
        for (let i = 0; i < resampled.length; i++) {
          const value = Math.abs(resampled[i]);
          if (value > maxAbs) maxAbs = value;
        }
        const normalized = maxAbs > 0 ? resampled.map((value) => value / maxAbs) : resampled;
        const audioTensor = np.array(normalized, {
          dtype: np.float32,
          shape: [1, normalized.length],
        });
        const encoded = runMimiEncode(tree.ref(model.mimi), audioTensor);
        const encodedTransposed = encoded.transpose([1, 0]);
        voiceEmbed = np.dot(encodedTransposed, model.flowLM.speakerProjWeight.ref.transpose()).astype(np.float16);
      } else {
        const voiceUrl = predefinedVoices[voice];
        console.log("Loading voice:", voice, voiceUrl);
        const audioPrompt = safetensors.parse(
          await cachedFetch(voiceUrl)
        ).tensors.audio_prompt;
        voiceEmbed = np
          .array(audioPrompt.data as Float32Array<ArrayBuffer>, {
            shape: audioPrompt.shape,
            dtype: np.float32,
          })
          .slice(0)
          .astype(np.float16);
      }

      const tokensAr = np.array(tokens, { dtype: np.uint32 });
      let embeds = model.flowLM.conditionerEmbed.ref.slice(tokensAr);
      embeds = np.concatenate([voiceEmbed, embeds]);

      const player = createStreamingPlayer();
      currentPlayerRef.current = player;
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        await playTTS(player, tree.ref(model as any), embeds, {
          framesAfterEos,
          temperature: 0.7,
          lsdDecodeSteps: 1,
          signal,
        });
      } finally {
        await player.close();
        currentPlayerRef.current = null;
      }
    } catch (error) {
      console.error("TTS error:", error);
    } finally {
      setIsSpeaking(false);
      setTtsLoading(false);
      currentPlayerRef.current = null;
    }
  }

  const interruptCompletion = useCallback(() => {
    if (isCompletingRef.current) {
      console.log("Interrupting completion");
      workerRef.current?.postMessage({ type: "interrupt" });
      completionAbortControllerRef.current?.abort();
      currentPlayerRef.current?.abort();
      isCompletingRef.current = false;
      setCompletionText(null);
      tokenBufferRef.current = "";
    }
  }, []);

  useEffect(() => {
    interruptCompletionRef.current = interruptCompletion;
  }, [interruptCompletion]);

  const triggerThoughtCompletion = useCallback(async () => {
    if (isCompletingRef.current || !whisperWorkerRef.current) return;

    const audioFrames = audioBufferRef.current;
    if (audioFrames.length === 0) return;

    const totalLength = audioFrames.reduce((sum, frame) => sum + frame.length, 0);
    if (totalLength < 1000) return;

    const combinedAudio = new Float32Array(totalLength);
    let offset = 0;
    for (const frame of audioFrames) {
      combinedAudio.set(frame, offset);
      offset += frame.length;
    }

    isCompletingRef.current = true;
    tokenBufferRef.current = "";
    completionAbortControllerRef.current = new AbortController();

    console.log("Triggering thought completion with", totalLength, "audio samples");

    whisperWorkerRef.current.postMessage({
      type: "generatePartial",
      data: { audio: combinedAudio, language: "en" },
    });
  }, []);

  useEffect(() => {
    triggerThoughtCompletionRef.current = triggerThoughtCompletion;
  }, [triggerThoughtCompletion]);

  // Initialize Whisper worker
  useEffect(() => {
    if (!whisperWorkerRef.current) {
      whisperWorkerRef.current = new Worker(new URL("./whisper-worker.ts", import.meta.url), {
        type: "module",
      });
    }

    const onWhisperMessage = (e: MessageEvent<WorkerResponse>) => {
      switch (e.data.status) {
        case "loading":
          setSttStatus("loading");
          break;
        case "initiate":
          if (e.data.file) {
            setSttProgressItems((prev) => [...prev, { file: e.data.file!, progress: e.data.loaded || 0, total: e.data.total || 0 }]);
          }
          break;
        case "progress":
          setSttProgressItems((prev) =>
            prev.map((item) => {
              if (item.file === e.data.file) {
                return { ...item, progress: e.data.loaded || 0, total: e.data.total || 0 };
              }
              return item;
            })
          );
          break;
        case "done":
          setSttProgressItems((prev) => prev.filter((item) => item.file !== e.data.file));
          break;
        case "ready":
          setSttStatus("ready");
          break;
        case "complete":
          // Full transcription - not used in minimalist mode
          break;
        case "partialComplete":
          if (e.data.output && typeof e.data.output === "string" && isCompletingRef.current) {
            let partialText = e.data.output.trim();
            // Remove trailing punctuation so completion flows naturally
            partialText = partialText.replace(/[.!?,;:]+$/, "");
            if (partialText) {
              console.log("Partial transcription:", partialText);
              setTranscribedText(partialText);
              workerRef.current?.postMessage({
                type: "generateCompletion",
                data: { partialText },
              });
            } else {
              isCompletingRef.current = false;
            }
          }
          break;
      }
    };

    whisperWorkerRef.current.addEventListener("message", onWhisperMessage);
    return () => {
      whisperWorkerRef.current?.removeEventListener("message", onWhisperMessage);
    };
  }, []);

  const whisperWorkerForVad = whisperWorkerRef;

  async function toggleMic() {
    if (vadLoading || vadError) return;

    if (micEnabled) {
      if (vadRef.current) await vadRef.current.pause();
      setMicEnabled(false);
      setUserSpeaking(false);
    } else {
      if (vadRef.current) {
        await vadRef.current.start();
        setMicEnabled(true);
      } else {
        setVadLoading(true);
        try {
          const vad = await MicVAD.new({
            startOnLoad: true,
            baseAssetPath: "/",
            onnxWASMBasePath: "/",
            onSpeechEnd: () => {
              setUserSpeaking(false);
              audioBufferRef.current = [];
              pauseFrameCountRef.current = 0;
              speechStartTimeRef.current = null;
              lastSpeechTimeRef.current = null;
            },
            onSpeechStart: () => {
              setUserSpeaking(true);
              speechStartTimeRef.current = performance.now();
              audioBufferRef.current = [];
              pauseFrameCountRef.current = 0;
              setTranscribedText(null);
              setCompletionText(null);
              interruptCompletionRef.current();
            },
            onFrameProcessed: (probs, audioFrame) => {
              const now = performance.now();

              if (probs.isSpeech > 0.6) {
                setUserSpeaking(true);
                lastSpeechTimeRef.current = now;
                pauseFrameCountRef.current = 0;
                if (audioFrame) {
                  audioBufferRef.current.push(new Float32Array(audioFrame));
                }
              } else if (probs.isSpeech < PAUSE_THRESHOLD) {
                pauseFrameCountRef.current++;
                if (audioFrame && pauseFrameCountRef.current < 10) {
                  audioBufferRef.current.push(new Float32Array(audioFrame));
                }

                const pauseDuration = pauseFrameCountRef.current * FRAME_DURATION_MS;
                const speechDuration = speechStartTimeRef.current
                  ? now - speechStartTimeRef.current
                  : 0;

                if (
                  pauseDuration >= PAUSE_DURATION_MS &&
                  speechDuration >= MIN_SPEECH_BEFORE_COMPLETION_MS &&
                  !isCompletingRef.current
                ) {
                  triggerThoughtCompletionRef.current();
                }
              }

              if (probs.isSpeech > 0.6 && isCompletingRef.current) {
                interruptCompletionRef.current();
              }
            },
          });
          vadRef.current = vad;
          setMicEnabled(true);
        } catch (error) {
          setVadError(error instanceof Error ? error.message : String(error));
        } finally {
          setVadLoading(false);
        }
      }
    }
  }

  useEffect(() => {
    return () => {
      if (vadRef.current) {
        vadRef.current.destroy().catch(console.error);
      }
    };
  }, []);

  // LLM worker
  useEffect(() => {
    if (!workerRef.current) {
      workerRef.current = new Worker(new URL("./worker.ts", import.meta.url), {
        type: "module",
      });
      workerRef.current.postMessage({ type: "check" });
    }

    const onMessageReceived = (e: MessageEvent<WorkerResponse>) => {
      switch (e.data.status) {
        case "loading":
          setLlmStatus("loading");
          setLoadingMessage(e.data.data || "");
          break;
        case "initiate":
          if (e.data.file) {
            setProgressItems((prev) => [...prev, { file: e.data.file!, progress: e.data.loaded || 0, total: e.data.total || 0 }]);
          }
          break;
        case "progress":
          setProgressItems((prev) =>
            prev.map((item) => {
              if (item.file === e.data.file) {
                return { ...item, progress: e.data.loaded || 0, total: e.data.total || 0 };
              }
              return item;
            })
          );
          break;
        case "done":
          setProgressItems((prev) => prev.filter((item) => item.file !== e.data.file));
          break;
        case "ready":
          setLlmStatus("ready");
          break;
        case "update":
          if (isCompletingRef.current && e.data.output) {
            tokenBufferRef.current += e.data.output;
            setCompletionText(tokenBufferRef.current);
          }
          break;
        case "complete":
          if (isCompletingRef.current) {
            const completionResult = tokenBufferRef.current.trim();
            console.log("Completion finished:", completionResult);

            if (completionResult && !completionAbortControllerRef.current?.signal.aborted) {
              const signal = completionAbortControllerRef.current?.signal;
              speakText(completionResult, signal);
            }

            isCompletingRef.current = false;
            tokenBufferRef.current = "";
          }
          break;
      }
    };

    workerRef.current.addEventListener("message", onMessageReceived);
    return () => {
      workerRef.current?.removeEventListener("message", onMessageReceived);
    };
  }, []);

  if (isWebGPUAvailable === null) {
    return (
      <div className="flex items-center justify-center h-screen bg-zinc-950">
        <Loader2 className="w-6 h-6 animate-spin text-zinc-600" />
      </div>
    );
  }

  if (!isWebGPUAvailable) {
    return (
      <div className="flex items-center justify-center h-screen bg-zinc-950 text-zinc-400 text-sm">
        WebGPU is not supported
      </div>
    );
  }

  const isLoading = llmStatus === "loading" || sttStatus === "loading";
  const isReady = llmStatus === "ready" && sttStatus === "ready";
  const isIdle = llmStatus === "idle" || sttStatus === "idle";

  return (
    <>
      <DownloadManager ref={downloadManagerRef} />

      <main className="flex flex-col h-screen bg-zinc-950 text-zinc-100">
        {/* Loading state */}
        {isIdle && !isLoading && (
          <div className="flex-1 flex flex-col items-center justify-center">
            <button
              onClick={() => {
                workerRef.current?.postMessage({ type: "load" });
                whisperWorkerRef.current?.postMessage({ type: "load" });
              }}
              className="px-4 py-2 text-sm text-zinc-400 border border-zinc-800 rounded-lg hover:border-zinc-600 hover:text-zinc-300 transition-colors"
            >
              Load Models
            </button>
          </div>
        )}

        {isLoading && (
          <div className="flex-1 flex flex-col items-center justify-center gap-4">
            <Loader2 className="w-5 h-5 animate-spin text-zinc-600" />
            <p className="text-xs text-zinc-600">{loadingMessage || "Loading..."}</p>
            <div className="w-64 space-y-2">
              {[...progressItems, ...sttProgressItems].map(({ file, progress, total }, i) => (
                <div key={i} className="text-xs">
                  <div className="text-zinc-700 truncate mb-1">{file}</div>
                  <div className="w-full bg-zinc-900 rounded-full h-1">
                    <div
                      className="bg-zinc-700 h-1 rounded-full transition-all"
                      style={{ width: total > 0 ? `${(progress / total) * 100}%` : "0%" }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Main view */}
        {isReady && (
          <>
            {/* Centered text display */}
            <div className="flex-1 flex items-center justify-center px-8">
              <div className="max-w-2xl text-center">
                {!transcribedText && !completionText && !userSpeaking && (
                  <p className="text-zinc-700 text-sm">
                    {micEnabled ? "Listening..." : "Click mic to start"}
                  </p>
                )}

                {userSpeaking && !transcribedText && (
                  <div className="flex items-center justify-center gap-2 text-zinc-500">
                    <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                  </div>
                )}

                {(transcribedText || completionText) && (
                  <p className="text-2xl leading-relaxed">
                    <span className="text-zinc-300">{transcribedText}</span>
                    {completionText && (
                      <span className="text-zinc-500 italic"> {completionText}</span>
                    )}
                  </p>
                )}

                {isSpeaking && (
                  <div className="mt-4 flex items-center justify-center gap-2 text-zinc-600 text-xs">
                    <Volume2 className="w-3 h-3 animate-pulse" />
                  </div>
                )}
              </div>
            </div>

            {/* Bottom controls */}
            <div className="p-6">
              <div className="max-w-md mx-auto flex items-center justify-center gap-4">
                {/* Mic button */}
                <button
                  onClick={toggleMic}
                  disabled={vadLoading}
                  className={`p-3 rounded-full transition-all ${
                    micEnabled
                      ? userSpeaking
                        ? "bg-red-500/20 text-red-400"
                        : "bg-zinc-800 text-zinc-300"
                      : "bg-zinc-900 text-zinc-600 hover:bg-zinc-800 hover:text-zinc-400"
                  }`}
                >
                  {vadLoading ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : micEnabled ? (
                    <Mic className="w-5 h-5" />
                  ) : (
                    <MicOff className="w-5 h-5" />
                  )}
                </button>

                {/* Voice selector */}
                <select
                  value={selectedVoice}
                  onChange={(e) => setSelectedVoice(e.target.value)}
                  disabled={isSpeaking}
                  className="text-xs bg-transparent border border-zinc-800 rounded px-2 py-1.5 text-zinc-500 focus:outline-none focus:border-zinc-600"
                >
                  {Object.keys(predefinedVoices).map((voice) => (
                    <option key={voice} value={voice} className="bg-zinc-900">
                      {voice}
                    </option>
                  ))}
                  <option value="custom" className="bg-zinc-900">custom</option>
                </select>

                {selectedVoice === "custom" && (
                  <>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="audio/wav,.wav"
                      className="hidden"
                      onChange={(e) => setCustomVoiceFile(e.target.files?.[0] || null)}
                    />
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      disabled={isSpeaking}
                      className="text-xs text-zinc-600 hover:text-zinc-400"
                    >
                      {customVoiceFile ? customVoiceFile.name.slice(0, 8) + "..." : "upload"}
                    </button>
                  </>
                )}

                {/* TTS toggle */}
                <button
                  onClick={() => setTtsEnabled(!ttsEnabled)}
                  className={`p-2 rounded transition-colors ${
                    ttsEnabled ? "text-zinc-400" : "text-zinc-700"
                  }`}
                >
                  {ttsEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
                </button>
              </div>
            </div>
          </>
        )}
      </main>
    </>
  );
}
