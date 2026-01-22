"use client";

import { useEffect, useState, useRef, FormEvent } from "react";
import { defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { cachedFetch, safetensors, tokenizers } from "@jax-js/loaders";
import { MessageSquare, Send, Square, Volume2, VolumeX, Loader2, Upload } from "lucide-react";

import DownloadManager, { DownloadManagerHandle } from "../tts/DownloadManager";
import { createStreamingPlayer, parseWav, resampleAudio, SAMPLE_RATE } from "../tts/audio";
import { playTTS } from "../tts/inference";
import { fromSafetensors, runMimiEncode, type PocketTTS } from "../tts/pocket-tts";

// Types for worker communication
interface WorkerResponse {
  status: string;
  data?: string;
  output?: string;
  tps?: number;
  numTokens?: number;
  state?: "thinking" | "answering";
  file?: string;
  progress?: number;  // percentage 0-100
  loaded?: number;    // bytes loaded
  total?: number;     // total bytes
}

interface Message {
  role: "user" | "assistant";
  content: string;
  answerIndex?: number;
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

const EXAMPLES = [
  "Tell me a short joke",
  "Explain what WebGPU is in one sentence",
  "Write a haiku about programming",
];

export default function LLMPage() {
  const downloadManagerRef = useRef<DownloadManagerHandle>(null);
  const workerRef = useRef<Worker | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // LLM state
  const [llmStatus, setLlmStatus] = useState<"idle" | "loading" | "ready">("idle");
  const [loadingMessage, setLoadingMessage] = useState("");
  const [progressItems, setProgressItems] = useState<ProgressItem[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [tps, setTps] = useState<number | null>(null);
  const [numTokens, setNumTokens] = useState<number | null>(null);

  // TTS state
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [ttsLoading, setTtsLoading] = useState(false);
  const [selectedVoice, setSelectedVoice] = useState("azelma");
  const [customVoiceFile, setCustomVoiceFile] = useState<File | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);

  // WebGPU availability (checked on client only to avoid hydration mismatch)
  const [isWebGPUAvailable, setIsWebGPUAvailable] = useState<boolean | null>(null);

  // Queue for TTS - speak completed responses
  const pendingSpeechRef = useRef<string | null>(null);

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

  async function speakText(text: string) {
    if (!ttsEnabled || !text.trim()) return;
    if (selectedVoice === "custom" && !customVoiceFile) return;

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

      const [preparedText, framesAfterEos] = prepareTextPrompt(text);
      const tokens = tokenizer.encode(preparedText);

      let voiceEmbed: np.Array;

      if (selectedVoice === "custom" && customVoiceFile) {
        // Voice cloning: encode custom audio
        console.log("Processing custom voice file...");
        const arrayBuffer = await customVoiceFile.arrayBuffer();
        const { samples, sampleRate } = parseWav(arrayBuffer);
        console.log(`Loaded WAV: ${samples.length} samples at ${sampleRate}Hz`);

        // Resample to 24kHz if needed
        const resampled = resampleAudio(samples, sampleRate, SAMPLE_RATE);
        console.log(`Resampled to ${resampled.length} samples at ${SAMPLE_RATE}Hz`);

        // Normalize to consistent peak amplitude
        let maxAbs = 0;
        for (let i = 0; i < resampled.length; i++) {
          const value = Math.abs(resampled[i]);
          if (value > maxAbs) maxAbs = value;
        }
        const normalized = maxAbs > 0 ? resampled.map((value) => value / maxAbs) : resampled;

        // Create audio tensor [1, T] in float32
        const audioTensor = np.array(normalized, {
          dtype: np.float32,
          shape: [1, normalized.length],
        });

        // Encode with Mimi encoder -> [512, T']
        console.log("Encoding audio with Mimi...");
        const encoded = runMimiEncode(tree.ref(model.mimi), audioTensor);
        console.log("Encoded shape:", encoded.shape);

        // Transpose to [T', 512]
        const encodedTransposed = encoded.transpose([1, 0]);

        // Project to conditioning space: [T', 512] @ [512, 1024] -> [T', 1024]
        voiceEmbed = np.dot(encodedTransposed, model.flowLM.speakerProjWeight.ref.transpose()).astype(np.float16);
        console.log("Voice embedding shape:", voiceEmbed.shape);
      } else {
        // Use predefined voice
        const audioPrompt = safetensors.parse(
          await cachedFetch(predefinedVoices[selectedVoice])
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
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        await playTTS(player, tree.ref(model as any), embeds, {
          framesAfterEos,
          temperature: 0.7,
          lsdDecodeSteps: 1,
        });
      } finally {
        await player.close();
      }
    } catch (error) {
      console.error("TTS error:", error);
    } finally {
      setIsSpeaking(false);
      setTtsLoading(false);
    }
  }

  function onSendMessage(message: string) {
    setMessages((prev) => [...prev, { role: "user", content: message }]);
    setTps(null);
    setNumTokens(null);
    setIsGenerating(true);
    setInput("");
  }

  function onInterrupt() {
    workerRef.current?.postMessage({ type: "interrupt" });
  }

  useEffect(() => {
    if (!textareaRef.current) return;
    const target = textareaRef.current;
    target.style.height = "auto";
    const newHeight = Math.min(Math.max(target.scrollHeight, 24), 200);
    target.style.height = `${newHeight}px`;
  }, [input]);

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
        case "start":
          setMessages((prev) => [...prev, { role: "assistant", content: "" }]);
          break;
        case "update":
          setTps(e.data.tps || null);
          setNumTokens(e.data.numTokens || null);
          setMessages((prev) => {
            const cloned = [...prev];
            const last = cloned.at(-1);
            if (last && last.role === "assistant") {
              const data: Message = {
                ...last,
                content: last.content + (e.data.output || ""),
              };
              if (data.answerIndex === undefined && e.data.state === "answering") {
                data.answerIndex = last.content.length;
              }
              cloned[cloned.length - 1] = data;
            }
            return cloned;
          });
          break;
        case "complete":
          setIsGenerating(false);
          // Get the final assistant message and queue it for TTS
          setMessages((prev) => {
            const lastMsg = prev.at(-1);
            if (lastMsg?.role === "assistant" && lastMsg.content) {
              // Extract just the answer part (after thinking)
              const answerText = lastMsg.answerIndex !== undefined
                ? lastMsg.content.slice(lastMsg.answerIndex)
                : lastMsg.content;
              pendingSpeechRef.current = answerText;
            }
            return prev;
          });
          break;
        case "error":
          console.error("Worker error:", e.data.data);
          setIsGenerating(false);
          break;
      }
    };

    workerRef.current.addEventListener("message", onMessageReceived);
    return () => {
      workerRef.current?.removeEventListener("message", onMessageReceived);
    };
  }, []);

  // Trigger generation when user sends message
  useEffect(() => {
    if (messages.filter((x) => x.role === "user").length === 0) return;
    if (messages.at(-1)?.role === "assistant") return;
    workerRef.current?.postMessage({
      type: "generate",
      data: { messages, reasonEnabled: false },
    });
  }, [messages]);

  // Auto-scroll chat
  useEffect(() => {
    if (!chatContainerRef.current || !isGenerating) return;
    const element = chatContainerRef.current;
    if (element.scrollHeight - element.scrollTop - element.clientHeight < 120) {
      element.scrollTop = element.scrollHeight;
    }
  }, [messages, isGenerating]);

  // Speak completed responses
  useEffect(() => {
    if (!isGenerating && pendingSpeechRef.current && ttsEnabled && !isSpeaking) {
      const text = pendingSpeechRef.current;
      pendingSpeechRef.current = null;
      speakText(text);
    }
  }, [isGenerating, ttsEnabled, isSpeaking]);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (input.trim() && !isGenerating) {
      onSendMessage(input.trim());
    }
  }

  // Show loading while checking WebGPU
  if (isWebGPUAvailable === null) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Loader2 className="w-8 h-8 animate-spin text-zinc-400" />
      </div>
    );
  }

  if (!isWebGPUAvailable) {
    return (
      <div className="fixed inset-0 bg-black/90 text-white text-2xl font-semibold flex justify-center items-center text-center p-4">
        WebGPU is not supported by this browser
      </div>
    );
  }

  return (
    <>
      <DownloadManager ref={downloadManagerRef} />

      <main className="flex flex-col h-screen bg-white dark:bg-zinc-900">
        {/* Header */}
        <header className="border-b border-zinc-200 dark:border-zinc-700 p-4">
          <div className="max-w-3xl mx-auto flex items-center justify-between">
            <h1 className="text-xl font-semibold flex items-center gap-2">
              <MessageSquare className="w-5 h-5" />
              Local LLM + TTS
            </h1>
            <div className="flex items-center gap-2">
              <select
                className="text-sm border rounded px-2 py-1 bg-white dark:bg-zinc-800 dark:border-zinc-600"
                value={selectedVoice}
                onChange={(e) => setSelectedVoice(e.target.value)}
                disabled={isSpeaking}
              >
                {Object.keys(predefinedVoices).map((voice) => (
                  <option key={voice} value={voice}>
                    {voice.charAt(0).toUpperCase() + voice.slice(1)}
                  </option>
                ))}
                <option value="custom">Custom Voice</option>
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
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isSpeaking}
                    className="text-sm border rounded px-2 py-1 bg-white dark:bg-zinc-800 dark:border-zinc-600 hover:bg-zinc-100 dark:hover:bg-zinc-700 flex items-center gap-1"
                  >
                    <Upload className="w-4 h-4" />
                    {customVoiceFile ? customVoiceFile.name.slice(0, 10) + "..." : "Upload WAV"}
                  </button>
                </>
              )}

              <button
                onClick={() => setTtsEnabled(!ttsEnabled)}
                className={`p-2 rounded-lg transition-colors ${
                  ttsEnabled
                    ? "bg-blue-100 text-blue-600 dark:bg-blue-900 dark:text-blue-300"
                    : "bg-zinc-100 text-zinc-400 dark:bg-zinc-800"
                }`}
                title={ttsEnabled ? "TTS enabled" : "TTS disabled"}
              >
                {ttsEnabled ? <Volume2 className="w-5 h-5" /> : <VolumeX className="w-5 h-5" />}
              </button>
            </div>
          </div>
        </header>

        {/* Loading state */}
        {llmStatus === "idle" && (
          <div className="flex-1 flex flex-col items-center justify-center p-8">
            <h2 className="text-2xl font-bold mb-2">Qwen3 0.6B WebGPU</h2>
            <p className="text-zinc-500 dark:text-zinc-400 text-center max-w-md mb-6">
              A local LLM running entirely in your browser with WebGPU. Responses will be spoken aloud using Pocket TTS.
            </p>
            <button
              onClick={() => workerRef.current?.postMessage({ type: "load" })}
              className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium"
            >
              Load Model
            </button>
          </div>
        )}

        {llmStatus === "loading" && (
          <div className="flex-1 flex flex-col items-center justify-center p-8">
            <Loader2 className="w-8 h-8 animate-spin mb-4 text-blue-500" />
            <p className="text-zinc-600 dark:text-zinc-400 mb-4">{loadingMessage}</p>
            <div className="w-full max-w-md space-y-2">
              {progressItems.map(({ file, progress, total }) => (
                <div key={file} className="text-sm">
                  <div className="text-zinc-500 dark:text-zinc-400 mb-1 truncate">{file}</div>
                  <div className="w-full bg-zinc-200 dark:bg-zinc-700 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full transition-all"
                      style={{ width: total > 0 ? `${(progress / total) * 100}%` : "0%" }}
                    />
                  </div>
                  <div className="text-xs text-zinc-400 mt-1">
                    {total > 0
                      ? `${(progress / 1024 / 1024).toFixed(1)} / ${(total / 1024 / 1024).toFixed(1)} MB`
                      : `${(progress / 1024 / 1024).toFixed(1)} MB`}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Chat interface */}
        {llmStatus === "ready" && (
          <>
            <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-4">
              <div className="max-w-3xl mx-auto space-y-4">
                {messages.length === 0 && (
                  <div className="text-center py-12">
                    <p className="text-zinc-500 dark:text-zinc-400 mb-4">Try one of these examples:</p>
                    <div className="flex flex-wrap justify-center gap-2">
                      {EXAMPLES.map((example) => (
                        <button
                          key={example}
                          onClick={() => onSendMessage(example)}
                          className="px-4 py-2 bg-zinc-100 dark:bg-zinc-800 rounded-lg hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors text-sm"
                        >
                          {example}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {messages.map((msg, i) => (
                  <div
                    key={i}
                    className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[80%] rounded-lg px-4 py-2 ${
                        msg.role === "user"
                          ? "bg-blue-500 text-white"
                          : "bg-zinc-100 dark:bg-zinc-800"
                      }`}
                    >
                      <p className="whitespace-pre-wrap">{msg.content || (isGenerating && i === messages.length - 1 ? "..." : "")}</p>
                    </div>
                  </div>
                ))}

                {isSpeaking && (
                  <div className="flex justify-start">
                    <div className="flex items-center gap-2 text-sm text-zinc-500 dark:text-zinc-400">
                      <Volume2 className="w-4 h-4 animate-pulse" />
                      Speaking...
                    </div>
                  </div>
                )}

                {ttsLoading && (
                  <div className="flex justify-start">
                    <div className="flex items-center gap-2 text-sm text-zinc-500 dark:text-zinc-400">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Loading TTS model...
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Stats bar */}
            {tps && (
              <div className="text-center text-sm text-zinc-500 dark:text-zinc-400 py-1">
                {isGenerating ? (
                  <span>{tps.toFixed(1)} tokens/sec</span>
                ) : (
                  <span>
                    Generated {numTokens} tokens ({tps.toFixed(1)} tokens/sec)
                    {" · "}
                    <button
                      onClick={() => {
                        workerRef.current?.postMessage({ type: "reset" });
                        setMessages([]);
                      }}
                      className="underline hover:text-zinc-700 dark:hover:text-zinc-300"
                    >
                      Reset
                    </button>
                  </span>
                )}
              </div>
            )}

            {/* Input area */}
            <form onSubmit={handleSubmit} className="border-t border-zinc-200 dark:border-zinc-700 p-4">
              <div className="max-w-3xl mx-auto flex gap-2">
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey && input.trim() && !isGenerating) {
                      e.preventDefault();
                      onSendMessage(input.trim());
                    }
                  }}
                  placeholder="Type a message..."
                  rows={1}
                  className="flex-1 resize-none rounded-lg border border-zinc-300 dark:border-zinc-600 bg-white dark:bg-zinc-800 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={isGenerating}
                />
                {isGenerating ? (
                  <button
                    type="button"
                    onClick={onInterrupt}
                    className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
                  >
                    <Square className="w-5 h-5" />
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={!input.trim()}
                    className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <Send className="w-5 h-5" />
                  </button>
                )}
              </div>
            </form>
          </>
        )}
      </main>
    </>
  );
}
