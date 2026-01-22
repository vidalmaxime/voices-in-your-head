"use client";

import { useState, useRef, FormEvent } from "react";
import { defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { cachedFetch, safetensors, tokenizers } from "@jax-js/loaders";
import { AudioLines, Download, Github, Upload } from "lucide-react";

import DownloadManager, { DownloadManagerHandle } from "./DownloadManager";
import { createStreamingPlayer, parseWav, resampleAudio, SAMPLE_RATE } from "./audio";
import { playTTS } from "./inference";
import { fromSafetensors, runMimiEncode, type PocketTTS } from "./pocket-tts";

// Cached large objects to download.
let _weights: safetensors.File | null = null;

/** Parse safetensors file, converting BF16 tensors to FP16 */
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
      // Convert BF16 to FP16
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

function prepareTextPrompt(text: string): [string, number] {
  // Ported from the Python repository.
  text = text.trim();
  if (text === "") throw new Error("Prompt cannot be empty");
  text = text.replace(/\s+/g, " ");
  const numberOfWords = text.split(" ").length;
  let framesAfterEosGuess = 3;
  if (numberOfWords <= 4) {
    framesAfterEosGuess = 5;
  }

  // Make sure it starts with an uppercase letter
  text = text.replace(/^(\p{Ll})/u, (c) => c.toLocaleUpperCase());

  // Let's make sure it ends with some kind of punctuation
  // If it ends with a letter or digit, we add a period.
  if (/[\p{L}\p{N}]$/u.test(text)) {
    text = text + ".";
  }

  // The model does not perform well when there are very few tokens, so
  // we can add empty spaces at the beginning to increase the token count.
  if (text.split(" ").length < 5) {
    text = " ".repeat(8) + text;
  }

  return [text, framesAfterEosGuess];
}

export default function TTSPage() {
  const downloadManagerRef = useRef<DownloadManagerHandle>(null);

  const [prompt, setPrompt] = useState("The sun is shining, and the birds are singing.");
  const [selectedVoice, setSelectedVoice] = useState("azelma");
  const [customVoiceFile, setCustomVoiceFile] = useState<File | null>(null);
  const [playing, setPlaying] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Advanced options
  const [seed, setSeed] = useState<number | null>(null);
  const [temperature, setTemperature] = useState(0.7);
  const [lsdDecodeSteps, setLsdDecodeSteps] = useState(1);

  async function downloadClipWeights(): Promise<safetensors.File> {
    if (_weights) return _weights;
    const weightsUrl =
      "https://huggingface.co/kyutai/pocket-tts/resolve/main/tts_b6369a24.safetensors";

    if (!downloadManagerRef.current) {
      throw new Error("Download manager not initialized");
    }

    try {
      const data = await downloadManagerRef.current.fetch("model weights", weightsUrl, {
        Authorization: "Bearer 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",
      });
      const result = parseSafetensorsWithBF16(data);
      _weights = result;
      return result;
    } catch (error) {
      alert("Error downloading weights: " + error);
      throw error;
    }
  }

  async function getModel(): Promise<PocketTTS> {
    if (_model) return _model;
    const weights = await downloadClipWeights();
    _model = fromSafetensors(weights);
    return _model;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async function getTokenizer(): Promise<any> {
    if (!_tokenizer) {
      _tokenizer = await tokenizers.loadSentencePiece(
        "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/fbf8280/tokenizer.model"
      );
    }
    return _tokenizer;
  }

  async function run() {
    const devices = await init();
    if (devices.includes("webgpu")) {
      defaultDevice("webgpu");
    } else {
      alert("WebGPU not supported on this device, required for inference");
      return;
    }

    const model = await getModel();
    const tokenizer = await getTokenizer();
    console.log("Model:", model);

    const [text, framesAfterEos] = prepareTextPrompt(prompt);
    const tokens = tokenizer.encode(text);
    console.log("Tokens:", tokens);

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

      // Create audio tensor [1, T] in float16 to match model precision
      const audioTensor = np.array(resampled, {
        dtype: np.float32,
        shape: [1, resampled.length],
      }).astype(np.float16);

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
    let embeds = model.flowLM.conditionerEmbed.ref.slice(tokensAr); // [seq_len, 1024]
    embeds = np.concatenate([voiceEmbed, embeds]);

    const player = createStreamingPlayer();
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      await playTTS(player, tree.ref(model as any), embeds, {
        framesAfterEos,
        seed,
        temperature,
        lsdDecodeSteps,
      });
      setAudioBlob(player.toWav());
    } finally {
      await player.close();
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setAudioBlob(null);
    setPlaying(true);
    try {
      await run();
    } finally {
      setPlaying(false);
    }
  }

  return (
    <>
      <DownloadManager ref={downloadManagerRef} />

      <main className="mx-4 my-8">
        <h1 className="text-2xl font-semibold mb-1">
          Kyutai Pocket TTS
          <a
            target="_blank"
            rel="noopener noreferrer"
            href="https://github.com/ekzhang/jax-js/tree/main/website/src/routes/tts"
          >
            <Github className="inline-block ml-2 -mt-1 w-6 h-6" />
          </a>
        </h1>
        <p className="text-lg text-gray-500">
          Text-to-speech AI voice model, running in your browser with{" "}
          <a href="/" className="text-blue-600 hover:underline">
            jax-js
          </a>
          .
        </p>

        <form className="mt-6" onSubmit={handleSubmit}>
          <textarea
            className="border-2 rounded p-2 w-full max-w-md"
            rows={6}
            placeholder="Enter your prompt here..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />

          <div className="flex gap-3 mt-1 h-9">
            <select
              className="border-2 rounded p-1"
              value={selectedVoice}
              onChange={(e) => setSelectedVoice(e.target.value)}
            >
              {Object.keys(predefinedVoices).map((voice) => (
                <option key={voice} value={voice}>
                  {voice.charAt(0).toLocaleUpperCase() + voice.slice(1)}
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
                  className="btn"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload size={16} />
                  {customVoiceFile ? customVoiceFile.name.slice(0, 12) : "Upload WAV"}
                </button>
              </>
            )}

            <button
              className="btn"
              type="submit"
              disabled={playing || prompt.trim() === "" || (selectedVoice === "custom" && !customVoiceFile)}
            >
              {playing ? (
                <AudioLines size={20} className="animate-pulse" />
              ) : (
                "Play"
              )}
            </button>

            {audioBlob && (
              <a
                className="btn"
                href={URL.createObjectURL(audioBlob)}
                download="tts_output.wav"
              >
                <Download size={20} />
              </a>
            )}
          </div>

          <details className="mt-8 max-w-md">
            <summary className="cursor-pointer text-gray-600 hover:text-gray-800">
              Advanced options
            </summary>
            <div className="mt-3 space-y-4 pl-2">
              <div>
                <label className="block text-sm text-gray-700">
                  Seed
                  <input
                    type="number"
                    className="block mt-1 border-2 rounded p-1 w-32"
                    placeholder="(random)"
                    value={seed ?? ""}
                    onChange={(e) =>
                      setSeed(e.target.value ? parseInt(e.target.value, 10) : null)
                    }
                  />
                </label>
              </div>

              <div>
                <label className="block text-sm text-gray-700">
                  Temperature: {temperature.toFixed(2)}
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    className="mt-1 w-full"
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  />
                </label>
              </div>

              <div>
                <label className="block text-sm text-gray-700">
                  LSD Decode Steps: {lsdDecodeSteps}
                  <input
                    type="range"
                    min="1"
                    max="4"
                    step="1"
                    className="mt-1 w-full"
                    value={lsdDecodeSteps}
                    onChange={(e) => setLsdDecodeSteps(parseInt(e.target.value, 10))}
                  />
                </label>
              </div>
            </div>
          </details>
        </form>
      </main>

      <style jsx>{`
        .btn {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
          padding: 0.25rem 0.75rem;
          border-radius: 0.25rem;
          border: 2px solid black;
          transition: background-color 0.15s, color 0.15s;
        }
        .btn:disabled {
          opacity: 0.5;
          cursor: wait;
        }
        .btn:not(:disabled):hover {
          background-color: black;
          color: white;
        }
      `}</style>
    </>
  );
}
