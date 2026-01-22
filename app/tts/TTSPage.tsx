"use client";

import { useState, useRef, FormEvent } from "react";
import { defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import { cachedFetch, safetensors, tokenizers } from "@jax-js/loaders";
import { AudioLines, Download, Github } from "lucide-react";

import DownloadManager, { DownloadManagerHandle } from "./DownloadManager";
import { createStreamingPlayer } from "./audio";
import { playTTS } from "./inference";
import { fromSafetensors, type PocketTTS } from "./pocket-tts";

// Cached large objects to download.
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
  const [playing, setPlaying] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);

  // Advanced options
  const [seed, setSeed] = useState<number | null>(null);
  const [temperature, setTemperature] = useState(0.7);
  const [lsdDecodeSteps, setLsdDecodeSteps] = useState(1);

  async function downloadClipWeights(): Promise<safetensors.File> {
    if (_weights) return _weights;
    const weightsUrl =
      "https://huggingface.co/ekzhang/jax-js-models/resolve/main/kyutai-pocket-tts_b6369a24-fp16.safetensors";

    if (!downloadManagerRef.current) {
      throw new Error("Download manager not initialized");
    }

    try {
      const data = await downloadManagerRef.current.fetch("model weights", weightsUrl);
      const result = safetensors.parse(data);
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

    const audioPrompt = safetensors.parse(
      await cachedFetch(predefinedVoices[selectedVoice])
    ).tensors.audio_prompt;
    const voiceEmbed = np
      .array(audioPrompt.data as Float32Array<ArrayBuffer>, {
        shape: audioPrompt.shape,
        dtype: np.float32,
      })
      .slice(0)
      .astype(np.float16);

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
            </select>
            <button
              className="btn"
              type="submit"
              disabled={playing || prompt.trim() === ""}
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
