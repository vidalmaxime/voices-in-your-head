"use client";

import { useEffect, useState, useRef, FormEvent } from "react";
import { AudioLines, Download, Github, Upload } from "lucide-react";

import type { AudioPlayer } from "./audio";
import { createStreamingPlayer } from "./audio";
import { PocketTTSOnnx, playTTSOnnx } from "./onnx";
import { acquireModelRuntimeLease, type ModelRuntimeLease } from "../model-runtime-lock";

export default function TTSPage() {
  const [prompt, setPrompt] = useState("The sun is shining, and the birds are singing.");
  const [voices, setVoices] = useState<string[]>([]);
  const [selectedVoice, setSelectedVoice] = useState<string>("");
  const [customVoiceFile, setCustomVoiceFile] = useState<File | null>(null);
  const [playing, setPlaying] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [isWebGPUAvailable] = useState<boolean | null>(() =>
    typeof navigator !== "undefined" &&
    !!(navigator as Navigator & { gpu?: unknown }).gpu
  );

  const fileInputRef = useRef<HTMLInputElement>(null);
  const playerRef = useRef<AudioPlayer | null>(null);
  const ttsRef = useRef<PocketTTSOnnx | null>(null);
  const seedRef = useRef<number | null>(null);
  const modelRuntimeLeaseRef = useRef<ModelRuntimeLease | null>(null);
  const modelRuntimeLeasePromiseRef = useRef<Promise<boolean> | null>(null);

  // Advanced options
  const [seed, setSeed] = useState<number | null>(null);
  const [temperature, setTemperature] = useState(0.7);
  const [lsdDecodeSteps, setLsdDecodeSteps] = useState(1);

  async function ensureModelRuntimeLease() {
    if (modelRuntimeLeaseRef.current) return true;
    if (modelRuntimeLeasePromiseRef.current) return modelRuntimeLeasePromiseRef.current;

    modelRuntimeLeasePromiseRef.current = acquireModelRuntimeLease()
      .then((lease) => {
        if (!lease) return false;
        modelRuntimeLeaseRef.current = lease;
        return true;
      })
      .finally(() => {
        modelRuntimeLeasePromiseRef.current = null;
      });
    return modelRuntimeLeasePromiseRef.current;
  }

  async function ensureTTS(): Promise<PocketTTSOnnx> {
    if (!await ensureModelRuntimeLease()) {
      throw new Error(
        "Voice models are already loaded in another tab, or this browser cannot provide a safe model lock.",
      );
    }
    if (!ttsRef.current) {
      ttsRef.current = new PocketTTSOnnx({
        onStatus: (message) => setStatusMessage(message),
        onProgress: (progress) =>
          setStatusMessage(`Downloading models... ${(progress * 100).toFixed(0)}%`),
        onAudio: (audio) => {
          const player = playerRef.current;
          if (!player) return;
          const chunk = audio instanceof Float32Array ? audio : new Float32Array(audio);
          player.playChunk(chunk);
        },
        onError: (err) => {
          console.error("TTS worker error:", err);
          setStatusMessage(err.message);
        },
      });
    }

    const tts = ttsRef.current;
    setStatus("loading");
    let resolvedSeed: number | undefined;
    if (typeof seed === "number") {
      resolvedSeed = seed;
      seedRef.current = seed;
    } else {
      resolvedSeed = seedRef.current ?? Math.floor(Math.random() * 2 ** 32);
      seedRef.current = resolvedSeed;
    }
    await tts.initialize({
      seed: resolvedSeed,
      temperature,
      lsd: lsdDecodeSteps,
    });
    const voiceList = await tts.loadVoices();
    setVoices(voiceList);
    if (!selectedVoice && voiceList.length > 0) {
      setSelectedVoice(voiceList[0]);
    }
    setStatus("ready");
    return tts;
  }

  useEffect(() => {
    return () => {
      playerRef.current?.abort();
      void ttsRef.current?.close();
      ttsRef.current = null;
      modelRuntimeLeaseRef.current?.release();
      modelRuntimeLeaseRef.current = null;
    };
  }, []);

  function openCustomVoicePicker() {
    if (!fileInputRef.current) return;
    fileInputRef.current.value = "";
    fileInputRef.current.click();
  }

  function selectVoice(voice: string) {
    if (voice === "custom") {
      openCustomVoicePicker();
      return;
    }
    setSelectedVoice(voice);
  }

  function selectCustomVoiceFile(file: File | null) {
    if (!file) return;
    setCustomVoiceFile(file);
    setSelectedVoice("custom");
  }

  async function run() {
    if (isWebGPUAvailable === false) {
      alert("WebGPU not supported on this device, required for ONNX inference");
      return;
    }

    const tts = await ensureTTS();
    const player = createStreamingPlayer();
    playerRef.current = player;

    try {
      let voice: string | Float32Array;
      if (selectedVoice === "custom") {
        if (!customVoiceFile) {
          throw new Error("Please upload a WAV file for custom voice");
        }
        await tts.encodeVoice(customVoiceFile);
        voice = "custom";
      } else {
        voice = selectedVoice;
      }

      await playTTSOnnx(player, tts, prompt, {
        voice,
        lsdDecodeSteps,
      });
      setAudioBlob(player.toWav());
    } finally {
      await player.close();
      playerRef.current = null;
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setAudioBlob(null);
    setPlaying(true);
    try {
      await run();
    } catch (err) {
      console.error("TTS error:", err);
      setStatus("error");
      setStatusMessage(err instanceof Error ? err.message : "TTS failed");
    } finally {
      setPlaying(false);
    }
  }

  return (
    <>
      <main className="mx-4 my-8">
        <h1 className="text-2xl font-semibold mb-1">
          Kyutai Pocket TTS
          <a
            target="_blank"
            rel="noopener noreferrer"
            href="https://huggingface.co/spaces/KevinAHM/pocket-tts-web"
          >
            <Github className="inline-block ml-2 -mt-1 w-6 h-6" />
          </a>
        </h1>
        <p className="text-lg text-gray-500">
          Text-to-speech AI voice model, running in your browser with ONNX.
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
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/wav,.wav"
              className="hidden"
              onChange={(e) => selectCustomVoiceFile(e.target.files?.[0] ?? null)}
            />
            <select
              className="border-2 rounded p-1"
              value={selectedVoice}
              onChange={(e) => selectVoice(e.target.value)}
              disabled={status !== "ready"}
            >
              {voices.map((voice) => (
                <option key={voice} value={voice}>
                  {voice.charAt(0).toLocaleUpperCase() + voice.slice(1)}
                </option>
              ))}
              <option value="custom">Custom Voice</option>
            </select>

            {selectedVoice === "custom" && (
              <button
                type="button"
                className="btn"
                onClick={openCustomVoicePicker}
              >
                <Upload size={16} />
                {customVoiceFile ? customVoiceFile.name.slice(0, 12) : "Upload WAV"}
              </button>
            )}

            <button
              className="btn"
              type="submit"
              disabled={
                playing ||
                prompt.trim() === "" ||
                (selectedVoice === "custom" && !customVoiceFile)
              }
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

          <details className="mt-6 max-w-md">
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

        {statusMessage && (
          <p className="mt-4 text-sm text-gray-600">{statusMessage}</p>
        )}
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
