"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, Mic, RotateCcw, Square } from "lucide-react";

import {
  MAX_ENROLLMENT_SECONDS,
  MIN_ENROLLMENT_SECONDS,
  VOICE_ENROLLMENT_SENTENCE,
  VoiceRecorder,
} from "../tts/voice-recorder";

type VoiceEnrollmentProps = {
  /** Encodes the recorded clip as the active voice. Rejects if it is unusable. */
  onRecorded: (file: File) => Promise<void>;
  /** Leaves enrollment without cloning, keeping whatever voice is selected. */
  onSkip: () => void;
  skipLabel: string;
};

type Phase = "idle" | "requesting" | "recording" | "processing";

function formatSeconds(seconds: number) {
  const whole = Math.floor(seconds);
  return `0:${String(whole).padStart(2, "0")}`;
}

export default function VoiceEnrollment({
  onRecorded,
  onSkip,
  skipLabel,
}: VoiceEnrollmentProps) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [elapsed, setElapsed] = useState(0);
  const [level, setLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const recorderRef = useRef<VoiceRecorder | null>(null);
  const stopRef = useRef<() => void>(() => {});

  useEffect(() => {
    return () => {
      void recorderRef.current?.cancel();
      recorderRef.current = null;
    };
  }, []);

  const stopRecording = useCallback(async () => {
    const recorder = recorderRef.current;
    if (!recorder || !recorder.isRecording) return;

    setPhase("processing");
    try {
      const { file, durationSeconds } = await recorder.stop();
      recorderRef.current = null;
      if (durationSeconds < MIN_ENROLLMENT_SECONDS) {
        setError(
          `Only ${formatSeconds(durationSeconds)} recorded. Read the whole sentence so the clone has enough voice to copy.`,
        );
        setPhase("idle");
        return;
      }
      await onRecorded(file);
    } catch (err) {
      recorderRef.current = null;
      setError(err instanceof Error ? err.message : String(err));
      setPhase("idle");
    }
  }, [onRecorded]);

  useEffect(() => {
    stopRef.current = () => void stopRecording();
  }, [stopRecording]);

  // Drives the elapsed readout and the hard stop at MAX_ENROLLMENT_SECONDS.
  useEffect(() => {
    if (phase !== "recording") return;
    const startedAt = performance.now();
    const id = window.setInterval(() => {
      const seconds = (performance.now() - startedAt) / 1000;
      setElapsed(seconds);
      if (seconds >= MAX_ENROLLMENT_SECONDS) stopRef.current();
    }, 100);
    return () => window.clearInterval(id);
  }, [phase]);

  async function startRecording() {
    if (phase !== "idle") return;
    await beginRecording();
  }

  /** Discards the take in progress and starts over, keeping the microphone. */
  async function restartRecording() {
    if (phase !== "recording") return;
    const recorder = recorderRef.current;
    recorderRef.current = null;
    setPhase("requesting");
    await recorder?.cancel();
    await beginRecording();
  }

  async function beginRecording() {
    setError(null);
    setPhase("requesting");
    setElapsed(0);
    setLevel(0);
    const recorder = new VoiceRecorder();
    recorderRef.current = recorder;
    try {
      await recorder.start({ onLevel: setLevel });
      setPhase("recording");
    } catch (err) {
      recorderRef.current = null;
      setPhase("idle");
      setError(
        err instanceof Error
          ? err.message
          : "Could not access the microphone. Check the browser permission and retry.",
      );
    }
  }

  const isRecording = phase === "recording";
  const isProcessing = phase === "processing";
  const isRequesting = phase === "requesting";
  const isBusy = isRequesting || isProcessing;

  return (
    <div className="flex-1 min-h-0 overflow-y-auto flex flex-col items-center gap-8 px-5 py-8 sm:px-8 sm:py-12">
      <header className="mt-auto flex max-w-xl flex-col items-center gap-4 text-center">
        <h1 className="text-2xl tracking-tight text-zinc-100 sm:text-3xl">Teach your voice to the voices.</h1>
        <div className="space-y-2 text-sm leading-relaxed text-zinc-400">
          <p>Record your voice by saying the following sentence.</p>
          <p>Speak naturally in a quiet room. It takes about ten seconds. Your voice stays on this device.</p>
        </div>
      </header>

      <section aria-label="Sentence to read aloud" className="w-full max-w-2xl rounded-2xl border border-zinc-800 bg-zinc-900/40 px-6 py-6 sm:px-10 sm:py-8">
        <p className="text-xl leading-relaxed text-zinc-200 sm:text-2xl sm:leading-relaxed">
          {VOICE_ENROLLMENT_SENTENCE}
        </p>
      </section>

      <div className="flex flex-col items-center gap-4">
        <button
          onClick={() => (isRecording ? void stopRecording() : void startRecording())}
          disabled={isBusy}
          aria-label={isRecording ? "Stop recording" : "Start recording"}
          className={`flex cursor-pointer items-center gap-3 rounded-full px-6 py-3 text-sm transition-all disabled:cursor-default ${
            isRecording
              ? "bg-red-500/15 text-red-300 hover:bg-red-500/25"
              : "border border-zinc-800 text-zinc-300 hover:border-zinc-600 disabled:opacity-50"
          }`}
        >
          {isProcessing ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Cloning your voice...
            </>
          ) : isRequesting ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Waiting for the microphone...
            </>
          ) : isRecording ? (
            <>
              <Square className="w-4 h-4 fill-current" />
              Stop and use my voice
            </>
          ) : (
            <>
              <Mic className="w-4 h-4" />
              Start recording
            </>
          )}
        </button>

        <div
          className={`flex items-center gap-3 transition-opacity ${
            isRecording ? "opacity-100" : "opacity-0"
          }`}
        >
          <button
            onClick={() => void restartRecording()}
            disabled={!isRecording}
            aria-label="Restart recording"
            title="Discard this take and start over"
            className="flex cursor-pointer items-center gap-1.5 text-xs text-zinc-500 transition-colors hover:text-zinc-300 disabled:cursor-default"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Restart
          </button>
          <div className="h-1 w-48 overflow-hidden rounded-full bg-zinc-900">
            <div
              className="h-1 rounded-full bg-red-400/70 transition-[width] duration-100"
              style={{ width: `${Math.round(level * 100)}%` }}
            />
          </div>
          <span className="text-xs tabular-nums text-zinc-600">
            {formatSeconds(elapsed)} / {formatSeconds(MAX_ENROLLMENT_SECONDS)}
          </span>
        </div>
      </div>

      <button
        onClick={onSkip}
        disabled={isRecording || isBusy}
        className="mb-auto cursor-pointer text-xs text-zinc-600 underline underline-offset-4 transition-colors hover:text-zinc-400 disabled:cursor-default disabled:opacity-40"
      >
        {skipLabel}
      </button>

      {error && (
        <p role="alert" className="max-w-sm text-center text-xs text-red-400">
          {error}
        </p>
      )}
    </div>
  );
}
