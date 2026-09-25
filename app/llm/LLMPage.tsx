"use client";

import { prefetchModelFiles } from "./model-loading";
import { POCKET_MODEL_CACHE, pocketPrefetchUrls } from "../tts/model-assets";

import { useEffect, useState, useRef, useCallback } from "react";
import IntroScreen from "./IntroScreen";
import ChorusStage, { chorusVoicePan } from "./ChorusStage";
import MicButton from "./MicButton";
import LoadingScreen from "./LoadingScreen";
import { Loader2, Volume2, RotateCcw, SlidersHorizontal, X } from "lucide-react";
import { MicVAD, NonRealTimeVAD } from "@ricky0123/vad-web";

import { BASE_PATH, assetUrl } from "../base-path";
import type { AudioPlayer } from "../tts/audio";
import { createStreamingPlayer, resampleAudio, warmAudioOutput } from "../tts/audio";
import { PocketTTSOnnx, playTTSOnnx } from "../tts/onnx";
import { acquireModelRuntimeLease, type ModelRuntimeLease } from "../model-runtime-lock";
import { sanitizeThoughtCompletion } from "./completion-logic";
import { appendContextTurn, type ContextTurn } from "./conversation-context";
import { ChorusFeed } from "./chorus-feed";
import { createStaggeredChorus } from "./staggered-chorus";
import { SpeechQueue } from "./speech-queue";
import {
  SpeculationCoordinator,
  type SpeculationAction,
  type SpeculationOptions,
  type SpeculationStats,
} from "./speculation";
import { requestMicrophoneStream } from "./microphone";
import VoiceEnrollment from "./VoiceEnrollment";
import {
  appendRollingAudioFrame,
  beginSpeechCapture,
  decideVadFrame,
  shouldFinalizeOnVadEnd,
  shouldRestartVadAfterPlayback,
  shouldSuppressVadInput,
  shouldTriggerOverlapCompletion,
} from "./vad-logic";

interface WorkerResponse {
  status: string;
  data?: string;
  output?: string;
  requestId?: number;
  tps?: number;
  numTokens?: number;
  firstTokenMs?: number;
  prefixCacheUsed?: boolean;
  state?: "thinking" | "answering";
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
  branchId?: number;
  branchComplete?: boolean;
  timings?: unknown;
  loadProfile?: Record<string, unknown>;
}

interface Branch {
  id: number;
  text: string;
  complete: boolean;
}

interface ProgressItem {
  file: string;
  progress: number;
  total: number;
}

type PerformanceMemory = {
  usedJSHeapSize: number;
  jsHeapSizeLimit: number;
};

type PerformanceWithMemory = Performance & {
  memory?: PerformanceMemory;
  measureUserAgentSpecificMemory?: () => Promise<{ bytes: number }>;
};

/** A labelled on/off row for the options panel. */
function OptionSwitch({
  label,
  detail,
  on,
  disabled = false,
  onToggle,
  children,
}: {
  label: string;
  detail?: string;
  on: boolean;
  disabled?: boolean;
  onToggle: () => void;
  children?: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="flex flex-col">
        <span className="text-white">{label}</span>
        {detail && <span className="text-xs text-white/40">{detail}</span>}
      </span>
      <span className="flex items-center gap-3">
        {children}
        <button
          type="button"
          role="switch"
          aria-checked={on}
          aria-label={label}
          disabled={disabled}
          onClick={onToggle}
          className={`relative block h-6 w-11 shrink-0 rounded-full border transition-colors disabled:opacity-50 ${
            on ? "border-white bg-white" : "border-white/30 bg-transparent"
          }`}
        >
          <span
            aria-hidden
            className={`absolute left-0.5 top-0.5 block h-[18px] w-[18px] rounded-full transition-transform duration-200 ${
              on ? "translate-x-5 bg-black" : "translate-x-0 bg-white/70"
            }`}
          />
        </button>
      </span>
    </div>
  );
}

function isLocalDebugHost() {
  return ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
}

/** Debug autoruns drive the pipeline headlessly, so they skip the voice gate. */
function hasDebugAutorunParams() {
  if (!isLocalDebugHost()) return false;
  const params = new URLSearchParams(window.location.search);
  return ["debugText", "debugAudio", "debugMicFixture", "debugCustomVoice"].some(
    (key) => params.has(key),
  );
}

/** A URL switch for local A/B runs; null anywhere but a local debug host. */
function debugParam(name: string) {
  if (typeof window === "undefined" || !isLocalDebugHost()) return null;
  return new URLSearchParams(window.location.search).get(name);
}

/**
 * Speculative transcription runs Parakeet during speech so the transcript is
 * ready at the endpoint. The completion speculates once per pause: it starts
 * from the pause-pass transcript, runs through the endpoint wait, and is
 * stopped if speech resumes. Completing every mid-speech revision ("full")
 * starved the single speech worker on a 16 GB machine and lost more than it
 * gained (see README.md). The debug flag forces a mode for A/B runs:
 * 0 = serial, stt = transcription only, pause = the default, full = every
 * revision.
 */
function debugSpeculativeParam() {
  return debugParam("debugSpeculative");
}

/**
 * `debugLlmModel=<key>` overrides the completion model for a benchmark run.
 * Keys map to onnx-community / HF ONNX exports; see README.md.
 */
const COMPLETION_MODEL_PRESETS: Record<
  string,
  { modelId: string; dtype: string; externalData: boolean; fewShot?: boolean; local?: boolean; maxResponseTokens?: number }
> = {
  "granite-1b": { modelId: "onnx-community/granite-4.0-1b-ONNX", dtype: "q4", externalData: true },
  "qwen2.5-0.5b": { modelId: "onnx-community/Qwen2.5-0.5B-Instruct", dtype: "q4", externalData: false },
  "qwen2.5-1.5b": { modelId: "onnx-community/Qwen2.5-1.5B-Instruct", dtype: "q4", externalData: true },
  "llama-3.2-1b": { modelId: "onnx-community/Llama-3.2-1B-Instruct", dtype: "q4", externalData: true },
  "smollm2-360m": { modelId: "HuggingFaceTB/SmolLM2-360M-Instruct", dtype: "q4", externalData: false },
  "lfm2.5-350m": { modelId: "LiquidAI/LFM2.5-350M-ONNX", dtype: "q4", externalData: true },
  "smollm2-1.7b": { modelId: "HuggingFaceTB/SmolLM2-1.7B-Instruct", dtype: "q4", externalData: true },
  // Personal finetune (scripts/finetune), served locally from public/models/
  // (gitignored). fewShot: false because the SFT data has no in-context
  // examples; local: true keeps it off the Hub entirely.
  "personal-350m": { modelId: "personal-350m", dtype: "q4", externalData: true, fewShot: false, local: true, maxResponseTokens: 32 },
  // The Weil + Camus finetune (README.md), published on
  // the Hub under a research and artistic use license. Trained on 8-24 word
  // tails (<=31 tokens), so it needs a wider output cap than the 16-token
  // default; decoding still stops at the first sentence end.
  "notebooks-350m": { modelId: "maxime/personal-notebooks-350m", dtype: "q4", externalData: true, fewShot: false, maxResponseTokens: 32 },
};

// The app ships one completion model, the Weil + Camus notebooks finetune.
// The other presets stay reachable on a local debug host through
// ?debugLlmModel=<key> for A/B runs (see README.md).
const DEFAULT_COMPLETION_MODEL_KEY = "notebooks-350m";

function initialCompletionModelKey() {
  if (typeof window !== "undefined" && isLocalDebugHost()) {
    const key = new URLSearchParams(window.location.search).get("debugLlmModel");
    if (key && COMPLETION_MODEL_PRESETS[key]) return key;
  }
  return DEFAULT_COMPLETION_MODEL_KEY;
}

function initialSpeculativeMode() {
  return debugSpeculativeParam() !== "0";
}

function initialSpeculationOptions(): SpeculationOptions {
  const mode = debugSpeculativeParam();
  return { speculateCompletions: mode === "full" ? "full" : mode === "stt" ? false : "pause" };
}

/** Audio buffered before playback starts; one 320 ms decode chunk by default. */
function initialTtsStartBufferMs() {
  const value = Number(debugParam("debugTtsBuffer"));
  return Number.isFinite(value) && value > 0 ? value : 320;
}

/** Stop decoding at the first sentence end unless `debugEarlyStop=0`. */
function initialEarlyStop() {
  return debugParam("debugEarlyStop") !== "0";
}

/** Keep the TTS voice state resident on the GPU unless `debugResidentState=0`. */
function initialResidentState() {
  return debugParam("debugResidentState") !== "0";
}

/**
 * Touch the language and speech models while the app is idle so the first
 * utterance after a quiet minute does not pay ~800 ms to its first token.
 * `debugKeepWarm=0` disables it for A/B runs.
 */
const MODEL_KEEP_WARM_MS = 20000;

function initialKeepWarm() {
  return debugParam("debugKeepWarm") !== "0";
}

function setLocalDebugPhase(phase: string) {
  document.documentElement.dataset.voicesDebugPhase = phase;
}

function getAudioContextConstructor() {
  return window.AudioContext ||
    (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
}

async function measurePageMemoryMb() {
  const perf = performance as PerformanceWithMemory;
  try {
    if (typeof perf.measureUserAgentSpecificMemory === "function") {
      const result = await Promise.race([
        perf.measureUserAgentSpecificMemory(),
        new Promise<null>((resolve) => window.setTimeout(() => resolve(null), 5000)),
      ]);
      if (result) return result.bytes / 1024 / 1024;
    }
  } catch {
    // The browser-wide API is optional and may reject when isolation is unavailable.
  }

  // JS heap excludes workers, WASM, and WebGPU allocations, so it is not a useful
  // substitute for the page-wide model-memory measurement.
  return null;
}

type DebugPipelineResult = {
  partialText: string;
  completionText: string;
  vadMs?: number;
  vadSegmentCount?: number;
  vadSpeechMs?: number;
  sttMs?: number;
  llmFirstTokenMs: number | null;
  llmTokens: number | null;
  llmTps: number | null;
  llmPrefixCacheUsed: boolean | null;
  llmMs: number;
  ttsFirstAudioMs: number | null;
  ttsUnderrunCount: number | null;
  ttsUnderrunMs: number | null;
  ttsMs: number;
  totalMs: number;
  memoryBeforeMb: number | null;
  memoryAfterMb: number | null;
};

type DebugMicrophoneRun = {
  fixture: string;
  startedAt: number;
  vadFrameCount: number;
  speechStartCount: number;
  sttStartedAt: number | null;
  sttMs: number | null;
  llmStartedAt: number | null;
  llmFirstTokenMs: number | null;
  llmMs: number | null;
  speculation: SpeculationStats | null;
  // Fine-grained marks (performance.now()) for the per-stage profile.
  llmQueuedAt: number | null;
  llmDoneAt: number | null;
  llmTimings: unknown;
  speakCalledAt: number | null;
  ttsTimings: unknown;
  ttsMetrics: unknown;
  playbackStartedAt: number | null;
};

declare global {
  interface Window {
    __voicesDebug?: {
      getState: () => {
        llmStatus: "idle" | "loading" | "ready";
        sttStatus: "idle" | "loading" | "ready";
        ttsBackend: "pocket";
        ttsEnabled: boolean;
        isSpeaking: boolean;
        memoryStats: {
          heapUsedMb: number | null;
          heapLimitMb: number | null;
          deviceGb: number | null;
        };
        loadTimes: Partial<Record<"llm" | "stt" | "tts", number>>;
      };
      measureMemory: () => Promise<number | null>;
      loadCoreModels: () => Promise<void>;
      runTextPipeline: (partialText: string, options?: {
        speak?: boolean;
        measureMemory?: boolean;
        usePrefixCache?: boolean;
      }) => Promise<DebugPipelineResult>;
      runAudioPipeline: (audio: Float32Array, options?: {
        speak?: boolean;
        measureMemory?: boolean;
        usePrefixCache?: boolean;
      }) => Promise<DebugPipelineResult>;
    };
  }
}

export default function LLMPage() {
  const workerRef = useRef<Worker | null>(null);
  const whisperWorkerRef = useRef<Worker | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const coreLoadRequestedRef = useRef(false);
  const loadProfilesRef = useRef<Record<string, unknown>>({});
  const ttsPrefetchRef = useRef<ReturnType<typeof prefetchModelFiles> | null>(null);
  const prefetchAbortRef = useRef<AbortController | null>(null);
  const prefetchEnabledRef = useRef(debugParam("debugLoadPrefetch") !== "0");
  const modelRuntimeLeaseRef = useRef<ModelRuntimeLease | null>(null);
  const modelRuntimeLeasePromiseRef = useRef<Promise<boolean> | null>(null);

  // Status
  const [llmStatus, setLlmStatus] = useState<"idle" | "loading" | "ready">("idle");
  const [sttStatus, setSttStatus] = useState<"idle" | "loading" | "ready">("idle");
  const [loadingMessage, setLoadingMessage] = useState("");
  const [modelLoadError, setModelLoadError] = useState<string | null>(null);
  const [progressItems, setProgressItems] = useState<ProgressItem[]>([]);
  const [sttProgressItems, setSttProgressItems] = useState<ProgressItem[]>([]);
  const loadStartRef = useRef<Record<"llm" | "stt" | "tts", number | null>>({
    llm: null,
    stt: null,
    tts: null,
  });
  const [loadTimes, setLoadTimes] = useState<Partial<Record<"llm" | "stt" | "tts", number>>>({});
  const [memoryStats, setMemoryStats] = useState<{
    heapUsedMb: number | null;
    heapLimitMb: number | null;
    deviceGb: number | null;
  }>({ heapUsedMb: null, heapLimitMb: null, deviceGb: null });

  // Display state - the main text shown
  const [transcribedText, setTranscribedText] = useState<string | null>(null);
  const [completionText, setCompletionText] = useState<string | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);

  // TTS state
  const [ttsEnabled] = useState(true);
  const ttsEnabledRef = useRef(ttsEnabled);
  useEffect(() => { ttsEnabledRef.current = ttsEnabled; }, [ttsEnabled]);
  const [ttsLoading, setTtsLoading] = useState(false);
  const [ttsProgress, setTtsProgress] = useState(0);
  const [voices, setVoices] = useState<string[]>([]);
  const [selectedVoice, setSelectedVoice] = useState("jean");
  const [customVoiceFile, setCustomVoiceFile] = useState<File | null>(null);
  const [voiceSetupComplete, setVoiceSetupComplete] = useState(hasDebugAutorunParams);
  // True while re-recording from the main view, where skipping keeps today's voice.
  const [isReRecordingVoice, setIsReRecordingVoice] = useState(false);
  const selectedVoiceRef = useRef(selectedVoice);
  const customVoiceFileRef = useRef(customVoiceFile);
  const ttsRef = useRef<PocketTTSOnnx | null>(null);
  const ttsLoadedRef = useRef(false);
  const ttsLoadPromiseRef = useRef<Promise<PocketTTSOnnx> | null>(null);
  const preparedVoiceKeyRef = useRef<string | null>(null);
  const voicePreparePromiseRef = useRef<{ key: string; promise: Promise<void> } | null>(null);

  const [contextMode, setContextMode] = useState(false);
  const contextModeRef = useRef(false);
  const conversationContextRef = useRef<ContextTurn[]>([]);
  const [contextTurns, setContextTurns] = useState(0);
  const rememberTurn = useCallback((transcript: string, completions: string[]) => {
    if (!contextModeRef.current) return;
    conversationContextRef.current = appendContextTurn(conversationContextRef.current, { transcript, completions });
    setContextTurns(conversationContextRef.current.length);
  }, []);

  // Branch mode state
  const [thoughtMode, setThoughtMode] = useState<"linear" | "branch" | "chorus">("chorus");
  const branchMode = thoughtMode !== "linear";
  const chorusModeRef = useRef(true);
  const [chorusVoiceCount, setChorusVoiceCount] = useState(8);
  const chorusVoiceCountRef = useRef(8);
  const activeChorusVoiceCountRef = useRef(8);
  const chorusFeedRef = useRef<ChorusFeed<Branch> | null>(null);
  const [chorusSpeakingIds, setChorusSpeakingIds] = useState<number[]>([]);
  const [chorusSpokenIds, setChorusSpokenIds] = useState<number[]>([]);
  // True once the whole Chorus has played (or was cut short): the stage
  // keeps every figure up until then, and lets them all go together.
  const [chorusFinished, setChorusFinished] = useState(false);
  const [chorusPreparing, setChorusPreparing] = useState(false);
  const [branches, setBranches] = useState<Branch[]>([]);
  const [speakingBranchId, setSpeakingBranchId] = useState<number | null>(null);
  const branchModeRef = useRef(branchMode);
  const branchTokenBuffersRef = useRef<Record<number, string>>({});

  // Overlap mode starts a completion while the user is still speaking.
  // No longer offered in the options; the plumbing stays for debug runs.
  const [overlapMode] = useState(false);
  const [overlapDelayMs] = useState(3000);
  const overlapModeRef = useRef(overlapMode);
  const overlapDelayMsRef = useRef(overlapDelayMs);
  const overlapTriggeredForUtteranceRef = useRef(false);
  const overlapCompletionActiveRef = useRef(false);

  // Speculative mode transcribes and completes while the user is still
  // speaking, then commits at the endpoint. See ./speculation.ts.
  // The speculative pipeline has no switch in the options; it stays on unless
  // a debug URL turns it off.
  const [speculativeMode] = useState(initialSpeculativeMode);
  const speculativeModeRef = useRef(speculativeMode);
  const [completionModelKey] = useState(initialCompletionModelKey);
  const speculationRef = useRef(new SpeculationCoordinator(initialSpeculationOptions()));
  const ttsStartBufferMsRef = useRef(initialTtsStartBufferMs());
  const earlyStopRef = useRef(initialEarlyStop());
  const keepWarmRef = useRef(initialKeepWarm());

  // Keep refs in sync
  useEffect(() => {
    selectedVoiceRef.current = selectedVoice;
  }, [selectedVoice]);
  useEffect(() => {
    customVoiceFileRef.current = customVoiceFile;
  }, [customVoiceFile]);
  useEffect(() => {
    branchModeRef.current = branchMode;
  }, [branchMode]);
  useEffect(() => {
    overlapModeRef.current = overlapMode;
  }, [overlapMode]);
  useEffect(() => {
    overlapDelayMsRef.current = overlapDelayMs;
  }, [overlapDelayMs]);

  useEffect(() => {
    if (!coreLoadRequestedRef.current) return;
    if (llmStatus === "ready" && sttStatus === "idle") {
      const id = window.setTimeout(() => {
        setLoadingMessage("Loading speech recognition...");
        loadStartRef.current.stt = performance.now();
        whisperWorkerRef.current?.postMessage({ type: "load" });
      }, 0);
      return () => window.clearTimeout(id);
    }
  }, [llmStatus, sttStatus]);

  // VAD state
  const vadRef = useRef<MicVAD | null>(null);
  const debugMicContextRef = useRef<AudioContext | null>(null);
  const debugMicSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const debugMicFixtureRef = useRef<string | null>(null);
  const debugMicRunRef = useRef<DebugMicrophoneRun | null>(null);
  const [vadLoading, setVadLoading] = useState(false);
  const [vadError, setVadError] = useState<string | null>(null);
  const [micEnabled, setMicEnabled] = useState(false);
  const [optionsOpen, setOptionsOpen] = useState(false);
  const optionsPanelRef = useRef<HTMLDivElement | null>(null);
  const optionsToggleRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!optionsOpen) return;
    const closeOnOutsidePointer = (event: PointerEvent) => {
      const target = event.target;
      if (!(target instanceof Node)) return;
      if (optionsPanelRef.current?.contains(target) || optionsToggleRef.current?.contains(target)) return;
      setOptionsOpen(false);
    };
    document.addEventListener("pointerdown", closeOnOutsidePointer, true);
    return () => document.removeEventListener("pointerdown", closeOnOutsidePointer, true);
  }, [optionsOpen]);
  const [userSpeaking, setUserSpeaking] = useState(false);
  const micEnabledRef = useRef(false);
  const ttsPlaybackActiveRef = useRef(false);
  const ttsVadIgnoreUntilRef = useRef(0);
  const ttsVadResumeTimerRef = useRef<number | null>(null);
  // Whether the voice detector is running, so a reset can bring it straight
  // back instead of waiting for the playback it cut short to wind down.
  const vadListeningRef = useRef(false);
  // Bumped by every reset; playback that ends under a newer epoch was cut
  // short on purpose and must not re-impose the post-speech guard.
  const resetEpochRef = useRef(0);

  // Thought completion state
  const pauseFrameCountRef = useRef(0);
  const audioBufferRef = useRef<Float32Array[]>([]);
  const preSpeechBufferRef = useRef<Float32Array[]>([]);
  const isCapturingSpeechRef = useRef(false);
  const isCompletingRef = useRef(false);
  const speechStartTimeRef = useRef<number | null>(null);
  const lastSpeechTimeRef = useRef<number | null>(null);
  const currentPlayerRef = useRef<AudioPlayer | null>(null);
  const ttsRunStartedAtRef = useRef<number | null>(null);
  const ttsFirstAudioMsRef = useRef<number | null>(null);
  const ttsUnderrunCountRef = useRef<number | null>(null);
  const ttsUnderrunMsRef = useRef<number | null>(null);
  const speakTextRef = useRef<(text: string, signal?: AbortSignal, chorus?: AsyncIterable<Branch>) => Promise<void>>(async () => {});
  const branchSpeechQueueRef = useRef(new SpeechQueue());
  const debugPipelineActiveRef = useRef(false);
  const debugAutorunKeyRef = useRef<string | null>(null);
  const debugVadPromiseRef = useRef<Promise<NonRealTimeVAD> | null>(null);
  const completionAbortControllerRef = useRef<AbortController | null>(null);
  const tokenBufferRef = useRef<string>("");
  const latestCompletionTextRef = useRef<string>("");
  const activePartialTextRef = useRef<string>("");
  const interruptCompletionRef = useRef<() => void>(() => {});
  const triggerThoughtCompletionRef = useRef<(options?: { overlap?: boolean }) => void>(() => {});

  // Thresholds
  const FRAME_DURATION_MS = 96;
  const MAX_AUDIO_BUFFER_MS = 8000;
  const MAX_AUDIO_BUFFER_SAMPLES = Math.ceil((MAX_AUDIO_BUFFER_MS / FRAME_DURATION_MS) * 1536);
  const PRE_SPEECH_FRAME_COUNT = 3;
  const TTS_VAD_RESUME_DELAY_MS = 700;

  const [isWebGPUAvailable] = useState<boolean | null>(() =>
    typeof navigator !== "undefined" &&
    !!(navigator as Navigator & { gpu?: unknown }).gpu
  );

  useEffect(() => {
    micEnabledRef.current = micEnabled;
  }, [micEnabled]);

  useEffect(() => {
    const readMemory = () => {
      const perf = performance as PerformanceWithMemory;
      const memory = perf.memory;
      setMemoryStats({
        heapUsedMb: memory ? Math.round(memory.usedJSHeapSize / 1024 / 1024) : null,
        heapLimitMb: memory ? Math.round(memory.jsHeapSizeLimit / 1024 / 1024) : null,
        deviceGb:
          "deviceMemory" in navigator && typeof navigator.deviceMemory === "number"
            ? navigator.deviceMemory
            : null,
      });
    };

    readMemory();
    const id = window.setInterval(readMemory, 2000);
    return () => window.clearInterval(id);
  }, []);

  function recordLoadProfile(kind: string, profile: Record<string, unknown>) {
    loadProfilesRef.current[kind] = profile;
    if (isLocalDebugHost()) {
      document.documentElement.dataset.voicesLoadProfile = JSON.stringify(loadProfilesRef.current);
    }
  }

  function prefetchTTS() {
    if (!coreLoadRequestedRef.current || !ttsEnabledRef.current || !prefetchEnabledRef.current || ttsPrefetchRef.current) return;
    prefetchAbortRef.current ??= new AbortController();
    ttsPrefetchRef.current = prefetchModelFiles(
      POCKET_MODEL_CACHE, pocketPrefetchUrls(),
      AbortSignal.any([prefetchAbortRef.current.signal, AbortSignal.timeout(300_000)]),
    );
  }

  function markLoadStart(kind: "llm" | "stt" | "tts") {
    loadStartRef.current[kind] = performance.now();
  }

  function markLoadReady(kind: "llm" | "stt" | "tts") {
    const startedAt = loadStartRef.current[kind];
    if (!startedAt) return;

    const seconds = (performance.now() - startedAt) / 1000;
    setLoadTimes((prev) => ({ ...prev, [kind]: seconds }));
    loadStartRef.current[kind] = null;
  }

  function pushAudioFrame(frame: Float32Array) {
    audioBufferRef.current.push(new Float32Array(frame));

    let retainedSamples = 0;
    for (let i = audioBufferRef.current.length - 1; i >= 0; i--) {
      retainedSamples += audioBufferRef.current[i].length;
      if (retainedSamples > MAX_AUDIO_BUFFER_SAMPLES) {
        audioBufferRef.current.splice(0, i + 1);
        break;
      }
    }
  }

  async function closeDebugMicrophoneSource() {
    const source = debugMicSourceRef.current;
    const context = debugMicContextRef.current;
    debugMicSourceRef.current = null;
    debugMicContextRef.current = null;
    try {
      source?.stop();
    } catch {
      // The fixture may already have reached its scheduled end.
    }
    source?.disconnect();
    await context?.close().catch(() => {});
  }

  async function createDebugMicrophoneStream(name: string) {
    void closeDebugMicrophoneSource();
    const AudioContextClass = getAudioContextConstructor();
    if (!AudioContextClass) throw new Error("Web Audio is not supported");
    const context = new AudioContextClass();
    debugMicContextRef.current = context;
    const updateContextState = () => {
      document.documentElement.dataset.voicesMicDebugContextState = context.state;
    };
    context.addEventListener("statechange", updateContextState);
    updateContextState();
    const resumePromise = context.resume().catch(() => {});
    const response = await fetch(assetUrl(`/api/debug-audio?name=${encodeURIComponent(name)}`));
    if (!response.ok) throw new Error(`Failed to load microphone fixture: ${response.status}`);

    const decoded = await context.decodeAudioData(await response.arrayBuffer());
    const silenceSamples = Math.round(decoded.sampleRate * 2);
    const buffer = context.createBuffer(1, decoded.length + silenceSamples, decoded.sampleRate);
    const channel = buffer.getChannelData(0);
    for (let inputChannel = 0; inputChannel < decoded.numberOfChannels; inputChannel++) {
      const input = decoded.getChannelData(inputChannel);
      for (let index = 0; index < input.length; index++) {
        channel[index] += input[index] / decoded.numberOfChannels;
      }
    }

    const source = context.createBufferSource();
    const destination = context.createMediaStreamDestination();
    source.buffer = buffer;
    source.connect(destination);
    debugMicSourceRef.current = source;
    debugMicFixtureRef.current = name;
    debugMicRunRef.current = {
      fixture: name,
      startedAt: performance.now() + 1000,
      vadFrameCount: 0,
      speechStartCount: 0,
      sttStartedAt: null,
      sttMs: null,
      llmStartedAt: null,
      llmFirstTokenMs: null,
      llmMs: null,
      speculation: null,
      llmQueuedAt: null,
      llmDoneAt: null,
      llmTimings: null,
      speakCalledAt: null,
      ttsTimings: null,
      ttsMetrics: null,
      playbackStartedAt: null,
    };
    document.documentElement.dataset.voicesMicDebugStatus = "running";
    delete document.documentElement.dataset.voicesMicDebugResult;
    await Promise.race([
      resumePromise,
      new Promise<void>((resolve) => window.setTimeout(resolve, 250)),
    ]);
    updateContextState();
    if (context.state !== "running") {
      throw new Error(
        "The browser suspended the debug audio clock; use a live microphone or a browser that allows local audio playback.",
      );
    }
    source.start(context.currentTime + 1);
    return destination.stream;
  }

  function unlockAudio() {
    try {
      const AudioContextClass = getAudioContextConstructor();
      if (!AudioContextClass) return;
      const audioContext = new AudioContextClass();
      void audioContext.resume().then(() => {
        const source = audioContext.createBufferSource();
        source.buffer = audioContext.createBuffer(1, 1, audioContext.sampleRate);
        source.connect(audioContext.destination);
        source.onended = () => {
          source.disconnect();
          void audioContext.close().catch(() => {});
        };
        source.start();
      });
    } catch {
      // Audio unlock is best-effort and depends on browser gesture policy.
    }
  }

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

    if (voice === "record") {
      void reopenVoiceRecording();
      return;
    }

    selectedVoiceRef.current = voice;
    setSelectedVoice(voice);
  }

  function selectCustomVoiceFile(file: File | null) {
    if (!file) return;
    customVoiceFileRef.current = file;
    selectedVoiceRef.current = "custom";
    setCustomVoiceFile(file);
    setSelectedVoice("custom");
  }

  /** Sends the user back to the enrollment screen, with the mic left idle. */
  async function reopenVoiceRecording() {
    if (micEnabledRef.current || micEnabled) {
      micEnabledRef.current = false;
      setMicEnabled(false);
      setUserSpeaking(false);
      vadListeningRef.current = false;
      await vadRef.current?.pause().catch(() => {});
    }
    currentPlayerRef.current?.abort();
    setIsReRecordingVoice(true);
    setVoiceSetupComplete(false);
  }

  /** Makes a freshly recorded clip the active voice; throws if it cannot be encoded. */
  async function applyRecordedVoice(file: File) {
    const previousVoice = selectedVoiceRef.current;
    const previousFile = customVoiceFileRef.current;

    selectedVoiceRef.current = "custom";
    customVoiceFileRef.current = file;
    setSelectedVoice("custom");
    setCustomVoiceFile(file);

    try {
      const tts = await ensureTTSLoaded();
      await prepareTTSVoice(tts, "custom", file);
      // Also builds the cached conditioning state for this voice.
      await warmTTSGraphs(tts);
    } catch (error) {
      // Keep the previous voice rather than leaving the app on one that failed.
      selectedVoiceRef.current = previousVoice;
      customVoiceFileRef.current = previousFile;
      setSelectedVoice(previousVoice);
      setCustomVoiceFile(previousFile);
      throw error;
    }

    setIsReRecordingVoice(false);
    setVoiceSetupComplete(true);
  }

  function getVoicePreparationKey(voice: string, customFile: File | null) {
    if (voice !== "custom") return `preset:${voice}`;
    if (!customFile) return null;
    return `custom:${customFile.name}:${customFile.size}:${customFile.lastModified}`;
  }

  async function prepareTTSVoice(
    tts: PocketTTSOnnx,
    voice = selectedVoiceRef.current,
    customFile = customVoiceFileRef.current,
  ) {
    const key = getVoicePreparationKey(voice, customFile);
    if (!key) return;
    if (preparedVoiceKeyRef.current === key) return;
    if (voicePreparePromiseRef.current?.key === key) {
      return voicePreparePromiseRef.current.promise;
    }

    const promise = (async () => {
      if (voice === "custom") {
        if (!customFile) return;
        await tts.encodeVoice(customFile);
        await tts.setVoice("custom");
      } else {
        await tts.setVoice(voice);
      }
      preparedVoiceKeyRef.current = key;
    })().finally(() => {
      if (voicePreparePromiseRef.current?.key === key) {
        voicePreparePromiseRef.current = null;
      }
    });

    voicePreparePromiseRef.current = { key, promise };
    return promise;
  }

  /**
   * Runs one throwaway generation so WebGPU compiles the speech graphs before
   * the first real utterance instead of during it. No player is attached, so
   * the audio callback drops every chunk.
   */
  async function warmTTSGraphs(tts: PocketTTSOnnx) {
    if (currentPlayerRef.current || ttsPlaybackActiveRef.current) return;
    try {
      await tts.generate("Warming up.");
    } catch (error) {
      console.warn("TTS warmup failed:", error);
    }
  }

  async function ensureTTSLoaded(): Promise<PocketTTSOnnx> {
    if (ttsLoadPromiseRef.current) return ttsLoadPromiseRef.current;
    if (ttsLoadedRef.current && ttsRef.current) return ttsRef.current;

    ttsLoadPromiseRef.current = (async () => {
      try {
        setTtsLoading(true);
        markLoadStart("tts");
        const loadSignal = prefetchAbortRef.current?.signal;
        const prefetchWaitStartedAt = performance.now();
        const prefetchResult = await ttsPrefetchRef.current;
        if (loadSignal?.aborted) throw new Error("Model loading cancelled");
        const prefetchWaitMs = performance.now() - prefetchWaitStartedAt;
        if (!ttsRef.current) {
          ttsRef.current = new PocketTTSOnnx({
            onLoadProfile: (profile) => recordLoadProfile("ttsFiles", profile),
            onStatus: (message) => {
              // Generation statuses arrive mid-playback; re-rendering the page
              // for each one only competes with audio scheduling.
              if (ttsPlaybackActiveRef.current) return;
              setLoadingMessage(message);
            },
            onProgress: (progress) => {
              setTtsProgress(progress);
              setLoadingMessage(`Downloading TTS... ${(progress * 100).toFixed(0)}%`);
            },
            onAudio: (audio) => {
              if (ttsFirstAudioMsRef.current === null && ttsRunStartedAtRef.current !== null) {
                ttsFirstAudioMsRef.current = performance.now() - ttsRunStartedAtRef.current;
              }
              const player = currentPlayerRef.current;
              if (!player) return;
              const chunk = audio instanceof Float32Array ? audio : new Float32Array(audio);
              player.playChunk(chunk);
            },
            onError: (err) => {
              console.error("TTS worker error:", err);
            },
            onMetrics: (metrics, phase) => {
              const run = debugMicRunRef.current;
              if (!run || ttsRunStartedAtRef.current === null) return;
              if (phase === "firstAudio") run.ttsTimings = metrics;
              else run.ttsMetrics = metrics;
            },
          });
        }

        await ttsRef.current.initialize({
          seed: Math.floor(Math.random() * 2 ** 32),
          temperature: 0.7,
          lsd: 1,
          residentState: initialResidentState(),
          flowCacheFrames: debugParam("debugFlowCache") === "1000" ? 1000 : 512,
        });
        const voiceList = await ttsRef.current.loadVoices();
        setVoices(voiceList);
        let voiceToPrepare = selectedVoiceRef.current;
        if (
          voiceList.length > 0 &&
          voiceToPrepare !== "custom" &&
          !voiceList.includes(voiceToPrepare)
        ) {
          voiceToPrepare = voiceList[0];
          selectedVoiceRef.current = voiceToPrepare;
          setSelectedVoice(voiceToPrepare);
        }
        ttsLoadedRef.current = true;
        const voiceStartedAt = performance.now();
        await prepareTTSVoice(ttsRef.current, voiceToPrepare, customVoiceFileRef.current);
        const warmupStartedAt = performance.now();
        setLoadingMessage("Warming up speech...");
        await warmTTSGraphs(ttsRef.current);
        recordLoadProfile("tts", {
          prefetchWaitMs, prefetch: prefetchResult,
          voicePreparationMs: warmupStartedAt - voiceStartedAt,
          warmupMs: performance.now() - warmupStartedAt,
        });
        markLoadReady("tts");
        return ttsRef.current;
      } finally {
        setTtsLoading(false);
        ttsLoadPromiseRef.current = null;
      }
    })();

    return ttsLoadPromiseRef.current;
  }

  async function speakText(text: string, signal?: AbortSignal, chorus?: AsyncIterable<Branch>) {
    if (signal?.aborted) return;
    if (!ttsEnabled || !text.trim()) {
      overlapCompletionActiveRef.current = false;
      overlapTriggeredForUtteranceRef.current = false;
      return;
    }

    const voice = selectedVoiceRef.current;
    const customFile = customVoiceFileRef.current;
    if (voice === "custom" && !customFile) {
      overlapCompletionActiveRef.current = false;
      overlapTriggeredForUtteranceRef.current = false;
      return;
    }

    setIsSpeaking(true);
    ttsPlaybackActiveRef.current = true;
    ttsRunStartedAtRef.current = performance.now();
    if (debugMicRunRef.current && debugMicRunRef.current.speakCalledAt === null) {
      debugMicRunRef.current.speakCalledAt = performance.now();
    }
    ttsFirstAudioMsRef.current = null;
    ttsUnderrunCountRef.current = null;
    ttsUnderrunMsRef.current = null;
    if (ttsVadResumeTimerRef.current !== null) {
      window.clearTimeout(ttsVadResumeTimerRef.current);
      ttsVadResumeTimerRef.current = null;
    }

    const debugMicFixture = debugMicFixtureRef.current;
    const shouldResumeVad = micEnabledRef.current && !!vadRef.current;
    // Stopping the microphone stream takes tens of milliseconds; it runs
    // alongside synthesis and is awaited before the microphone restarts.
    let vadPaused: Promise<void> = Promise.resolve();
    const epoch = resetEpochRef.current;
    // What the Chorus said, to remember as this turn once it has all played.
    const chorusSpoken: string[] = [];
    if (shouldResumeVad) {
      vadListeningRef.current = false;
      vadPaused = vadRef.current?.pause().catch(() => {}) ?? Promise.resolve();
      setUserSpeaking(false);
      audioBufferRef.current = [];
      preSpeechBufferRef.current = [];
      isCapturingSpeechRef.current = false;
      pauseFrameCountRef.current = 0;
      speechStartTimeRef.current = null;
      lastSpeechTimeRef.current = null;
    }

    try {
      const tts = await ensureTTSLoaded();
      if (signal?.aborted) return;

      const chorusBranchIds: number[] = [];
      const voiceCount = activeChorusVoiceCountRef.current;
      const chorusPlayer = chorus ? createStaggeredChorus(
        index => createStreamingPlayer({
          minBufferMs: ttsStartBufferMsRef.current,
          retainAudio: false,
          // The track is created on its first audio chunk, after its branch id
          // was pushed, so the utterance's seed is known: the voice sounds from
          // where its character pops out of the brain.
          pan: chorusVoicePan(chorusBranchIds[0] ?? 0, index, voiceCount),
          // Voices are mixed hot, gentler than equal power (1 / sqrt N): a
          // three-voice Chorus sits 2.9 dB down instead of 4.8, so a lone voice
          // in the stagger is nearly full volume. Coinciding peaks are caught by
          // the shared limiter rather than clipping at the output.
          gain: 1 / Math.pow(voiceCount, 0.3),
          limit: true,
          onStarted: () => {
            setChorusPreparing(false);
            setChorusSpeakingIds(ids => [...ids, chorusBranchIds[index]]);
          },
        }),
        index => {
          setChorusSpeakingIds(ids => ids.filter(id => id !== chorusBranchIds[index]));
          setChorusSpokenIds(ids => [...ids, chorusBranchIds[index]]);
        },
      ) : null;
      const player = chorusPlayer ?? createStreamingPlayer({
        minBufferMs: ttsStartBufferMsRef.current,
        retainAudio: false,
      });
      currentPlayerRef.current = player;
      const abortPlayback = () => player.abort();
      signal?.addEventListener("abort", abortPlayback, { once: true });
      try {
        await prepareTTSVoice(tts, voice, customFile);

        if (chorus && chorusPlayer) {
          setChorusPreparing(true);
          setChorusSpokenIds([]);
          setChorusFinished(false);
          for await (const branch of chorus) {
            chorusBranchIds.push(branch.id);
            if (signal?.aborted || player.aborted) return;
            await tts.generate(branch.text, signal);
            chorusSpoken.push(branch.text);
            chorusPlayer.finishTrack();
          }
        } else {
          await playTTSOnnx(player, tts, text, {
            voice,
            lsdDecodeSteps: 1,
            signal,
          });
        }
      } finally {
        const run = debugMicRunRef.current;
        if (run && run.playbackStartedAt === null) run.playbackStartedAt = player.startedAt;
        await player.close();
        signal?.removeEventListener("abort", abortPlayback);
        ttsUnderrunCountRef.current = player.underrunCount;
        ttsUnderrunMsRef.current = player.underrunMs;
        currentPlayerRef.current = null;
      }
    } catch (error) {
      console.error("TTS error:", error);
    } finally {
      setChorusPreparing(false);
      setChorusSpeakingIds([]);
      setChorusFinished(true);
      ttsPlaybackActiveRef.current = false;
      // Cut short by a reset: the detector is already back, no guard needed.
      const cutShort = resetEpochRef.current !== epoch;
      if (chorus && chorusSpoken.length > 0 && !cutShort) {
        rememberTurn(activePartialTextRef.current, chorusSpoken);
      }
      if (!cutShort) ttsVadIgnoreUntilRef.current = performance.now() + TTS_VAD_RESUME_DELAY_MS;
      await vadPaused;
      if (debugMicFixture) {
        const run = debugMicRunRef.current;
        if (run) {
          const since = (from: number | null, to: number | null) =>
            from === null || to === null ? null : Math.round(to - from);
          document.documentElement.dataset.voicesMicDebugStatus = "complete";
          document.documentElement.dataset.voicesMicDebugResult = JSON.stringify({
            ok: true,
            fixture: run.fixture,
            partialText: activePartialTextRef.current,
            completionText: text,
            vadFrameCount: run.vadFrameCount,
            speechStartCount: run.speechStartCount,
            sttMs: run.sttMs,
            llmFirstTokenMs: run.llmFirstTokenMs,
            llmMs: run.llmMs,
            speculation: run.speculation ?? speculationRef.current.statistics,
            ttsFirstAudioMs: ttsFirstAudioMsRef.current,
            ttsUnderrunCount: ttsUnderrunCountRef.current,
            ttsUnderrunMs: ttsUnderrunMsRef.current,
            responseMs: run.sttStartedAt === null
              ? null
              : performance.now() - run.sttStartedAt,
            totalMs: performance.now() - run.startedAt,
            // Endpoint to the first sample scheduled on the output device.
            deadAirMs: since(run.sttStartedAt, run.playbackStartedAt),
            profile: {
              endpointToLlmQueuedMs: since(run.sttStartedAt, run.llmQueuedAt),
              llmQueuedToDoneMs: since(run.llmQueuedAt, run.llmDoneAt),
              llmDoneToSpeakMs: since(run.llmDoneAt, run.speakCalledAt),
              speakToPlaybackMs: since(run.speakCalledAt, run.playbackStartedAt),
              llm: run.llmTimings,
              tts: run.ttsTimings,
              ttsGeneration: run.ttsMetrics,
            },
          });
          debugMicRunRef.current = null;
        }
        debugMicFixtureRef.current = null;
        micEnabledRef.current = false;
        setMicEnabled(false);
      } else if (shouldResumeVad && micEnabledRef.current && !cutShort) {
        ttsVadResumeTimerRef.current = window.setTimeout(() => {
          ttsVadResumeTimerRef.current = null;
          if (shouldRestartVadAfterPlayback({
            micEnabled: micEnabledRef.current,
            ttsPlaybackActive: ttsPlaybackActiveRef.current,
          }) && !vadListeningRef.current) {
            vadListeningRef.current = true;
            void vadRef.current?.start().catch(console.error);
          }
        }, TTS_VAD_RESUME_DELAY_MS);
      }
      setIsSpeaking(false);
      setTtsLoading(false);
      currentPlayerRef.current = null;
      ttsRunStartedAtRef.current = null;
      overlapCompletionActiveRef.current = false;
      overlapTriggeredForUtteranceRef.current = false;
    }
  }

  useEffect(() => {
    speakTextRef.current = speakText;
  });

  useEffect(() => {
    if (!ttsEnabled || !ttsLoadedRef.current || !ttsRef.current) return;

    void prepareTTSVoice(ttsRef.current, selectedVoice, customVoiceFile).catch((err) => {
      console.error("TTS voice preparation failed:", err);
    });
    // prepareTTSVoice reads current mutable TTS refs; re-run only when the selected source changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedVoice, customVoiceFile, ttsEnabled]);

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

  function selectThoughtMode(mode: "linear" | "branch" | "chorus") {
    if (mode === thoughtMode) return;
    resetAll(false);
    branchModeRef.current = mode !== "linear";
    chorusModeRef.current = mode === "chorus";
    setThoughtMode(mode);
  }

  async function loadCoreModels() {
    unlockAudio();
    setModelLoadError(null);
    if (!await ensureModelRuntimeLease()) {
      setModelLoadError(
        "Voice models are already loaded in another tab, or this browser cannot provide a safe model lock.",
      );
      return;
    }
    if (!prefetchAbortRef.current || prefetchAbortRef.current.signal.aborted) {
      prefetchAbortRef.current = new AbortController();
    }
    coreLoadRequestedRef.current = true;

    if (llmStatus === "idle") {
      setLoadingMessage("Loading language model...");
      markLoadStart("llm");
      workerRef.current?.postMessage({
        type: "load",
        data: { completionModel: COMPLETION_MODEL_PRESETS[completionModelKey] },
      });
      return;
    }

    if (llmStatus === "ready" && sttStatus === "idle") {
      setLoadingMessage("Loading speech recognition...");
      markLoadStart("stt");
      whisperWorkerRef.current?.postMessage({ type: "load" });
    }
  }

  useEffect(() => {
    if (!ttsEnabled) return;
    if (!coreLoadRequestedRef.current) return;
    if (llmStatus !== "ready" || sttStatus !== "ready") return;

    void ensureTTSLoaded().catch((err) => {
      console.error("TTS preload failed:", err);
    });
    // ensureTTSLoaded owns its in-flight promise; re-run only across model readiness or TTS toggles.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ttsEnabled, llmStatus, sttStatus]);

  async function runDebugTextPipeline(
    partialText: string,
    options: { speak?: boolean; measureMemory?: boolean; usePrefixCache?: boolean } = {},
  ) {
    const trimmedText = partialText.trim();
    if (!trimmedText) {
      throw new Error("Debug text pipeline requires non-empty text");
    }
    if (llmStatus !== "ready") {
      throw new Error("Language model is not ready");
    }
    if (!workerRef.current) {
      throw new Error("Language worker is not available");
    }
    if (debugPipelineActiveRef.current || isCompletingRef.current) {
      throw new Error("A completion is already running");
    }

    const worker = workerRef.current;
    const speak = options.speak ?? true;
    const shouldMeasureMemory = options.measureMemory ?? true;
    const memoryBeforeMb = shouldMeasureMemory ? await measurePageMemoryMb() : null;
    debugPipelineActiveRef.current = true;
    setLocalDebugPhase("llm");
    setTranscribedText(trimmedText);
    setCompletionText("");
    activePartialTextRef.current = trimmedText;
    const startedAt = performance.now();

    return new Promise<DebugPipelineResult>((resolve, reject) => {
      let completion = "";
      let settled = false;
      const llmStartedAt = performance.now();
      let llmFirstTokenMs: number | null = null;
      let llmTokens: number | null = null;
      let llmTps: number | null = null;
      let llmPrefixCacheUsed: boolean | null = null;

      const finish = () => {
        if (settled) return;
        settled = true;
        worker.removeEventListener("message", onDebugMessage);
      };

      const fail = (error: unknown) => {
        finish();
        debugPipelineActiveRef.current = false;
        reject(error instanceof Error ? error : new Error(String(error)));
      };

      const onDebugMessage = (e: MessageEvent<WorkerResponse>) => {
        if (e.data.status === "error") {
          fail(e.data.data ?? "Language worker error");
          return;
        }

        if (e.data.status === "update" && e.data.output && e.data.branchId === undefined) {
          llmFirstTokenMs ??= performance.now() - llmStartedAt;
          llmTokens = e.data.numTokens ?? llmTokens;
          llmTps = e.data.tps ?? llmTps;
          completion += e.data.output;
          setCompletionText(sanitizeThoughtCompletion(completion, trimmedText));
          return;
        }

        if (e.data.status !== "complete") return;

        const llmMs = performance.now() - llmStartedAt;
        llmFirstTokenMs = e.data.firstTokenMs ?? llmFirstTokenMs;
        llmTokens = e.data.numTokens ?? llmTokens;
        llmTps = e.data.tps ?? llmTps;
        llmPrefixCacheUsed = e.data.prefixCacheUsed ?? llmPrefixCacheUsed;
        finish();

        const spokenText = sanitizeThoughtCompletion(completion, trimmedText);
        rememberTurn(trimmedText, [spokenText]);
        const ttsStartedAt = performance.now();
        setLocalDebugPhase(speak && spokenText ? "tts" : "done");
        const speechPromise =
          speak && spokenText ? speakTextRef.current(spokenText) : Promise.resolve();

        speechPromise
          .then(async () => {
            const totalMs = performance.now() - startedAt;
            const ttsMs = performance.now() - ttsStartedAt;
            const memoryAfterMb = shouldMeasureMemory ? await measurePageMemoryMb() : null;
            debugPipelineActiveRef.current = false;
            setLocalDebugPhase("done");
            resolve({
              partialText: trimmedText,
              completionText: spokenText,
              llmFirstTokenMs,
              llmTokens,
              llmTps,
              llmPrefixCacheUsed,
              llmMs,
              ttsFirstAudioMs: speak ? ttsFirstAudioMsRef.current : null,
              ttsUnderrunCount: speak ? ttsUnderrunCountRef.current : null,
              ttsUnderrunMs: speak ? ttsUnderrunMsRef.current : null,
              ttsMs,
              totalMs,
              memoryBeforeMb,
              memoryAfterMb,
            });
          })
          .catch(fail);
      };

      worker.addEventListener("message", onDebugMessage);
      worker.postMessage({
        type: "generateCompletion",
        data: {
          partialText: trimmedText,
          context: contextModeRef.current ? conversationContextRef.current : undefined,
          usePrefixCache: options.usePrefixCache,
        },
      });
    });
  }

  async function runDebugAudioPipeline(
    audio: Float32Array,
    options: { speak?: boolean; measureMemory?: boolean; usePrefixCache?: boolean } = {},
  ) {
    if (sttStatus !== "ready") {
      throw new Error("Speech recognition model is not ready");
    }
    if (!whisperWorkerRef.current) {
      throw new Error("Speech worker is not available");
    }
    if (debugPipelineActiveRef.current || isCompletingRef.current) {
      throw new Error("A pipeline run is already active");
    }
    if (audio.length < 1000) {
      throw new Error("Debug audio is too short");
    }

    const whisperWorker = whisperWorkerRef.current;
    debugPipelineActiveRef.current = true;
    setLocalDebugPhase("stt");
    const sttStartedAt = performance.now();

    return new Promise<DebugPipelineResult>((resolve, reject) => {
      let settled = false;

      const finish = () => {
        if (settled) return;
        settled = true;
        whisperWorker.removeEventListener("message", onDebugWhisperMessage);
      };

      const fail = (error: unknown) => {
        finish();
        debugPipelineActiveRef.current = false;
        reject(error instanceof Error ? error : new Error(String(error)));
      };

      const onDebugWhisperMessage = (e: MessageEvent<WorkerResponse>) => {
        if (e.data.status === "error") {
          fail(e.data.data ?? "Speech worker error");
          return;
        }

        if (e.data.status !== "partialComplete") return;

        finish();
        const partialText = (e.data.output ?? "").trim().replace(/[.!?,;:]+$/, "");
        if (!partialText || partialText === "[BLANK_AUDIO]") {
          debugPipelineActiveRef.current = false;
          reject(new Error("Speech worker did not transcribe usable text"));
          return;
        }

        const sttMs = performance.now() - sttStartedAt;
        debugPipelineActiveRef.current = false;
        runDebugTextPipeline(partialText, options)
          .then((result) => {
            resolve({
              ...result,
              sttMs,
              // runDebugTextPipeline measures its own latency before the
              // optional browser-wide memory probe. Keep that diagnostic cost
              // out of the end-to-end voice latency as well.
              totalMs: sttMs + result.totalMs,
            });
          })
          .catch(reject);
      };

      whisperWorker.addEventListener("message", onDebugWhisperMessage);
      whisperWorker.postMessage({
        type: "generatePartial",
        data: { audio, language: "en" },
      }, [audio.buffer]);
    });
  }

  async function loadDebugAudio(url: string) {
    setLocalDebugPhase("audio-fetch");
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch debug audio: ${response.status}`);
    }

    const arrayBuffer = await response.arrayBuffer();
    setLocalDebugPhase("audio-decode");
    const AudioContextClass = getAudioContextConstructor();
    if (!AudioContextClass) {
      throw new Error("Web Audio is not available");
    }
    const audioContext = new AudioContextClass();
    try {
      const decoded = await audioContext.decodeAudioData(arrayBuffer.slice(0));
      const mono = new Float32Array(decoded.length);
      for (let channel = 0; channel < decoded.numberOfChannels; channel++) {
        const data = decoded.getChannelData(channel);
        for (let i = 0; i < data.length; i++) {
          mono[i] += data[i] / decoded.numberOfChannels;
        }
      }

      return decoded.sampleRate === 16000
        ? mono
        : resampleAudio(mono, decoded.sampleRate, 16000);
    } finally {
      await audioContext.close().catch(() => {});
    }
  }

  async function segmentDebugAudioWithVad(audio: Float32Array) {
    setLocalDebugPhase("vad");
    const startedAt = performance.now();
    debugVadPromiseRef.current ??= NonRealTimeVAD.new({
      modelURL: "/silero_vad_legacy.onnx",
      ortConfig: (ort) => {
        ort.env.wasm.wasmPaths = "/";
        ort.env.wasm.numThreads = 1;
      },
      modelFetcher: async (url) => {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Failed to load debug VAD model: ${response.status}`);
        }
        return response.arrayBuffer();
      },
    }).catch((error) => {
      debugVadPromiseRef.current = null;
      throw error;
    });

    const vad = await debugVadPromiseRef.current;
    const segments: Array<{ audio: Float32Array; start: number; end: number }> = [];
    for await (const segment of vad.run(audio, 16000)) {
      if (segment.audio.length > 0) segments.push(segment);
    }
    if (segments.length === 0) {
      throw new Error("Silero VAD did not detect speech in the debug fixture");
    }

    const longest = segments.reduce((best, segment) =>
      segment.audio.length > best.audio.length ? segment : best
    );
    return {
      audio: longest.audio.slice(),
      vadMs: performance.now() - startedAt,
      vadSegmentCount: segments.length,
      vadSpeechMs: (longest.audio.length / 16000) * 1000,
    };
  }

  async function loadDebugCustomVoice(name: string) {
    const response = await fetch(assetUrl(`/api/debug-audio?name=${encodeURIComponent(name)}`));
    if (!response.ok) {
      throw new Error(`Failed to fetch debug voice: ${response.status}`);
    }
    return new File([await response.blob()], name, { type: "audio/wav" });
  }

  useEffect(() => {
    if (!isLocalDebugHost()) return;

    const emitDebugResult = (id: string, payload: unknown) => {
      document.dispatchEvent(new CustomEvent("voices-debug:result", {
        detail: { id, ok: true, payload },
      }));
    };
    const emitDebugError = (id: string, error: unknown) => {
      document.dispatchEvent(new CustomEvent("voices-debug:result", {
        detail: {
          id,
          ok: false,
          error: error instanceof Error ? error.message : String(error),
        },
      }));
    };
    const getDebugState = () => ({
      llmStatus,
      sttStatus,
      ttsBackend: "pocket" as const,
      ttsEnabled,
      isSpeaking,
      memoryStats,
      loadTimes,
    });
    const onDebugState = (event: Event) => {
      const id = (event as CustomEvent<{ id?: string }>).detail?.id ?? crypto.randomUUID();
      emitDebugResult(id, getDebugState());
    };
    const onDebugText = (event: Event) => {
      const detail = (event as CustomEvent<{
        id?: string;
        partialText?: string;
        speak?: boolean;
        measureMemory?: boolean;
        usePrefixCache?: boolean;
      }>).detail ?? {};
      const id = detail.id ?? crypto.randomUUID();
      runDebugTextPipeline(detail.partialText ?? "", {
        speak: detail.speak,
        measureMemory: detail.measureMemory,
        usePrefixCache: detail.usePrefixCache,
      })
        .then((result) => emitDebugResult(id, result))
        .catch((error) => emitDebugError(id, error));
    };
    const onDebugAudio = (event: Event) => {
      const detail = (event as CustomEvent<{
        id?: string;
        audio?: Float32Array;
        speak?: boolean;
        measureMemory?: boolean;
        usePrefixCache?: boolean;
      }>).detail ?? {};
      const id = detail.id ?? crypto.randomUUID();
      if (!detail.audio) {
        emitDebugError(id, "Debug audio is required");
        return;
      }
      runDebugAudioPipeline(detail.audio, {
        speak: detail.speak,
        measureMemory: detail.measureMemory,
        usePrefixCache: detail.usePrefixCache,
      })
        .then((result) => emitDebugResult(id, result))
        .catch((error) => emitDebugError(id, error));
    };

    document.documentElement.dataset.voicesDebug = "ready";
    document.addEventListener("voices-debug:get-state", onDebugState);
    document.addEventListener("voices-debug:run-text", onDebugText);
    document.addEventListener("voices-debug:run-audio", onDebugAudio);

    window.__voicesDebug = {
      getState: getDebugState,
      measureMemory: measurePageMemoryMb,
      loadCoreModels,
      runTextPipeline: runDebugTextPipeline,
      runAudioPipeline: runDebugAudioPipeline,
    };

    return () => {
      if (window.__voicesDebug?.runTextPipeline === runDebugTextPipeline) {
        delete window.__voicesDebug;
      }
      delete document.documentElement.dataset.voicesDebug;
      document.removeEventListener("voices-debug:get-state", onDebugState);
      document.removeEventListener("voices-debug:run-text", onDebugText);
      document.removeEventListener("voices-debug:run-audio", onDebugAudio);
    };
  });

  useEffect(() => {
    if (!isLocalDebugHost()) return;
    if (llmStatus !== "ready" || sttStatus !== "ready" || ttsLoading) return;

    const params = new URLSearchParams(window.location.search);
    const debugTexts = params.getAll("debugText").map((text) => text.trim()).filter(Boolean);
    const debugText = debugTexts[0] ?? null;
    const debugAudio = params.get("debugAudio");
    const debugMicFixture = params.get("debugMicFixture");
    const debugVad = params.get("debugVad") === "1";
    const debugCustomVoice = params.get("debugCustomVoice");
    if (!debugText && !debugAudio && !debugMicFixture) return;

    const speak = params.get("debugSpeak") !== "0";
    const measureMemory = params.get("debugMemory") !== "0";
    const completionCacheModes = params.getAll("debugCompletionCache");
    const parsedRuns = Number.parseInt(params.get("debugRuns") ?? "1", 10);
    const runCount = Number.isFinite(parsedRuns) ? Math.min(10, Math.max(1, parsedRuns)) : 1;
    const key = JSON.stringify({
      debugText,
      debugTexts,
      debugAudio,
      debugMicFixture,
      debugVad,
      debugCustomVoice,
      speak,
      measureMemory,
      completionCacheModes,
      runCount,
    });
    if (debugAutorunKeyRef.current === key) return;
    debugAutorunKeyRef.current = key;

    document.documentElement.dataset.voicesDebugStatus = "running";
    setLocalDebugPhase(debugMicFixture ? "mic-setup" : debugAudio ? "audio-fetch" : "llm");
    delete document.documentElement.dataset.voicesDebugResult;

    (async () => {
      if (debugCustomVoice) {
        const file = await loadDebugCustomVoice(debugCustomVoice);
        selectedVoiceRef.current = "custom";
        customVoiceFileRef.current = file;
        setSelectedVoice("custom");
        setCustomVoiceFile(file);
        const tts = await ensureTTSLoaded();
        await prepareTTSVoice(tts, "custom", file);
      }

      if (debugMicFixture) {
        document.documentElement.dataset.voicesDebugStatus = "mic-ready";
        document.documentElement.dataset.voicesDebugResult = JSON.stringify({
          ok: true,
          fixture: debugMicFixture,
          voice: debugCustomVoice ? "custom" : selectedVoiceRef.current,
        });
        return;
      }

      const runs: DebugPipelineResult[] = [];
      for (let index = 0; index < runCount; index++) {
        setLocalDebugPhase(`run-${index + 1}-${debugAudio ? "stt" : "llm"}`);
        const runText = debugTexts[index % debugTexts.length] ?? "";
        const usePrefixCache = completionCacheModes.length === 0 ||
          completionCacheModes[index % completionCacheModes.length] !== "0";
        let result: DebugPipelineResult;
        if (debugAudio) {
          const loadedAudio = await loadDebugAudio(debugAudio);
          const vadResult = debugVad
            ? await segmentDebugAudioWithVad(loadedAudio)
            : null;
          result = await runDebugAudioPipeline(vadResult?.audio ?? loadedAudio, {
            speak,
            measureMemory,
            usePrefixCache,
          });
          if (vadResult) {
            result = {
              ...result,
              vadMs: vadResult.vadMs,
              vadSegmentCount: vadResult.vadSegmentCount,
              vadSpeechMs: vadResult.vadSpeechMs,
              totalMs: result.totalMs + vadResult.vadMs,
            };
          }
        } else {
          result = await runDebugTextPipeline(runText, {
              speak,
              measureMemory,
              usePrefixCache,
            });
        }
        runs.push(result);
      }

      document.documentElement.dataset.voicesDebugStatus = "complete";
      document.documentElement.dataset.voicesDebugResult = JSON.stringify({
        ok: true,
        result: runs[runs.length - 1],
        runs,
      });
    })().catch((error) => {
      document.documentElement.dataset.voicesDebugStatus = "error";
      document.documentElement.dataset.voicesDebugResult = JSON.stringify({
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    });
  });

  async function speakBranch(branchId: number) {
    if (ttsPlaybackActiveRef.current || isCompletingRef.current) return;
    const branch = branches.find(b => b.id === branchId);
    if (!branch || !branch.text.trim()) return;

    await branchSpeechQueueRef.current.enqueue(async () => {
      setSpeakingBranchId(branchId);
      try {
        await speakTextRef.current(branch.text);
      } finally {
        setSpeakingBranchId(prev => prev === branchId ? null : prev);
      }
    });
  }

  function collectUtteranceAudio(): Float32Array | null {
    const frames = audioBufferRef.current;
    const totalLength = frames.reduce((sum, frame) => sum + frame.length, 0);
    if (totalLength < 1000) return null;
    const combined = new Float32Array(totalLength);
    let offset = 0;
    for (const frame of frames) {
      combined.set(frame, offset);
      offset += frame.length;
    }
    return combined;
  }

  /**
   * Carries out what the coordinator decided: worker messages, UI updates,
   * and speaking a committed completion. Actions may cascade, so the list is
   * processed in order.
   */
  const applySpeculationActions = useCallback((actions: SpeculationAction[]) => {
    const coordinator = speculationRef.current;
    for (const action of actions) {
      switch (action.type) {
        case "stt": {
          const audio = collectUtteranceAudio();
          if (!audio || !whisperWorkerRef.current) {
            applySpeculationActions(
              coordinator.onSttDropped({ requestId: action.requestId, now: performance.now() }),
            );
            break;
          }
          // The buffer keeps growing, so the pass gets its own copy.
          whisperWorkerRef.current.postMessage(
            { type: "generatePartial", data: { audio, language: "en", requestId: action.requestId } },
            [audio.buffer],
          );
          break;
        }
        case "llm-start": {
          const run = debugMicRunRef.current;
          if (run && run.llmQueuedAt === null) run.llmQueuedAt = performance.now();
          workerRef.current?.postMessage({
            type: "generateCompletion",
            data: {
              partialText: action.transcript,
              context: contextModeRef.current ? conversationContextRef.current : undefined,
              requestId: action.candidateId,
              postedAtEpoch: performance.timeOrigin + performance.now(),
              earlyStop: earlyStopRef.current,
            },
          });
          break;
        }
        case "llm-interrupt":
          workerRef.current?.postMessage({ type: "interrupt" });
          break;
        case "commit-transcript": {
          const now = performance.now();
          const run = debugMicRunRef.current;
          if (run && run.sttStartedAt !== null) {
            run.sttMs = now - run.sttStartedAt;
            run.llmStartedAt = now;
          }
          if (!action.transcript) {
            // Silence or noise: nothing to complete.
            isCompletingRef.current = false;
            overlapCompletionActiveRef.current = false;
            overlapTriggeredForUtteranceRef.current = false;
            break;
          }
          activePartialTextRef.current = action.transcript;
          setTranscribedText(action.transcript);
          break;
        }
        case "completion-update":
          if (!isCompletingRef.current) break;
          if (action.text) {
            const run = debugMicRunRef.current;
            if (run && run.llmFirstTokenMs === null && run.llmStartedAt !== null) {
              run.llmFirstTokenMs = performance.now() - run.llmStartedAt;
            }
            latestCompletionTextRef.current = action.text;
            setCompletionText(action.text);
          }
          break;
        case "commit-completion": {
          if (!isCompletingRef.current) break;
          if (!action.complete) {
            latestCompletionTextRef.current = action.text;
            setCompletionText(action.text || null);
            break;
          }
          const run = debugMicRunRef.current;
          if (run && run.llmStartedAt !== null) {
            run.llmMs = performance.now() - run.llmStartedAt;
            if (run.llmFirstTokenMs === null && action.text) run.llmFirstTokenMs = run.llmMs;
            run.speculation = coordinator.statistics;
            run.llmDoneAt = performance.now();
          }
          rememberTurn(activePartialTextRef.current, [action.text]);
          setCompletionText(action.text || null);
          if (action.text && !completionAbortControllerRef.current?.signal.aborted) {
            const signal = completionAbortControllerRef.current?.signal;
            void speakTextRef.current(action.text, signal);
          } else {
            overlapCompletionActiveRef.current = false;
            overlapTriggeredForUtteranceRef.current = false;
          }
          isCompletingRef.current = false;
          tokenBufferRef.current = "";
          latestCompletionTextRef.current = "";
          break;
        }
      }
    }
  }, [rememberTurn]);

  function speculationActive() {
    return speculativeModeRef.current && !branchModeRef.current;
  }

  const interruptCompletion = useCallback(() => {
    if (isCompletingRef.current) {
      chorusFeedRef.current?.finish();
      chorusFeedRef.current = null;
      workerRef.current?.postMessage({ type: "interrupt" });
      completionAbortControllerRef.current?.abort();
      currentPlayerRef.current?.abort();
      isCompletingRef.current = false;
      overlapCompletionActiveRef.current = false;
      overlapTriggeredForUtteranceRef.current = false;
      setCompletionText(null);
      tokenBufferRef.current = "";
      latestCompletionTextRef.current = "";
      activePartialTextRef.current = "";
      // Clear branches in branch mode
      setBranches([]);
      setChorusFinished(false);
      branchTokenBuffersRef.current = {};
      setSpeakingBranchId(null);
    }
  }, []);

  useEffect(() => {
    interruptCompletionRef.current = interruptCompletion;
  }, [interruptCompletion]);

  const resetAll = useCallback((clearContext = true) => {
    if (clearContext) {
      conversationContextRef.current = [];
      setContextTurns(0);
    }
    chorusFeedRef.current?.finish();
    chorusFeedRef.current = null;
    applySpeculationActions(speculationRef.current.reset());
    // Stop any ongoing completion
    workerRef.current?.postMessage({ type: "interrupt" });
    completionAbortControllerRef.current?.abort();
    currentPlayerRef.current?.abort();
    // Listen again at once: no post-speech guard, no resume timer, and the
    // detector restarted here if playback had paused it.
    resetEpochRef.current++;
    ttsVadIgnoreUntilRef.current = 0;
    if (ttsVadResumeTimerRef.current !== null) {
      window.clearTimeout(ttsVadResumeTimerRef.current);
      ttsVadResumeTimerRef.current = null;
    }
    if (micEnabledRef.current && vadRef.current && !vadListeningRef.current) {
      vadListeningRef.current = true;
      void vadRef.current.start().catch(console.error);
    }
    isCompletingRef.current = false;
    overlapCompletionActiveRef.current = false;
    overlapTriggeredForUtteranceRef.current = false;

    // Clear all text
    setTranscribedText(null);
    setCompletionText(null);
    tokenBufferRef.current = "";
    latestCompletionTextRef.current = "";
    activePartialTextRef.current = "";

    // Clear branches
    setBranches([]);
    setChorusFinished(false);
    branchTokenBuffersRef.current = {};
    setSpeakingBranchId(null);

    // Clear audio buffer
    audioBufferRef.current = [];
    preSpeechBufferRef.current = [];
    isCapturingSpeechRef.current = false;
    pauseFrameCountRef.current = 0;
    speechStartTimeRef.current = null;
    lastSpeechTimeRef.current = null;
  }, [applySpeculationActions]);

  const triggerThoughtCompletion = useCallback(async (options?: { overlap?: boolean }) => {
    if (isCompletingRef.current || !whisperWorkerRef.current) return;

    // Speculative endpoint: the coordinator already has (or is computing) the
    // transcript, and may already have a completion for it.
    const coordinator = speculationRef.current;
    if (speculationActive() && !options?.overlap && coordinator.currentPhase === "listening") {
      isCompletingRef.current = true;
      overlapCompletionActiveRef.current = false;
      tokenBufferRef.current = "";
      latestCompletionTextRef.current = "";
      completionAbortControllerRef.current = new AbortController();
      if (debugMicRunRef.current) {
        debugMicRunRef.current.sttStartedAt = performance.now();
      }
      warmAudioOutput();
      applySpeculationActions(coordinator.finalize({ now: performance.now() }));
      audioBufferRef.current = [];
      isCapturingSpeechRef.current = false;
      return;
    }
    // Overlap and branch completions take the serial path; drop any
    // speculative state so a late result cannot leak into it.
    applySpeculationActions(coordinator.reset());

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
    overlapCompletionActiveRef.current = options?.overlap === true;
    tokenBufferRef.current = "";
    latestCompletionTextRef.current = "";
    completionAbortControllerRef.current = new AbortController();
    if (debugMicRunRef.current) {
      debugMicRunRef.current.sttStartedAt = performance.now();
    }
    warmAudioOutput();

    // Reset branch state when starting new completion
    if (branchModeRef.current) {
      setBranches([]);
      setChorusFinished(false);
      branchTokenBuffersRef.current = {};
      branchSpeechQueueRef.current = new SpeechQueue();
      setSpeakingBranchId(null);
    }

    whisperWorkerRef.current.postMessage({
      type: "generatePartial",
      data: { audio: combinedAudio, language: "en" },
    }, [combinedAudio.buffer]);
    audioBufferRef.current = [];
    isCapturingSpeechRef.current = false;
    // applySpeculationActions is stable; the rest are refs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
        case "weights-ready":
          prefetchTTS();
          break;
        case "loading":
          setSttStatus("loading");
          setLoadingMessage(e.data.data || "Loading speech recognition...");
          break;
        case "error":
          // A failed speculative pass is reissued by the coordinator; only an
          // error outside one means the model itself is unusable.
          if (speculationRef.current.ownsSttRequest(e.data.requestId)) {
            console.warn("Speculative transcription failed:", e.data.data);
            applySpeculationActions(speculationRef.current.onSttDropped({
              requestId: e.data.requestId!,
              now: performance.now(),
            }));
            break;
          }
          console.error("Speech worker error:", e.data.data);
          setSttStatus("idle");
          setLoadingMessage(e.data.data || "Speech recognition failed to load");
          isCompletingRef.current = false;
          overlapCompletionActiveRef.current = false;
          overlapTriggeredForUtteranceRef.current = false;
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
          // Finished files stay in the list, complete, so the load's total
          // measure holds steady.
          setSttProgressItems((prev) => prev.map((item) => (item.file === e.data.file ? { ...item, progress: item.total } : item)));
          break;
        case "ready":
          if (e.data.loadProfile) recordLoadProfile("stt", e.data.loadProfile);
          // The voice model loads next. Raise its flag in the same render, or
          // the enrollment screen shows for one frame before the TTS preload
          // effect runs and the loading screen returns.
          if (ttsEnabledRef.current && !ttsLoadedRef.current) setTtsLoading(true);
          setSttStatus("ready");
          markLoadReady("stt");
          break;
        case "complete":
          // Full transcription - not used in minimalist mode
          break;
        case "partialSuperseded":
          if (speculationRef.current.ownsSttRequest(e.data.requestId)) {
            applySpeculationActions(speculationRef.current.onSttDropped({
              requestId: e.data.requestId!,
              now: performance.now(),
            }));
          }
          break;
        case "partialComplete":
          if (speculationRef.current.ownsSttRequest(e.data.requestId)) {
            const output = typeof e.data.output === "string" ? e.data.output : "";
            applySpeculationActions(speculationRef.current.onSttResult({
              requestId: e.data.requestId!,
              transcript: output === "[BLANK_AUDIO]" ? "" : output,
              now: performance.now(),
            }));
            break;
          }
          if (e.data.output && typeof e.data.output === "string" && isCompletingRef.current) {
            if (debugMicRunRef.current?.sttStartedAt !== null && debugMicRunRef.current) {
              debugMicRunRef.current.sttMs =
                performance.now() - debugMicRunRef.current.sttStartedAt;
            }
            let partialText = e.data.output.trim();
            // Skip blank audio detection
            if (partialText === "[BLANK_AUDIO]") {
              isCompletingRef.current = false;
              overlapCompletionActiveRef.current = false;
              overlapTriggeredForUtteranceRef.current = false;
              break;
            }
            // Remove trailing punctuation so completion flows naturally
            partialText = partialText.replace(/[.!?,;:]+$/, "");
            if (partialText) {
              setTranscribedText(partialText);
              activePartialTextRef.current = partialText;
              if (debugMicRunRef.current) {
                debugMicRunRef.current.llmStartedAt = performance.now();
              }

              if (branchModeRef.current) {
                // Snapshot the count for this round; control changes affect the next one.
                const numBranches = chorusModeRef.current ? chorusVoiceCountRef.current : 3;
                activeChorusVoiceCountRef.current = numBranches;
                workerRef.current?.postMessage({
                  type: "generateBranches",
                  data: { partialText, numBranches, context: contextModeRef.current ? conversationContextRef.current : undefined },
                });
              } else {
                // Linear mode: single completion
                if (debugMicRunRef.current) {
                  debugMicRunRef.current.llmQueuedAt = performance.now();
                }
                workerRef.current?.postMessage({
                  type: "generateCompletion",
                  data: {
                    partialText,
                    context: contextModeRef.current ? conversationContextRef.current : undefined,
                    postedAtEpoch: performance.timeOrigin + performance.now(),
                    earlyStop: earlyStopRef.current,
                  },
                });
              }
            } else {
              isCompletingRef.current = false;
              overlapCompletionActiveRef.current = false;
              overlapTriggeredForUtteranceRef.current = false;
            }
          }
          break;
      }
    };

    whisperWorkerRef.current.addEventListener("message", onWhisperMessage);
    return () => {
      whisperWorkerRef.current?.removeEventListener("message", onWhisperMessage);
      whisperWorkerRef.current?.terminate();
      whisperWorkerRef.current = null;
    };
  }, [applySpeculationActions]);

  // Keep-warm ticks: only while both models are ready, the tab is visible,
  // and nothing is being captured, completed, or spoken.
  useEffect(() => {
    if (!keepWarmRef.current) return;
    if (llmStatus !== "ready" || sttStatus !== "ready") return;
    const tick = () => {
      if (document.visibilityState !== "visible") return;
      if (
        isCompletingRef.current ||
        isCapturingSpeechRef.current ||
        ttsPlaybackActiveRef.current ||
        debugPipelineActiveRef.current
      ) {
        return;
      }
      workerRef.current?.postMessage({ type: "keepWarm" });
      whisperWorkerRef.current?.postMessage({ type: "keepWarm" });
    };
    const id = window.setInterval(tick, MODEL_KEEP_WARM_MS);
    return () => window.clearInterval(id);
  }, [llmStatus, sttStatus]);

  async function toggleMic() {
    if (vadLoading) return;
    unlockAudio();

    if (micEnabled) {
      if (ttsVadResumeTimerRef.current !== null) {
        window.clearTimeout(ttsVadResumeTimerRef.current);
        ttsVadResumeTimerRef.current = null;
      }
      vadListeningRef.current = false;
      if (vadRef.current) await vadRef.current.pause();
      setMicEnabled(false);
      setUserSpeaking(false);
      overlapCompletionActiveRef.current = false;
      overlapTriggeredForUtteranceRef.current = false;
      applySpeculationActions(speculationRef.current.reset());
    } else {
      setVadError(null);
      const debugMicFixture = isLocalDebugHost()
        ? new URLSearchParams(window.location.search).get("debugMicFixture")
        : null;
      // A fixture run closes its audio context when it ends, so the VAD that
      // was built on it cannot be resumed; rebuild it for the next run.
      if (vadRef.current && debugMicFixture) {
        await vadRef.current.destroy().catch(() => {});
        vadRef.current = null;
        await closeDebugMicrophoneSource();
      }
      if (vadRef.current) {
        vadListeningRef.current = true;
        await vadRef.current.start();
        setMicEnabled(true);
      } else {
        setVadLoading(true);
        try {
          debugMicFixtureRef.current = debugMicFixture;
          let initialDebugStream = debugMicFixture
            ? createDebugMicrophoneStream(debugMicFixture)
            : null;
          const getMicStream = debugMicFixture
            ? () => {
                const stream = initialDebugStream ?? createDebugMicrophoneStream(debugMicFixture);
                initialDebugStream = null;
                return stream;
              }
            : () => requestMicrophoneStream(() =>
                navigator.mediaDevices.getUserMedia({
                  audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    autoGainControl: true,
                    noiseSuppression: true,
                  },
                })
              );
          const vad = await MicVAD.new({
            startOnLoad: true,
            baseAssetPath: `${BASE_PATH}/`,
            onnxWASMBasePath: `${BASE_PATH}/`,
            audioContext: debugMicFixture
              ? debugMicContextRef.current ?? undefined
              : undefined,
            getStream: getMicStream,
            pauseStream: async (stream) => {
              for (const track of stream.getTracks()) track.stop();
              if (debugMicFixtureRef.current) await closeDebugMicrophoneSource();
            },
            resumeStream: getMicStream,
            onSpeechEnd: () => {
              if (shouldSuppressVadInput({
                ttsPlaybackActive: ttsPlaybackActiveRef.current,
                now: performance.now(),
                ignoreUntil: ttsVadIgnoreUntilRef.current,
              })) {
                return;
              }
              setUserSpeaking(false);
              overlapTriggeredForUtteranceRef.current = false;

              // onFrameProcessed owns a continuous utterance buffer. If the
              // shorter thought-pause detector did not fire, VAD end is the
              // reliable fallback so short utterances are not discarded.
              if (shouldFinalizeOnVadEnd({
                bufferedFrameCount: audioBufferRef.current.length,
                isCapturing: isCapturingSpeechRef.current,
                isCompleting: isCompletingRef.current,
              })) {
                triggerThoughtCompletionRef.current();
              } else {
                audioBufferRef.current = [];
                isCapturingSpeechRef.current = false;
              }
              pauseFrameCountRef.current = 0;
              speechStartTimeRef.current = null;
              lastSpeechTimeRef.current = null;
            },
            onSpeechStart: () => {
              if (shouldSuppressVadInput({
                ttsPlaybackActive: ttsPlaybackActiveRef.current,
                now: performance.now(),
                ignoreUntil: ttsVadIgnoreUntilRef.current,
              })) {
                return;
              }
              if (debugMicRunRef.current) {
                debugMicRunRef.current.speechStartCount++;
              }
              setUserSpeaking(true);
              if (overlapCompletionActiveRef.current) return;

              overlapTriggeredForUtteranceRef.current = false;
              speechStartTimeRef.current = performance.now();
              const wasCapturing = isCapturingSpeechRef.current;
              const capture = beginSpeechCapture({
                audioFrames: audioBufferRef.current,
                preSpeechFrames: preSpeechBufferRef.current,
                isCapturing: isCapturingSpeechRef.current,
              });
              audioBufferRef.current = capture.audioFrames;
              preSpeechBufferRef.current = capture.preSpeechFrames;
              isCapturingSpeechRef.current = true;
              pauseFrameCountRef.current = 0;
              setTranscribedText(null);
              setCompletionText(null);
              interruptCompletionRef.current();
              if (!wasCapturing && speculationActive()) {
                applySpeculationActions(speculationRef.current.beginUtterance({
                  now: performance.now(),
                  initialSamples: audioBufferRef.current.reduce((sum, frame) => sum + frame.length, 0),
                }));
              }
            },
            onFrameProcessed: (probs, audioFrame) => {
              if (shouldSuppressVadInput({
                ttsPlaybackActive: ttsPlaybackActiveRef.current,
                now: performance.now(),
                ignoreUntil: ttsVadIgnoreUntilRef.current,
              })) {
                return;
              }
              if (debugMicRunRef.current) {
                debugMicRunRef.current.vadFrameCount++;
              }
              const now = performance.now();
              if (audioFrame && !isCapturingSpeechRef.current) {
                preSpeechBufferRef.current = appendRollingAudioFrame(
                  preSpeechBufferRef.current,
                  audioFrame,
                  PRE_SPEECH_FRAME_COUNT,
                );
              }
              const decision = decideVadFrame({
                isSpeech: probs.isSpeech,
                now,
                pauseFrameCount: pauseFrameCountRef.current,
                speechStartedAt: speechStartTimeRef.current,
                isCapturing: isCapturingSpeechRef.current,
                isCompleting: isCompletingRef.current,
                allowCompletionInterruption: !overlapCompletionActiveRef.current,
              });
              pauseFrameCountRef.current = decision.pauseFrameCount;
              let seededCaptureFromPreSpeech = false;

              if (decision.interruptCompletion) {
                  interruptCompletionRef.current();
                  speechStartTimeRef.current = now;
                  audioBufferRef.current = preSpeechBufferRef.current;
                  preSpeechBufferRef.current = [];
                  isCapturingSpeechRef.current = true;
                  seededCaptureFromPreSpeech = true;
                  if (speculationActive()) {
                    applySpeculationActions(speculationRef.current.beginUtterance({
                      now,
                      initialSamples: audioBufferRef.current.reduce((sum, frame) => sum + frame.length, 0),
                    }));
                  }
              }

              if (decision.userSpeaking === true) {
                setUserSpeaking(true);
                lastSpeechTimeRef.current = now;
              }

              if (decision.triggerCompletion) {
                triggerThoughtCompletionRef.current();
              }

              // Keep all frames between speech start and finalization. Dropping
              // medium-confidence frames clips quiet phonemes before STT sees
              // them, which is much more damaging than retaining brief pauses.
              const appended = !!audioFrame && decision.appendFrame && !seededCaptureFromPreSpeech;
              if (audioFrame && appended) {
                pushAudioFrame(audioFrame);
              }

              // Feed the coordinator after the frame is in the buffer so a
              // transcription pass it requests sees the whole utterance so far.
              if (isCapturingSpeechRef.current && speculationActive()) {
                applySpeculationActions(speculationRef.current.onFrame({
                  now,
                  probability: probs.isSpeech,
                  samples: audioFrame?.length ?? 1536,
                  appended,
                }));
              }

              if (shouldTriggerOverlapCompletion({
                enabled: overlapModeRef.current,
                delayMs: overlapDelayMsRef.current,
                now,
                speechStartedAt: speechStartTimeRef.current,
                isCapturing: isCapturingSpeechRef.current,
                isCompleting: isCompletingRef.current,
                alreadyTriggered: overlapTriggeredForUtteranceRef.current,
              })) {
                overlapTriggeredForUtteranceRef.current = true;
                triggerThoughtCompletionRef.current({ overlap: true });
              }
            },
          });
          vadRef.current = vad;
          vadListeningRef.current = true;
          setMicEnabled(true);
        } catch (error) {
          if (debugMicFixtureRef.current) {
            document.documentElement.dataset.voicesMicDebugStatus = "error";
            document.documentElement.dataset.voicesMicDebugResult = JSON.stringify({
              ok: false,
              error: error instanceof Error ? error.message : String(error),
            });
            debugMicFixtureRef.current = null;
            debugMicRunRef.current = null;
            await closeDebugMicrophoneSource();
          }
          setVadError(error instanceof Error ? error.message : String(error));
        } finally {
          setVadLoading(false);
        }
      }
    }
  }

  useEffect(() => {
    return () => {
      if (ttsVadResumeTimerRef.current !== null) {
        window.clearTimeout(ttsVadResumeTimerRef.current);
      }
      if (vadRef.current) {
        vadRef.current.destroy().catch(console.error);
      }
      void closeDebugMicrophoneSource();
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
        case "weights-ready":
          if (coreLoadRequestedRef.current && prefetchEnabledRef.current) {
            whisperWorkerRef.current?.postMessage({ type: "prefetch" });
          }
          break;
        case "start":
          if (debugMicRunRef.current && debugMicRunRef.current.llmStartedAt === null) {
            debugMicRunRef.current.llmStartedAt = performance.now();
          }
          break;
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
          setProgressItems((prev) => prev.map((item) => (item.file === e.data.file ? { ...item, progress: item.total } : item)));
          break;
        case "ready":
          if (e.data.loadProfile) recordLoadProfile("llm", e.data.loadProfile);
          setLlmStatus("ready");
          markLoadReady("llm");
          break;
        case "update":
          if (speculationRef.current.ownsLlmRequest(e.data.requestId)) {
            if (e.data.output) {
              applySpeculationActions(speculationRef.current.onLlmToken({
                requestId: e.data.requestId!,
                text: e.data.output,
              }));
            }
            break;
          }
          if (isCompletingRef.current && e.data.output) {
            const debugRun = debugMicRunRef.current;
            if (
              debugRun &&
              debugRun.llmFirstTokenMs === null &&
              debugRun.llmStartedAt !== null
            ) {
              debugRun.llmFirstTokenMs = performance.now() - debugRun.llmStartedAt;
            }
            if (branchModeRef.current && e.data.branchId !== undefined) {
              // Branch mode: accumulate to specific branch
              const branchId = e.data.branchId;
              branchTokenBuffersRef.current[branchId] =
                (branchTokenBuffersRef.current[branchId] || "") + e.data.output;
              const cleanedBranchText = sanitizeThoughtCompletion(
                branchTokenBuffersRef.current[branchId],
                activePartialTextRef.current,
              );

              setBranches(prev => {
                const existing = prev.find(b => b.id === branchId);
                if (existing) {
                  return prev.map(b =>
                    b.id === branchId
                      ? { ...b, text: cleanedBranchText }
                      : b
                  );
                } else {
                  return [...prev, { id: branchId, text: cleanedBranchText, complete: false }];
                }
              });
            } else {
              // Linear mode
              tokenBufferRef.current += e.data.output;
              const cleanedCompletionText = sanitizeThoughtCompletion(
                tokenBufferRef.current,
                activePartialTextRef.current,
              );
              if (cleanedCompletionText) {
                latestCompletionTextRef.current = cleanedCompletionText;
                setCompletionText(cleanedCompletionText);
              }
            }
          }
          break;
        case "branchComplete":
          if (isCompletingRef.current && e.data.branchId !== undefined) {
            const branchId = e.data.branchId;
            const branchText = sanitizeThoughtCompletion(
              branchTokenBuffersRef.current[branchId] ?? "",
              activePartialTextRef.current,
            );

            // Mark branch as complete
            setBranches(prev =>
              prev.map(b =>
                b.id === branchId ? { ...b, text: branchText, complete: true } : b
              )
            );

            // Start Chorus as soon as the first branch is ready. Its feed stays
            // open while the language worker prepares the remaining branches.
            if (chorusModeRef.current && branchText && !completionAbortControllerRef.current?.signal.aborted) {
              const signal = completionAbortControllerRef.current?.signal;
              if (!chorusFeedRef.current) {
                const feed = new ChorusFeed<Branch>();
                chorusFeedRef.current = feed;
                void branchSpeechQueueRef.current.enqueue(() =>
                  speakTextRef.current("Chorus", signal, feed.read(signal))
                );
              }
              chorusFeedRef.current.push({ id: branchId, text: branchText, complete: true });
            }
            if (!chorusModeRef.current && branchText && !completionAbortControllerRef.current?.signal.aborted) {
              const signal = completionAbortControllerRef.current?.signal;
              void branchSpeechQueueRef.current.enqueue(async () => {
                if (signal?.aborted) return;
                setSpeakingBranchId(branchId);
                try {
                  await speakTextRef.current(branchText, signal);
                } finally {
                  setSpeakingBranchId(prev => prev === branchId ? null : prev);
                }
              });
            }
          }
          break;
        case "error":
          // A failed completion must not leave the pipeline waiting for it.
          console.error("Language worker error:", e.data.data);
          chorusFeedRef.current?.finish();
          chorusFeedRef.current = null;
          if (speculationRef.current.ownsLlmRequest(e.data.requestId)) {
            applySpeculationActions(speculationRef.current.onLlmComplete({
              requestId: e.data.requestId!,
              now: performance.now(),
            }));
          } else if (isCompletingRef.current && e.data.requestId === undefined) {
            isCompletingRef.current = false;
            overlapCompletionActiveRef.current = false;
            overlapTriggeredForUtteranceRef.current = false;
          }
          break;
        case "complete":
          if (debugMicRunRef.current && e.data.timings) {
            debugMicRunRef.current.llmTimings = e.data.timings;
          }
          if (speculationRef.current.ownsLlmRequest(e.data.requestId)) {
            applySpeculationActions(speculationRef.current.onLlmComplete({
              requestId: e.data.requestId!,
              now: performance.now(),
            }));
            break;
          }
          if (isCompletingRef.current) {
            if (debugMicRunRef.current?.llmStartedAt !== null && debugMicRunRef.current) {
              debugMicRunRef.current.llmMs =
                performance.now() - debugMicRunRef.current.llmStartedAt;
              debugMicRunRef.current.llmDoneAt = performance.now();
              if (typeof e.data.firstTokenMs === "number") {
                debugMicRunRef.current.llmFirstTokenMs = e.data.firstTokenMs;
              }
            }
            if (branchModeRef.current) {
              rememberTurn(activePartialTextRef.current, Object.values(branchTokenBuffersRef.current)
                .map(text => sanitizeThoughtCompletion(text, activePartialTextRef.current)));
              // Branch mode complete - all branches done
              const hasBranchCompletion = Object.values(branchTokenBuffersRef.current)
                .some((text) => sanitizeThoughtCompletion(
                  text,
                  activePartialTextRef.current,
                ));
              if (!hasBranchCompletion) {
                overlapCompletionActiveRef.current = false;
                overlapTriggeredForUtteranceRef.current = false;
              }
              chorusFeedRef.current?.finish();
              chorusFeedRef.current = null;
              isCompletingRef.current = false;
              branchTokenBuffersRef.current = {};
            } else {
              // Linear mode complete
              const completionResult = sanitizeThoughtCompletion(
                tokenBufferRef.current,
                activePartialTextRef.current,
              ) || latestCompletionTextRef.current;
              rememberTurn(activePartialTextRef.current, [completionResult]);
              setCompletionText(completionResult || null);

              if (completionResult && !completionAbortControllerRef.current?.signal.aborted) {
                const signal = completionAbortControllerRef.current?.signal;
                void speakTextRef.current(completionResult, signal);
              } else {
                overlapCompletionActiveRef.current = false;
                overlapTriggeredForUtteranceRef.current = false;
              }

              isCompletingRef.current = false;
              tokenBufferRef.current = "";
              latestCompletionTextRef.current = "";
            }
          }
          break;
      }
    };

    workerRef.current.addEventListener("message", onMessageReceived);
    return () => {
      workerRef.current?.removeEventListener("message", onMessageReceived);
      // Reset KV cache on unmount to free memory
      workerRef.current?.postMessage({ type: "reset" });
      workerRef.current?.terminate();
      workerRef.current = null;
      chorusFeedRef.current?.finish();
      chorusFeedRef.current = null;
      currentPlayerRef.current?.abort();
      prefetchAbortRef.current?.abort();
      ttsPrefetchRef.current = null;
      coreLoadRequestedRef.current = false;
      // Shutdown TTS worker
      ttsRef.current?.close();
      ttsRef.current = null;
      ttsLoadedRef.current = false;
      preparedVoiceKeyRef.current = null;
      voicePreparePromiseRef.current = null;
      modelRuntimeLeaseRef.current?.release();
      modelRuntimeLeaseRef.current = null;
    };
  }, [applySpeculationActions, rememberTurn]);

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

  const isLoading = llmStatus === "loading" || sttStatus === "loading" || ttsLoading;
  const isReady = llmStatus === "ready" && sttStatus === "ready" && !ttsLoading;
  const isIdle = llmStatus === "idle" || sttStatus === "idle";

  return (
    <>
      <main className={`flex flex-col h-[100svh] overflow-hidden ${thoughtMode === "chorus" && isReady && voiceSetupComplete ? "bg-black" : "bg-zinc-950"} text-zinc-100`}>
        {/* Loading state */}
        {isIdle && !isLoading && (
          <IntroScreen
            onLoad={() => void loadCoreModels()}
            error={modelLoadError}
          />
        )}

        {isLoading && (
          <div className="flex-1 min-h-0">
            <LoadingScreen
              llmItems={progressItems}
              sttItems={sttProgressItems}
              ttsProgress={ttsProgress}
              ttsLoading={ttsLoading}
              message={loadingMessage}
            />
          </div>
        )}

        {/* Voice enrollment: clone the user's own voice before the main view */}
        {isReady && !voiceSetupComplete && (
          <VoiceEnrollment
            onRecorded={applyRecordedVoice}
            onSkip={() => {
              setIsReRecordingVoice(false);
              setVoiceSetupComplete(true);
            }}
            skipLabel={
              isReRecordingVoice
                ? "Keep the current voice"
                : `Use a default voice instead${selectedVoice === "custom" ? "" : ` (${selectedVoice})`}`
            }
          />
        )}

        {/* Main view */}
        {isReady && voiceSetupComplete && (
          <>
            {/* Centered text display */}
            <div className={`relative flex-1 flex items-center justify-center ${thoughtMode === "chorus" ? "" : "px-8"}`}>
              <div className={`text-center ${thoughtMode === "chorus" ? "w-full h-full" : "max-w-2xl"}`}>
                {/* The same words, in the same place, as the Chorus field's subtitle. */}
                {thoughtMode !== "chorus" && !transcribedText && !completionText && (
                  <p className="absolute left-1/2 bottom-[9%] w-[90%] max-w-[60ch] -translate-x-1/2 text-[16px] text-white/55">
                    {micEnabled ? "Start a thought, then pause. Let the voices take it from there." : "Click on the microphone to start"}
                  </p>
                )}

                {transcribedText && !branchMode && (
                  <p className="text-2xl leading-relaxed">
                    <span className="text-white">{transcribedText}</span>
                    {completionText && (
                      <span className="text-zinc-500 italic"> {completionText}</span>
                    )}
                  </p>
                )}

                {/* Chorus: each voice pops out of the brain as a character. */}
                {thoughtMode === "chorus" && (
                  <ChorusStage
                    transcript={transcribedText ?? ""}
                    placeholder={micEnabled ? "Start a thought, then pause. Let the voices take it from there." : "Click on the microphone to start"}
                    branches={branches}
                    speakingIds={chorusSpeakingIds}
                    voiceCount={chorusVoiceCount}
                    spokenIds={chorusSpokenIds}
                    finished={chorusFinished}
                  />
                )}

                {/* Branch mode display */}
                {thoughtMode === "branch" && transcribedText && (
                  <div className="text-left">
                    <p className="text-2xl text-white leading-relaxed mb-4">{transcribedText}</p>
                    {branches.length > 0 && (
                      <div className="flex flex-col gap-1 pl-4">
                        {branches.map((branch, index) => {
                          const curve = branches.length === 1 ? 0 : index === 0 ? -8 : index === branches.length - 1 ? 8 : 0;
                          return (
                            <div
                              key={branch.id}
                              className="flex items-center cursor-pointer group"
                              onClick={() => speakBranch(branch.id)}
                            >
                              <svg width="28" height="20" viewBox="0 0 28 20" className="shrink-0 mr-1">
                                <path
                                  d={`M0,10 Q10,10 14,${10 + curve * 0.5} T28,${10 + curve}`}
                                  fill="none"
                                  stroke={(chorusSpeakingIds.includes(branch.id) || speakingBranchId === branch.id) ? "#71717a" : "#3f3f46"}
                                  strokeWidth="1.5"
                                  strokeLinecap="round"
                                  className="transition-colors group-hover:stroke-zinc-500"
                                />
                              </svg>
                              <span
                                className={`text-2xl italic transition-colors ${
                                  (chorusSpeakingIds.includes(branch.id) || speakingBranchId === branch.id)
                                    ? 'text-zinc-400'
                                    : 'text-zinc-500 group-hover:text-zinc-400'
                                }`}
                              >
                                {branch.text}
                                {!branch.complete && <span className="animate-pulse">...</span>}
                                {(chorusSpeakingIds.includes(branch.id) || speakingBranchId === branch.id) && (
                                  <Volume2 className="inline-block w-4 h-4 ml-2 animate-pulse" />
                                )}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}

                <div className={`mt-4 flex items-center justify-center gap-2 text-zinc-600 text-xs ${isSpeaking && !branchMode ? 'opacity-100' : 'opacity-0'}`}>
                  <Volume2 className="w-3 h-3 animate-pulse" />
                </div>
              </div>
            </div>

            {/* Bottom controls: the microphone, reset beside it, everything else
                behind a small options toggle. */}
            <div className="relative p-6">
              {/* The options float above the bar, so opening them moves nothing. */}
              {optionsOpen && (
                <div ref={optionsPanelRef} className="absolute left-1/2 bottom-full z-20 mb-20 w-[min(30rem,calc(100vw-3rem))] -translate-x-1/2 rounded-2xl border border-white/10 bg-black/90 p-5 text-[15px] backdrop-blur">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="audio/wav,.wav"
                    className="hidden"
                    onChange={(e) => selectCustomVoiceFile(e.target.files?.[0] ?? null)}
                  />
                  <div className="grid gap-4">
                    <label className="flex items-center justify-between gap-4">
                      <span className="text-white/60">Voice</span>
                      <span className="flex items-center gap-2">
                        <select
                          value={selectedVoice}
                          onChange={(e) => selectVoice(e.target.value)}
                          disabled={isSpeaking}
                          className="rounded-md border border-white/15 bg-transparent px-2 py-1 text-white focus:outline-none focus:border-white/50 disabled:opacity-50"
                        >
                          {voices.map((voice) => (
                            <option key={voice} value={voice} className="bg-zinc-900">
                              {voice}
                            </option>
                          ))}
                          {voices.length === 0 && selectedVoice !== "custom" && (
                            <option value={selectedVoice} className="bg-zinc-900">
                              {selectedVoice}
                            </option>
                          )}
                          <option value="custom" className="bg-zinc-900">a recording of mine</option>
                          <option value="record" className="bg-zinc-900">record my voice</option>
                        </select>
                        {selectedVoice === "custom" && (
                          <button
                            onClick={openCustomVoicePicker}
                            disabled={isSpeaking}
                            className="text-white/60 underline underline-offset-4 hover:text-white disabled:opacity-50"
                          >
                            {customVoiceFile ? customVoiceFile.name.slice(0, 14) + (customVoiceFile.name.length > 14 ? "…" : "") : "choose a file"}
                          </button>
                        )}
                      </span>
                    </label>

                    <div className="flex items-center justify-between gap-4">
                      <span className="text-white/60">Answer with</span>
                      <span className="inline-flex rounded-md border border-white/15 p-0.5" role="group" aria-label="How the voices answer">
                        {([["linear", "one voice"], ["chorus", "a chorus"]] as const).map(([mode, name]) => (
                          <button
                            key={mode}
                            onClick={() => selectThoughtMode(mode)}
                            disabled={isSpeaking}
                            aria-pressed={thoughtMode === mode}
                            className={`rounded px-3 py-1 transition-colors disabled:opacity-50 ${
                              thoughtMode === mode ? "bg-white text-black" : "text-white/70 hover:text-white"
                            }`}
                          >
                            {name}
                          </button>
                        ))}
                      </span>
                    </div>

                    {thoughtMode === "chorus" && (
                      <label className="flex items-center justify-between gap-4">
                        <span className="text-white/60">Voices in the chorus</span>
                        <select
                          value={chorusVoiceCount}
                          onChange={(event) => {
                            const count = Number(event.target.value);
                            chorusVoiceCountRef.current = count;
                            setChorusVoiceCount(count);
                          }}
                          className="rounded-md border border-white/15 bg-transparent px-2 py-1 text-white focus:outline-none focus:border-white/50"
                        >
                          {Array.from({ length: 20 }, (_, index) => index + 1).map((count) => (
                            <option key={count} value={count} className="bg-zinc-900">{count}</option>
                          ))}
                        </select>
                      </label>
                    )}

                    <OptionSwitch
                      label="Remember previous messages"
                      detail={contextMode && contextTurns > 0 ? `${contextTurns} ${contextTurns === 1 ? "message" : "messages"} so far` : "up to 8 messages"}
                      on={contextMode}
                      disabled={isSpeaking}
                      onToggle={() => {
                        resetAll(false);
                        contextModeRef.current = !contextModeRef.current;
                        setContextMode(contextModeRef.current);
                      }}
                    />

                  </div>
                </div>
              )}

              {/* The microphone sits on the centre line; reset hangs off its right. */}
              <div className="relative flex items-center justify-center">
                <MicButton
                  listening={micEnabled}
                  speaking={userSpeaking}
                  loading={vadLoading}
                  busy={isSpeaking || chorusPreparing || (thoughtMode === "chorus" && !!transcribedText && !chorusFinished)}
                  onClick={() => void toggleMic()}
                />
                <button
                  onClick={() => resetAll()}
                  aria-label="Reset"
                  title="Reset"
                  className="mic-reset absolute left-1/2 top-1/2 ml-[54px] -translate-y-1/2"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                </button>
              </div>

              {chorusPreparing && (
                <span role="status" className="absolute left-6 bottom-6 text-xs text-zinc-500">Starting Chorus…</span>
              )}

              <button
                ref={optionsToggleRef}
                onClick={() => setOptionsOpen((open) => !open)}
                aria-expanded={optionsOpen}
                aria-label={optionsOpen ? "Hide options" : "Show options"}
                title={optionsOpen ? "Hide options" : "Options"}
                className="absolute right-6 bottom-6 flex items-center gap-1.5 rounded px-2 py-1 text-xs text-zinc-600 transition-colors hover:text-zinc-300"
              >
                {optionsOpen ? <X className="w-3.5 h-3.5" /> : <SlidersHorizontal className="w-3.5 h-3.5" />}
                <span>{optionsOpen ? "close" : "options"}</span>
              </button>
              {vadError && (
                <p role="alert" className="absolute left-1/2 top-0 -translate-x-1/2 -translate-y-full text-center text-xs text-red-400">
                  {vadError}
                </p>
              )}
            </div>
          </>
        )}
      </main>
    </>
  );
}
