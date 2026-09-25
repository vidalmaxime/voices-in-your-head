export type VadFrameDecisionInput = {
  isSpeech: number;
  now: number;
  pauseFrameCount: number;
  speechStartedAt: number | null;
  isCapturing: boolean;
  isCompleting: boolean;
  allowCompletionInterruption?: boolean;
};

export type VadFrameDecision = {
  appendFrame: boolean;
  interruptCompletion: boolean;
  pauseFrameCount: number;
  restartCapture: boolean;
  triggerCompletion: boolean;
  userSpeaking: boolean | null;
};

export const VAD_THRESHOLDS = {
  speech: 0.6,
  pause: 0.3,
  pauseDurationMs: 400,
  minimumSpeechMs: 500,
  frameDurationMs: 96,
} as const;

export function appendRollingAudioFrame(
  frames: Float32Array[],
  frame: Float32Array,
  maxFrames: number,
) {
  if (maxFrames <= 0) return [];
  return [...frames, frame.slice()].slice(-maxFrames);
}

export function beginSpeechCapture(input: {
  audioFrames: Float32Array[];
  preSpeechFrames: Float32Array[];
  isCapturing: boolean;
}) {
  return {
    audioFrames: input.isCapturing ? input.audioFrames : input.preSpeechFrames,
    preSpeechFrames: [] as Float32Array[],
  };
}

export function shouldSuppressVadInput(input: {
  ttsPlaybackActive: boolean;
  now: number;
  ignoreUntil: number;
}) {
  return input.ttsPlaybackActive || input.now < input.ignoreUntil;
}

export function shouldRestartVadAfterPlayback(input: {
  micEnabled: boolean;
  ttsPlaybackActive: boolean;
}) {
  return input.micEnabled && !input.ttsPlaybackActive;
}

export function decideVadFrame(input: VadFrameDecisionInput): VadFrameDecision {
  const isSpeech = input.isSpeech > VAD_THRESHOLDS.speech;
  const isPause = input.isSpeech < VAD_THRESHOLDS.pause;
  const restartCapture =
    isSpeech && input.isCompleting && (input.allowCompletionInterruption ?? true);
  const isCapturing = input.isCapturing || restartCapture;

  let pauseFrameCount = input.pauseFrameCount;
  let userSpeaking: boolean | null = null;
  if (isSpeech) {
    pauseFrameCount = 0;
    userSpeaking = true;
  } else if (isPause) {
    pauseFrameCount++;
  }

  const pauseDuration = pauseFrameCount * VAD_THRESHOLDS.frameDurationMs;
  const speechDuration = input.speechStartedAt === null ? 0 : input.now - input.speechStartedAt;
  const triggerCompletion =
    isPause &&
    !input.isCompleting &&
    pauseDuration >= VAD_THRESHOLDS.pauseDurationMs &&
    speechDuration >= VAD_THRESHOLDS.minimumSpeechMs;

  return {
    appendFrame: isCapturing && !triggerCompletion,
    interruptCompletion: restartCapture,
    pauseFrameCount,
    restartCapture,
    triggerCompletion,
    userSpeaking,
  };
}

export function shouldTriggerOverlapCompletion(input: {
  enabled: boolean;
  delayMs: number;
  now: number;
  speechStartedAt: number | null;
  isCapturing: boolean;
  isCompleting: boolean;
  alreadyTriggered: boolean;
}) {
  return (
    input.enabled &&
    input.isCapturing &&
    !input.isCompleting &&
    !input.alreadyTriggered &&
    input.speechStartedAt !== null &&
    input.now - input.speechStartedAt >= Math.max(0, input.delayMs)
  );
}

export function shouldFinalizeOnVadEnd(input: {
  bufferedFrameCount: number;
  isCapturing: boolean;
  isCompleting: boolean;
}) {
  return input.isCapturing && !input.isCompleting && input.bufferedFrameCount > 0;
}
