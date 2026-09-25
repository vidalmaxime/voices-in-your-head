import assert from "node:assert/strict";
import test from "node:test";

import {
  appendRollingAudioFrame,
  beginSpeechCapture,
  decideVadFrame,
  shouldRestartVadAfterPlayback,
  shouldSuppressVadInput,
  shouldFinalizeOnVadEnd,
  shouldTriggerOverlapCompletion,
} from "./vad-logic.ts";

test("pre-speech buffer retains copied newest frames", () => {
  const source = Float32Array.from([1]);
  let frames = appendRollingAudioFrame([], source, 2);
  source[0] = 99;
  frames = appendRollingAudioFrame(frames, Float32Array.from([2]), 2);
  frames = appendRollingAudioFrame(frames, Float32Array.from([3]), 2);

  assert.deepEqual(frames.map((frame) => [...frame]), [[2], [3]]);
});

test("pre-speech buffer can be disabled", () => {
  assert.deepEqual(appendRollingAudioFrame([], Float32Array.from([1]), 0), []);
});

test("speech start seeds a new capture from pre-speech frames", () => {
  const preSpeechFrames = [Float32Array.from([1]), Float32Array.from([2])];
  const result = beginSpeechCapture({
    audioFrames: [],
    preSpeechFrames,
    isCapturing: false,
  });

  assert.equal(result.audioFrames, preSpeechFrames);
  assert.deepEqual(result.preSpeechFrames, []);
});

test("late speech-start callback preserves an already restarted capture", () => {
  const restartedFrames = [Float32Array.from([1]), Float32Array.from([2])];
  const result = beginSpeechCapture({
    audioFrames: restartedFrames,
    preSpeechFrames: [],
    isCapturing: true,
  });

  assert.equal(result.audioFrames, restartedFrames);
  assert.deepEqual(result.preSpeechFrames, []);
});

test("suppresses VAD during playback and its echo tail", () => {
  assert.equal(shouldSuppressVadInput({
    ttsPlaybackActive: true,
    now: 2000,
    ignoreUntil: 0,
  }), true);
  assert.equal(shouldSuppressVadInput({
    ttsPlaybackActive: false,
    now: 1999,
    ignoreUntil: 2000,
  }), true);
  assert.equal(shouldSuppressVadInput({
    ttsPlaybackActive: false,
    now: 2000,
    ignoreUntil: 2000,
  }), false);
});

test("restarts VAD only for an enabled mic with no active playback", () => {
  assert.equal(shouldRestartVadAfterPlayback({ micEnabled: true, ttsPlaybackActive: false }), true);
  assert.equal(shouldRestartVadAfterPlayback({ micEnabled: false, ttsPlaybackActive: false }), false);
  assert.equal(shouldRestartVadAfterPlayback({ micEnabled: true, ttsPlaybackActive: true }), false);
});

test("retains medium-confidence phonemes during an utterance", () => {
  const decision = decideVadFrame({
    isSpeech: 0.45,
    now: 800,
    pauseFrameCount: 0,
    speechStartedAt: 0,
    isCapturing: true,
    isCompleting: false,
  });

  assert.equal(decision.appendFrame, true);
  assert.equal(decision.triggerCompletion, false);
});

test("finalizes after a real thought pause", () => {
  const decision = decideVadFrame({
    isSpeech: 0.1,
    now: 1100,
    pauseFrameCount: 4,
    speechStartedAt: 0,
    isCapturing: true,
    isCompleting: false,
  });

  assert.equal(decision.pauseFrameCount, 5);
  assert.equal(decision.triggerCompletion, true);
  assert.equal(decision.appendFrame, false);
});

test("resumed speech interrupts completion and starts a fresh capture", () => {
  const decision = decideVadFrame({
    isSpeech: 0.9,
    now: 1800,
    pauseFrameCount: 6,
    speechStartedAt: 0,
    isCapturing: false,
    isCompleting: true,
  });

  assert.equal(decision.interruptCompletion, true);
  assert.equal(decision.restartCapture, true);
  assert.equal(decision.appendFrame, true);
  assert.equal(decision.pauseFrameCount, 0);
});

test("overlap completion keeps ongoing speech from cancelling it", () => {
  const decision = decideVadFrame({
    isSpeech: 0.9,
    now: 3200,
    pauseFrameCount: 0,
    speechStartedAt: 0,
    isCapturing: false,
    isCompleting: true,
    allowCompletionInterruption: false,
  });

  assert.equal(decision.interruptCompletion, false);
  assert.equal(decision.restartCapture, false);
  assert.equal(decision.appendFrame, false);
});

test("overlap mode triggers once after continuous speech reaches its delay", () => {
  const base = {
    enabled: true,
    delayMs: 3000,
    speechStartedAt: 1000,
    isCapturing: true,
    isCompleting: false,
    alreadyTriggered: false,
  };

  assert.equal(shouldTriggerOverlapCompletion({ ...base, now: 3999 }), false);
  assert.equal(shouldTriggerOverlapCompletion({ ...base, now: 4000 }), true);
  assert.equal(shouldTriggerOverlapCompletion({ ...base, now: 5000, alreadyTriggered: true }), false);
  assert.equal(shouldTriggerOverlapCompletion({ ...base, now: 5000, isCapturing: false }), false);
  assert.equal(shouldTriggerOverlapCompletion({ ...base, now: 5000, isCompleting: true }), false);
});

test("VAD end finalizes buffered speech only when no completion is active", () => {
  assert.equal(shouldFinalizeOnVadEnd({
    bufferedFrameCount: 8,
    isCapturing: true,
    isCompleting: false,
  }), true);
  assert.equal(shouldFinalizeOnVadEnd({
    bufferedFrameCount: 8,
    isCapturing: true,
    isCompleting: true,
  }), false);
});
