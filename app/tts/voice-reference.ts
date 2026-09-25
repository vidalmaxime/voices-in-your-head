export const MAX_VOICE_UPLOAD_SECONDS = 30;
// Mimi's exported encoder has a 1,000-frame transformer cache. Staying below
// its roughly 20-second boundary avoids opaque ORT failures on long prompts.
export const MAX_VOICE_REFERENCE_SECONDS = 15;
export const MIN_VOICE_REFERENCE_SECONDS = 1;

const ANALYSIS_WINDOW_MS = 20;
const TRIM_PADDING_MS = 250;
const MIN_SIGNAL_RMS = 0.001;

/** Prepare mono PCM for Pocket TTS without changing the speaker's natural level. */
export function prepareVoiceReference(
  samples: Float32Array,
  sampleRate: number,
): Float32Array {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new Error("Custom voice has an invalid sample rate.");
  }

  const maxUploadSamples = Math.floor(MAX_VOICE_UPLOAD_SECONDS * sampleRate);
  const limited = samples.subarray(0, maxUploadSamples);
  if (limited.length < MIN_VOICE_REFERENCE_SECONDS * sampleRate) {
    throw new Error("Custom voice must contain at least 1 second of speech.");
  }

  let mean = 0;
  for (let i = 0; i < limited.length; i++) {
    const value = limited[i];
    if (!Number.isFinite(value)) {
      throw new Error("Custom voice contains invalid audio samples.");
    }
    mean += value;
  }
  mean /= limited.length;

  const centered = new Float32Array(limited.length);
  let sumSquares = 0;
  for (let i = 0; i < limited.length; i++) {
    const value = Math.max(-1, Math.min(1, limited[i] - mean));
    centered[i] = value;
    sumSquares += value * value;
  }

  const overallRms = Math.sqrt(sumSquares / centered.length);
  if (overallRms < MIN_SIGNAL_RMS) {
    throw new Error(
      "Custom voice is silent or too quiet. Upload a clean recording with audible speech.",
    );
  }

  const windowSamples = Math.max(
    1,
    Math.round((ANALYSIS_WINDOW_MS / 1000) * sampleRate),
  );
  const activeThreshold = Math.max(MIN_SIGNAL_RMS, overallRms * 0.15);
  let firstActive = 0;
  let lastActive = centered.length;
  let foundActive = false;

  for (let start = 0; start < centered.length; start += windowSamples) {
    const end = Math.min(start + windowSamples, centered.length);
    let windowSquares = 0;
    for (let i = start; i < end; i++) {
      windowSquares += centered[i] * centered[i];
    }
    const windowRms = Math.sqrt(windowSquares / (end - start));
    if (windowRms >= activeThreshold) {
      if (!foundActive) firstActive = start;
      lastActive = end;
      foundActive = true;
    }
  }

  if (!foundActive) {
    throw new Error(
      "Custom voice does not contain clear speech. Use a clean WAV with one speaker.",
    );
  }

  const padding = Math.round((TRIM_PADDING_MS / 1000) * sampleRate);
  const trimStart = Math.max(0, firstActive - padding);
  const trimEnd = Math.min(centered.length, lastActive + padding);
  if (trimEnd - trimStart < MIN_VOICE_REFERENCE_SECONDS * sampleRate) {
    throw new Error("Custom voice must contain at least 1 second of speech.");
  }

  const maxReferenceSamples = Math.floor(
    MAX_VOICE_REFERENCE_SECONDS * sampleRate,
  );
  return centered.slice(
    trimStart,
    Math.min(trimEnd, trimStart + maxReferenceSamples),
  );
}
