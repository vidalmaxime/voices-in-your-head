// Audio playback utilities for TTS output.

export const SAMPLE_RATE = 24000; // 24kHz sample rate for Mimi codec

/** Parse a WAV file and return Float32Array samples (mono, normalized to [-1, 1]). */
export function parseWav(buffer: ArrayBuffer): { samples: Float32Array; sampleRate: number } {
  const view = new DataView(buffer);

  // Verify RIFF header
  const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (riff !== "RIFF") throw new Error("Not a valid WAV file (missing RIFF header)");

  const wave = String.fromCharCode(view.getUint8(8), view.getUint8(9), view.getUint8(10), view.getUint8(11));
  if (wave !== "WAVE") throw new Error("Not a valid WAV file (missing WAVE format)");

  // Find fmt chunk
  let offset = 12;
  let audioFormat = 0, numChannels = 0, sampleRate = 0, bitsPerSample = 0;

  while (offset < buffer.byteLength) {
    const chunkId = String.fromCharCode(
      view.getUint8(offset), view.getUint8(offset + 1),
      view.getUint8(offset + 2), view.getUint8(offset + 3)
    );
    const chunkSize = view.getUint32(offset + 4, true);

    if (chunkId === "fmt ") {
      audioFormat = view.getUint16(offset + 8, true);
      numChannels = view.getUint16(offset + 10, true);
      sampleRate = view.getUint32(offset + 12, true);
      bitsPerSample = view.getUint16(offset + 22, true);
    } else if (chunkId === "data") {
      const dataOffset = offset + 8;
      const dataSize = chunkSize;

      if (audioFormat !== 1 && audioFormat !== 3) {
        throw new Error(`Unsupported audio format: ${audioFormat} (only PCM supported)`);
      }

      let samples: Float32Array;

      if (audioFormat === 3) {
        // IEEE float
        samples = new Float32Array(buffer, dataOffset, dataSize / 4);
      } else if (bitsPerSample === 16) {
        const int16 = new Int16Array(buffer, dataOffset, dataSize / 2);
        samples = new Float32Array(int16.length);
        for (let i = 0; i < int16.length; i++) {
          samples[i] = int16[i] / 32768;
        }
      } else if (bitsPerSample === 24) {
        const numSamples = dataSize / 3;
        samples = new Float32Array(numSamples);
        for (let i = 0; i < numSamples; i++) {
          const b0 = view.getUint8(dataOffset + i * 3);
          const b1 = view.getUint8(dataOffset + i * 3 + 1);
          const b2 = view.getInt8(dataOffset + i * 3 + 2);
          const value = (b2 << 16) | (b1 << 8) | b0;
          samples[i] = value / 8388608;
        }
      } else if (bitsPerSample === 32) {
        const int32 = new Int32Array(buffer, dataOffset, dataSize / 4);
        samples = new Float32Array(int32.length);
        for (let i = 0; i < int32.length; i++) {
          samples[i] = int32[i] / 2147483648;
        }
      } else {
        throw new Error(`Unsupported bits per sample: ${bitsPerSample}`);
      }

      // Convert to mono by averaging channels
      if (numChannels > 1) {
        const mono = new Float32Array(samples.length / numChannels);
        for (let i = 0; i < mono.length; i++) {
          let sum = 0;
          for (let c = 0; c < numChannels; c++) {
            sum += samples[i * numChannels + c];
          }
          mono[i] = sum / numChannels;
        }
        return { samples: mono, sampleRate };
      }

      return { samples, sampleRate };
    }

    offset += 8 + chunkSize;
    if (chunkSize % 2 !== 0) offset++; // Pad byte
  }

  throw new Error("No data chunk found in WAV file");
}

/** Resample audio to a target sample rate using linear interpolation. */
export function resampleAudio(samples: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate === toRate) return samples;

  const ratio = fromRate / toRate;
  const newLength = Math.floor(samples.length / ratio);
  const result = new Float32Array(newLength);

  for (let i = 0; i < newLength; i++) {
    const srcIndex = i * ratio;
    const srcIndexFloor = Math.floor(srcIndex);
    const srcIndexCeil = Math.min(srcIndexFloor + 1, samples.length - 1);
    const t = srcIndex - srcIndexFloor;
    result[i] = samples[srcIndexFloor] * (1 - t) + samples[srcIndexCeil] * t;
  }

  return result;
}

export interface AudioPlayer {
  /** Play a chunk of PCM samples (Float32Array in range [-1, 1]). */
  playChunk(samples: Float32Array): void;

  /** Resume audio context if suspended (required after user interaction). */
  resume(): Promise<void>;

  /** Wait for all queued audio to finish, then close the audio context. */
  close(): Promise<void>;

  /** Get all played audio as a WAV blob. */
  toWav(): Blob;

  /** Get the underlying AudioContext. */
  readonly context: AudioContext;
}

/**
 * Creates a streaming audio player for playing PCM chunks as they're generated.
 * Each chunk is scheduled to play immediately after the previous one.
 */
/**
 * Converts PCM samples (Float32Array in range [-1, 1]) to a WAV file Blob.
 */
export function samplesToWav(
  samples: Float32Array,
  sampleRate = SAMPLE_RATE,
): Blob {
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const blockAlign = numChannels * (bitsPerSample / 8);
  const dataSize = samples.length * (bitsPerSample / 8);
  const headerSize = 44;

  const buffer = new ArrayBuffer(headerSize + dataSize);
  const view = new DataView(buffer);

  // RIFF header
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, "WAVE");

  // fmt subchunk
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true); // subchunk size
  view.setUint16(20, 1, true); // audio format (PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);

  // data subchunk
  writeString(view, 36, "data");
  view.setUint32(40, dataSize, true);

  // Write PCM samples as 16-bit integers
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    view.setInt16(offset, int16, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

function writeString(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

export function createStreamingPlayer(): AudioPlayer {
  const audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
  let nextStartTime = audioCtx.currentTime;
  let lastEndedPromise: Promise<void> = Promise.resolve();
  const chunks: Float32Array[] = [];

  return {
    playChunk(samples: Float32Array) {
      chunks.push(samples.slice());

      const buffer = audioCtx.createBuffer(1, samples.length, SAMPLE_RATE);
      buffer.getChannelData(0).set(samples);

      const source = audioCtx.createBufferSource();
      source.buffer = buffer;
      source.connect(audioCtx.destination);

      // Schedule this chunk right after the previous one
      const startTime = Math.max(nextStartTime, audioCtx.currentTime);
      source.start(startTime);
      nextStartTime = startTime + buffer.duration;

      // Track when this source finishes playing
      lastEndedPromise = new Promise((resolve) => {
        source.onended = () => resolve();
      });
    },

    async resume() {
      if (audioCtx.state === "suspended") {
        await audioCtx.resume();
      }
    },

    async close() {
      await lastEndedPromise;
      await audioCtx.close();
    },

    toWav() {
      const totalLength = chunks.reduce((sum, c) => sum + c.length, 0);
      const combined = new Float32Array(totalLength);
      let offset = 0;
      for (const chunk of chunks) {
        combined.set(chunk, offset);
        offset += chunk.length;
      }
      return samplesToWav(combined);
    },

    get context() {
      return audioCtx;
    },
  };
}