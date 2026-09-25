// Captures a short microphone clip as WAV so it can be used as a Pocket TTS
// voice reference, without asking the user to produce a file themselves.

import { requestMicrophoneStream } from "../llm/microphone";
import { samplesToWav } from "./audio";
import { MAX_VOICE_REFERENCE_SECONDS } from "./voice-reference";

/**
 * Enrollment prompt. Covers every English consonant class (including the rare
 * /Z/, /T/, /D/, /tS/, /dZ/), all common vowels and diphthongs, and reads in
 * roughly 10 seconds at a natural pace.
 */
export const VOICE_ENROLLMENT_SENTENCE =
  "The quiet zebra watched a huge yellow ship glide through the harbor, " +
  "while children chased pigeons and Joshua measured fresh bread. " +
  "Think of five brave youngsters.";

/** Hard stop, matching the reference window the encoder actually keeps. */
export const MAX_ENROLLMENT_SECONDS = MAX_VOICE_REFERENCE_SECONDS;
/** Below this the clip is too short to clone from reliably. */
export const MIN_ENROLLMENT_SECONDS = 4;

const RECORDED_VOICE_FILENAME = "my-voice.wav";
const CHUNK_SAMPLES = 2048;

// Kept inline (like PCMPlayerWorklet) so the module URL is a blob and needs no
// public asset of its own.
const RECORDER_WORKLET = `
class VoiceRecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(${CHUNK_SAMPLES});
    this.offset = 0;
    this.port.onmessage = () => {
      this.flush();
      this.port.postMessage({ flushed: true });
    };
  }

  flush() {
    if (this.offset === 0) return;
    const chunk = this.buffer.slice(0, this.offset);
    this.offset = 0;
    this.port.postMessage(chunk, [chunk.buffer]);
  }

  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input) return true;
    for (let i = 0; i < input.length; i++) {
      this.buffer[this.offset++] = input[i];
      if (this.offset === this.buffer.length) this.flush();
    }
    return true;
  }
}

registerProcessor('voice-recorder', VoiceRecorderProcessor);
`;

export type VoiceRecording = {
  file: File;
  durationSeconds: number;
};

function getAudioContextConstructor() {
  return (
    window.AudioContext ||
    (window as typeof window & { webkitAudioContext?: typeof AudioContext })
      .webkitAudioContext
  );
}

/** Records raw mono PCM from the microphone and packages it as a WAV file. */
export class VoiceRecorder {
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private node: AudioWorkletNode | null = null;
  private chunks: Float32Array[] = [];
  private sampleCount = 0;
  private flushResolve: (() => void) | null = null;
  private onLevel?: (level: number) => void;

  get isRecording(): boolean {
    return this.node !== null;
  }

  async start(options: { onLevel?: (level: number) => void } = {}): Promise<void> {
    if (this.node) return;
    this.onLevel = options.onLevel;
    this.chunks = [];
    this.sampleCount = 0;

    const AudioContextClass = getAudioContextConstructor();
    if (!AudioContextClass) throw new Error("Web Audio is not supported");

    // Browser voice processing (AGC, denoise) reshapes timbre, which is exactly
    // what the clone needs to keep, so the capture stays raw.
    this.stream = await requestMicrophoneStream(
      () =>
        navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: false,
            autoGainControl: false,
            noiseSuppression: false,
          },
        }),
      // Enrollment is where first-time users meet the permission prompt.
      30000,
    );

    try {
      const context = new AudioContextClass();
      this.context = context;
      const blob = new Blob([RECORDER_WORKLET], { type: "application/javascript" });
      const workletUrl = URL.createObjectURL(blob);
      try {
        await context.audioWorklet.addModule(workletUrl);
      } finally {
        URL.revokeObjectURL(workletUrl);
      }

      const node = new AudioWorkletNode(context, "voice-recorder");
      node.port.onmessage = (event) => this.handleChunk(event.data);
      this.source = context.createMediaStreamSource(this.stream);
      // Muted sink: the graph only runs while the node reaches a destination.
      const sink = context.createGain();
      sink.gain.value = 0;
      this.source.connect(node);
      node.connect(sink);
      sink.connect(context.destination);
      this.node = node;

      await context.resume();
    } catch (error) {
      await this.teardown();
      throw error;
    }
  }

  private handleChunk(data: Float32Array | { flushed: true }) {
    if (!(data instanceof Float32Array)) {
      this.flushResolve?.();
      this.flushResolve = null;
      return;
    }

    this.chunks.push(data);
    this.sampleCount += data.length;

    if (!this.onLevel) return;
    let squares = 0;
    for (let i = 0; i < data.length; i++) squares += data[i] * data[i];
    const rms = Math.sqrt(squares / data.length);
    this.onLevel(Math.min(1, Math.sqrt(rms) * 2));
  }

  /** Stops capture and returns the recorded clip as a WAV file. */
  async stop(): Promise<VoiceRecording> {
    const context = this.context;
    const node = this.node;
    if (!context || !node) throw new Error("Recorder is not running");

    await new Promise<void>((resolve) => {
      this.flushResolve = resolve;
      node.port.postMessage("flush");
      window.setTimeout(resolve, 100);
    });
    this.flushResolve = null;

    const sampleRate = context.sampleRate;
    const samples = new Float32Array(this.sampleCount);
    let offset = 0;
    for (const chunk of this.chunks) {
      samples.set(chunk, offset);
      offset += chunk.length;
    }
    this.chunks = [];
    this.sampleCount = 0;
    await this.teardown();

    const file = new File([samplesToWav(samples, sampleRate)], RECORDED_VOICE_FILENAME, {
      type: "audio/wav",
    });
    return { file, durationSeconds: samples.length / sampleRate };
  }

  /** Stops capture and discards whatever was recorded. */
  async cancel(): Promise<void> {
    this.chunks = [];
    this.sampleCount = 0;
    await this.teardown();
  }

  private async teardown(): Promise<void> {
    this.flushResolve = null;
    this.onLevel = undefined;
    if (this.node) {
      this.node.port.onmessage = null;
      this.node.disconnect();
      this.node = null;
    }
    this.source?.disconnect();
    this.source = null;
    for (const track of this.stream?.getTracks() ?? []) track.stop();
    this.stream = null;
    const context = this.context;
    this.context = null;
    await context?.close().catch(() => {});
  }
}
