// Speculative pipeline coordinator.
//
// The serial pipeline waits for the endpoint, then transcribes, then completes,
// then speaks. Because a completion continues what the speaker is already
// saying, most of that work can start while they are still talking:
//
// - Speech recognition re-runs on the growing utterance, and once more at the
//   first pause frame, so the transcript is usually known when the endpoint
//   fires 400 ms later.
// - Completions are generated speculatively and kept as candidates keyed by
//   transcript. In "pause" mode (the default) one completion starts from the
//   pause-pass transcript, so it runs during the endpoint wait and is stopped
//   the moment speech resumes. In "full" mode every transcript revision starts
//   one. At commit, an exact match plays as is; a candidate whose completion
//   begins with the speaker's final words is used with those words removed,
//   the way speculative decoding accepts a verified prefix. Everything else
//   regenerates from the final transcript.
//
// This module is pure: it holds state and returns actions, and the page owns
// the workers, buffers, and UI. Every entry point is synchronous so it can be
// tested without a browser.

// Explicit extensions let node:test load this module without a bundler.
import { sanitizeThoughtCompletion } from "./completion-logic.ts";
import { normalizeTranscript } from "./stt-metrics.ts";
import { VAD_THRESHOLDS } from "./vad-logic.ts";

export const SPECULATION = {
  /** Minimum interval between mid-speech transcription refreshes. */
  sttRefreshMs: 700,
  /** New audio required before a mid-speech refresh is worth a pass. */
  sttMinNewSamples: 16000 * 0.4,
  /** Below this many words a speculative completion is noise. */
  llmMinWords: 3,
  /**
   * A candidate may absorb this many trailing words of the final transcript
   * when its completion predicted them.
   */
  tolerantMaxExtraWords: 4,
  /** Words the completion must still contain after absorbing predicted ones. */
  tolerantMinRemainingWords: 3,
} as const;

export type SttReason = "refresh" | "pause" | "final";

export type SpeculationAction =
  | { type: "stt"; requestId: number; reason: SttReason }
  | { type: "llm-start"; candidateId: number; transcript: string }
  | { type: "llm-interrupt"; candidateId: number }
  | { type: "commit-transcript"; transcript: string }
  | {
      type: "commit-completion";
      candidateId: number;
      text: string;
      complete: boolean;
      match: "exact" | "tolerant" | "final";
    }
  | { type: "completion-update"; candidateId: number; text: string };

export type CandidateStatus = "generating" | "complete" | "interrupted";

export type Candidate = {
  id: number;
  transcript: string;
  key: string;
  tokens: string;
  status: CandidateStatus;
  startedAt: number;
  completedAt: number | null;
  /** Set once the candidate is committed with predicted words removed. */
  absorbedWords: number;
  /** Which transcription pass produced the candidate's transcript. */
  reason: SttReason;
  /** Samples of the utterance the transcript covered when the candidate started. */
  sampleCount: number;
};

export type SpeculationPhase = "idle" | "listening" | "finalizing" | "committed";

/**
 * When completions run before the endpoint:
 * - `false`: never; only the transcript is speculated.
 * - `"pause"`: once per pause, from the pause-pass transcript, so the
 *   completion overlaps the endpoint wait and is interrupted if speech resumes.
 * - `"full"`: for every transcript revision, including mid-speech refreshes.
 * `true` is accepted as `"full"` for compatibility.
 */
export type SpeculateCompletions = false | "pause" | "full";

export type SpeculationOptions = {
  speculateCompletions?: SpeculateCompletions | boolean;
};

export type SpeculationStats = {
  sttRequests: number;
  sttRefreshes: number;
  sttPauseRefreshes: number;
  sttFinalRequests: number;
  /** Time from finalize() until the transcript was known, in ms. */
  transcriptWaitMs: number | null;
  /** True when finalize() found the transcript already computed. */
  transcriptHidden: boolean;
  candidatesStarted: number;
  candidatesInterrupted: number;
  match: "exact" | "tolerant" | "final" | null;
  /** Time from transcript commit until the completion was final, in ms. */
  completionWaitMs: number | null;
  /** How long before the endpoint the committed completion had started, in ms. */
  completionHeadStartMs: number | null;
};

function emptyStats(): SpeculationStats {
  return {
    sttRequests: 0,
    sttRefreshes: 0,
    sttPauseRefreshes: 0,
    sttFinalRequests: 0,
    transcriptWaitMs: null,
    transcriptHidden: false,
    candidatesStarted: 0,
    candidatesInterrupted: 0,
    match: null,
    completionWaitMs: null,
    completionHeadStartMs: null,
  };
}

/** Removes trailing punctuation the way the page does before completing. */
export function prepareTranscriptForCompletion(text: string) {
  return text.trim().replace(/[.!?,;:]+$/, "");
}

function transcriptKey(text: string) {
  return normalizeTranscript(prepareTranscriptForCompletion(text));
}

/**
 * When `finalTranscript` extends `candidateTranscript` by a few words and the
 * candidate's completion begins with exactly those words, returns the
 * completion with the predicted words removed. Returns null otherwise.
 */
export function absorbPredictedWords(input: {
  candidateTranscript: string;
  finalTranscript: string;
  completion: string;
  maxExtraWords?: number;
  minRemainingWords?: number;
}): { text: string; absorbed: number } | null {
  const maxExtra = input.maxExtraWords ?? SPECULATION.tolerantMaxExtraWords;
  const minRemaining = input.minRemainingWords ?? SPECULATION.tolerantMinRemainingWords;

  const candidateWords = transcriptKey(input.candidateTranscript).split(" ").filter(Boolean);
  const finalWords = transcriptKey(input.finalTranscript).split(" ").filter(Boolean);
  if (candidateWords.length === 0 || finalWords.length <= candidateWords.length) return null;
  if (finalWords.length - candidateWords.length > maxExtra) return null;
  for (let i = 0; i < candidateWords.length; i++) {
    if (candidateWords[i] !== finalWords[i]) return null;
  }

  const extras = finalWords.slice(candidateWords.length);
  const rawWords = input.completion.trim().split(/\s+/).filter(Boolean);
  let consumed = 0;
  for (let i = 0; i < extras.length; i++) {
    const raw = rawWords[i];
    if (!raw) return null;
    const normalized = normalizeTranscript(raw);
    // A raw word that normalizes to several tokens cannot be matched cleanly.
    if (normalized !== extras[i]) return null;
    consumed++;
  }

  const remaining = rawWords.slice(consumed);
  if (remaining.length < minRemaining) return null;
  return { text: remaining.join(" "), absorbed: consumed };
}

export class SpeculationCoordinator {
  private phase: SpeculationPhase = "idle";
  private nextRequestId = 1;
  private nextCandidateId = 1;

  private sampleCount = 0;
  private lastVoicedSampleEnd = 0;
  private lastFrameVoiced = false;
  private lastSttSentAt = -Infinity;
  private lastSttSampleCount = 0;
  /** A pause onset arrived while a pass was running; issue one when it returns. */
  private pausePending = false;
  private sttInFlight: { requestId: number; sampleCount: number; reason: SttReason } | null = null;
  private lastSttResult: { transcript: string; sampleCount: number; reason: SttReason } | null = null;
  private awaitingSttRequestId: number | null = null;

  private candidates = new Map<number, Candidate>();
  private activeCandidateId: number | null = null;
  private pendingTranscript: string | null = null;
  private pendingIsFinal = false;
  private pendingReason: SttReason = "final";
  private pendingSampleCount = 0;

  private finalTranscript: string | null = null;
  private committedCandidateId: number | null = null;
  private finalizedAt: number | null = null;
  private transcriptCommittedAt: number | null = null;

  private stats = emptyStats();
  private readonly completionMode: SpeculateCompletions;

  constructor(options: SpeculationOptions = {}) {
    const mode = options.speculateCompletions;
    this.completionMode = mode === true ? "full" : mode === undefined ? false : mode;
  }

  get completions(): SpeculateCompletions {
    return this.completionMode;
  }

  get currentPhase(): SpeculationPhase {
    return this.phase;
  }

  get isSttInFlight(): boolean {
    return this.sttInFlight !== null;
  }

  get statistics(): SpeculationStats {
    return { ...this.stats };
  }

  ownsSttRequest(requestId: number | undefined): boolean {
    if (requestId === undefined) return false;
    return this.sttInFlight?.requestId === requestId || this.awaitingSttRequestId === requestId;
  }

  ownsLlmRequest(requestId: number | undefined): boolean {
    return requestId !== undefined && this.candidates.has(requestId);
  }

  /** Starts a new utterance. Any earlier utterance is discarded. */
  beginUtterance(input: { now: number; initialSamples?: number }): SpeculationAction[] {
    const actions = this.reset();
    this.phase = "listening";
    this.sampleCount = input.initialSamples ?? 0;
    this.lastVoicedSampleEnd = 0;
    this.lastFrameVoiced = false;
    this.lastSttSentAt = -Infinity;
    this.lastSttSampleCount = 0;
    this.pausePending = false;
    return actions;
  }

  /** Discards the utterance and asks the page to stop any running completion. */
  reset(): SpeculationAction[] {
    const actions: SpeculationAction[] = [];
    const active =
      this.activeCandidateId === null ? null : this.candidates.get(this.activeCandidateId) ?? null;
    // The active candidate stays registered until its "complete" message
    // arrives, so the worker is known to be busy and the late message is
    // recognized instead of leaking to the legacy handler.
    this.candidates.clear();
    if (active) {
      if (active.status === "generating") {
        active.status = "interrupted";
        actions.push({ type: "llm-interrupt", candidateId: active.id });
      }
      this.candidates.set(active.id, active);
    } else {
      this.activeCandidateId = null;
    }
    this.phase = "idle";
    this.sampleCount = 0;
    this.lastVoicedSampleEnd = 0;
    this.lastFrameVoiced = false;
    this.sttInFlight = null;
    this.lastSttResult = null;
    this.awaitingSttRequestId = null;
    this.pausePending = false;
    this.pendingTranscript = null;
    this.pendingIsFinal = false;
    this.finalTranscript = null;
    this.committedCandidateId = null;
    this.finalizedAt = null;
    this.transcriptCommittedAt = null;
    this.stats = emptyStats();
    return actions;
  }

  /**
   * One VAD frame. `appended` says whether the page kept the frame in the
   * utterance buffer; `probability` is the VAD speech probability.
   */
  onFrame(input: {
    now: number;
    probability: number;
    samples: number;
    appended: boolean;
  }): SpeculationAction[] {
    if (this.phase !== "listening") return [];
    if (input.appended) this.sampleCount += input.samples;

    const actions: SpeculationAction[] = [];
    const voiced = input.probability >= VAD_THRESHOLDS.pause;
    if (voiced) {
      this.lastVoicedSampleEnd = this.sampleCount;
      // Speech resumed; the pass that matters is the one after the next pause.
      this.pausePending = false;
    }
    // A completion started from a pause transcript assumed the utterance was
    // over. Confirmed speech (the same threshold that resets the endpoint)
    // makes it stale, so stop it now rather than when the next pass returns.
    if (input.probability > VAD_THRESHOLDS.speech) {
      const active =
        this.activeCandidateId === null ? null : this.candidates.get(this.activeCandidateId) ?? null;
      if (active?.status === "generating" && active.reason === "pause") {
        actions.push(...this.interruptActive());
      }
    }
    const pauseOnset = !voiced && this.lastFrameVoiced;
    this.lastFrameVoiced = voiced;
    if (this.lastVoicedSampleEnd === 0) return actions;

    if (pauseOnset) this.pausePending = true;
    if (this.sttInFlight || this.sampleCount <= this.lastSttSampleCount) return actions;

    if (this.pausePending) {
      this.pausePending = false;
      actions.push(this.requestStt("pause", input.now));
      return actions;
    }
    // Mid-speech refreshes exist to feed speculative completions on every
    // revision; pause mode and transcription-only mode need only the pause pass.
    if (
      this.completionMode === "full" &&
      voiced &&
      input.now - this.lastSttSentAt >= SPECULATION.sttRefreshMs &&
      this.sampleCount - this.lastSttSampleCount >= SPECULATION.sttMinNewSamples
    ) {
      actions.push(this.requestStt("refresh", input.now));
    }
    return actions;
  }

  private requestStt(reason: SttReason, now: number): SpeculationAction & { type: "stt" } {
    const requestId = this.nextRequestId++;
    this.sttInFlight = { requestId, sampleCount: this.sampleCount, reason };
    this.lastSttSentAt = now;
    this.lastSttSampleCount = this.sampleCount;
    this.stats.sttRequests++;
    if (reason === "refresh") this.stats.sttRefreshes++;
    if (reason === "pause") this.stats.sttPauseRefreshes++;
    if (reason === "final") this.stats.sttFinalRequests++;
    return { type: "stt", requestId, reason };
  }

  /** The endpoint fired: the utterance is over and needs a completion. */
  finalize(input: { now: number }): SpeculationAction[] {
    if (this.phase !== "listening") return [];
    this.phase = "finalizing";
    this.finalizedAt = input.now;

    const covered = (sampleCount: number) => sampleCount > this.lastVoicedSampleEnd;

    if (this.lastSttResult && covered(this.lastSttResult.sampleCount)) {
      this.stats.transcriptHidden = true;
      return this.commitTranscript(this.lastSttResult.transcript, input.now);
    }
    if (this.sttInFlight && covered(this.sttInFlight.sampleCount)) {
      this.awaitingSttRequestId = this.sttInFlight.requestId;
      return [];
    }
    const action = this.requestStt("final", input.now);
    this.awaitingSttRequestId = action.requestId;
    return [action];
  }

  onSttResult(input: { requestId: number; transcript: string; now: number }): SpeculationAction[] {
    const inFlight = this.sttInFlight;
    if (!inFlight || inFlight.requestId !== input.requestId) return [];
    this.sttInFlight = null;
    this.lastSttResult = {
      transcript: input.transcript,
      sampleCount: inFlight.sampleCount,
      reason: inFlight.reason,
    };

    if (this.phase === "finalizing" && this.awaitingSttRequestId === input.requestId) {
      this.awaitingSttRequestId = null;
      return this.commitTranscript(input.transcript, input.now);
    }
    if (this.phase !== "listening") return [];

    const actions: SpeculationAction[] = [];
    // A pause onset that arrived during this pass gets its own pass now, so the
    // transcript that covers all speech is underway before the endpoint.
    if (this.pausePending && this.sampleCount > this.lastSttSampleCount) {
      this.pausePending = false;
      actions.push(this.requestStt("pause", input.now));
    }
    if (this.completionMode === false) return actions;
    // Pause mode speculates once per pause. A pause transcript that speech has
    // already overtaken is not worth a completion; the next pause pass is.
    if (this.completionMode === "pause") {
      if (inFlight.reason !== "pause") return actions;
      if (this.lastVoicedSampleEnd > inFlight.sampleCount) return actions;
    }

    const transcript = prepareTranscriptForCompletion(input.transcript);
    const key = transcriptKey(transcript);
    if (key.split(" ").filter(Boolean).length < SPECULATION.llmMinWords) return actions;
    if (this.hasCandidateFor(key)) return actions;

    // The pause-onset transcript is the likely final one, so it displaces a
    // stale generation instead of waiting behind it.
    if (this.activeCandidateId !== null) {
      this.pendingTranscript = transcript;
      this.pendingIsFinal = false;
      this.pendingReason = inFlight.reason;
      this.pendingSampleCount = inFlight.sampleCount;
      if (inFlight.reason === "pause") actions.push(...this.interruptActive());
      return actions;
    }
    actions.push(this.startCandidate(transcript, input.now, inFlight.reason, inFlight.sampleCount));
    return actions;
  }

  /** The speech worker dropped or superseded a request without a result. */
  onSttDropped(input: { requestId: number; now: number }): SpeculationAction[] {
    if (this.sttInFlight?.requestId !== input.requestId) return [];
    this.sttInFlight = null;
    if (this.awaitingSttRequestId === input.requestId) {
      const action = this.requestStt("final", input.now);
      this.awaitingSttRequestId = action.requestId;
      return [action];
    }
    return [];
  }

  onLlmToken(input: { requestId: number; text: string }): SpeculationAction[] {
    const candidate = this.candidates.get(input.requestId);
    if (!candidate || candidate.status !== "generating") return [];
    candidate.tokens += input.text;
    if (this.committedCandidateId !== candidate.id) return [];
    const text = this.committedText(candidate);
    return text ? [{ type: "completion-update", candidateId: candidate.id, text }] : [];
  }

  /**
   * The worker's final message. Its decoded output includes the prompt, so the
   * completion is the text accumulated from `onLlmToken`, as in the serial path.
   */
  onLlmComplete(input: { requestId: number; now: number }): SpeculationAction[] {
    const candidate = this.candidates.get(input.requestId);
    if (!candidate) return [];
    const wasInterrupted = candidate.status === "interrupted";
    if (candidate.status === "generating") {
      candidate.status = "complete";
    }
    candidate.completedAt = input.now;
    if (this.activeCandidateId === candidate.id) this.activeCandidateId = null;

    const actions: SpeculationAction[] = [];
    if (this.committedCandidateId === candidate.id && !wasInterrupted) {
      this.stats.completionWaitMs =
        this.transcriptCommittedAt === null ? null : input.now - this.transcriptCommittedAt;
      actions.push({
        type: "commit-completion",
        candidateId: candidate.id,
        text: this.committedText(candidate),
        complete: true,
        match: this.stats.match ?? "final",
      });
    }

    if (this.pendingTranscript !== null && this.activeCandidateId === null) {
      const transcript = this.pendingTranscript;
      const isFinal = this.pendingIsFinal;
      const reason = this.pendingReason;
      const sampleCount = this.pendingSampleCount;
      this.pendingTranscript = null;
      this.pendingIsFinal = false;
      if (isFinal) {
        actions.push(...this.startFinalCandidate(transcript, input.now));
      } else if (
        this.phase === "listening" &&
        !this.hasCandidateFor(transcriptKey(transcript)) &&
        // A queued pause candidate that speech has overtaken is stale too.
        !(reason === "pause" && this.lastVoicedSampleEnd > sampleCount)
      ) {
        actions.push(this.startCandidate(transcript, input.now, reason, sampleCount));
      }
    }

    if (wasInterrupted) this.candidates.delete(candidate.id);
    if (this.phase === "idle" && this.activeCandidateId === null) this.candidates.clear();
    return actions;
  }

  private commitTranscript(rawTranscript: string, now: number): SpeculationAction[] {
    const transcript = prepareTranscriptForCompletion(rawTranscript);
    this.phase = "committed";
    this.finalTranscript = transcript;
    this.transcriptCommittedAt = now;
    this.stats.transcriptWaitMs = this.finalizedAt === null ? null : now - this.finalizedAt;
    const actions: SpeculationAction[] = [{ type: "commit-transcript", transcript }];
    if (!transcript) return actions;

    const key = transcriptKey(transcript);
    const exact = this.findCandidate((c) => c.key === key && c.status !== "interrupted");

    // A finished candidate wins over one still generating, even when the
    // unfinished one is the exact transcript: the point of committing is to
    // start speaking now.
    if (exact && exact.status === "complete") {
      this.committedCandidateId = exact.id;
      this.stats.match = "exact";
      this.stats.completionWaitMs = 0;
      this.stats.completionHeadStartMs = this.headStart(exact, now);
      actions.push({
        type: "commit-completion",
        candidateId: exact.id,
        text: this.committedText(exact),
        complete: true,
        match: "exact",
      });
      return actions;
    }

    for (const candidate of this.candidatesNewestFirst()) {
      if (candidate.status !== "complete") continue;
      const completion = sanitizeThoughtCompletion(candidate.tokens, candidate.transcript);
      const absorbed = absorbPredictedWords({
        candidateTranscript: candidate.transcript,
        finalTranscript: transcript,
        completion,
      });
      if (!absorbed) continue;
      candidate.absorbedWords = absorbed.absorbed;
      this.committedCandidateId = candidate.id;
      this.stats.match = "tolerant";
      this.stats.completionWaitMs = 0;
      this.stats.completionHeadStartMs = this.headStart(candidate, now);
      actions.push({
        type: "commit-completion",
        candidateId: candidate.id,
        text: absorbed.text,
        complete: true,
        match: "tolerant",
      });
      // Whatever the worker is generating now is no longer needed.
      actions.push(...this.interruptActive());
      return actions;
    }

    if (exact) {
      this.committedCandidateId = exact.id;
      this.stats.match = "exact";
      this.stats.completionHeadStartMs = this.headStart(exact, now);
      actions.push({
        type: "commit-completion",
        candidateId: exact.id,
        text: this.committedText(exact),
        complete: false,
        match: "exact",
      });
      return actions;
    }

    this.stats.match = "final";
    if (this.activeCandidateId !== null) {
      this.pendingTranscript = transcript;
      this.pendingIsFinal = true;
      actions.push(...this.interruptActive());
      return actions;
    }
    actions.push(...this.startFinalCandidate(transcript, now));
    return actions;
  }

  private startFinalCandidate(transcript: string, now: number): SpeculationAction[] {
    if (this.phase !== "committed" || this.finalTranscript !== transcript) return [];
    const action = this.startCandidate(transcript, now);
    this.committedCandidateId = action.candidateId;
    return [
      action,
      {
        type: "commit-completion",
        candidateId: action.candidateId,
        text: "",
        complete: false,
        match: "final",
      },
    ];
  }

  private startCandidate(
    transcript: string,
    now: number,
    reason: SttReason = "final",
    sampleCount = this.sampleCount,
  ): SpeculationAction & { type: "llm-start" } {
    const id = this.nextCandidateId++;
    this.candidates.set(id, {
      id,
      transcript,
      key: transcriptKey(transcript),
      tokens: "",
      status: "generating",
      startedAt: now,
      completedAt: null,
      absorbedWords: 0,
      reason,
      sampleCount,
    });
    this.activeCandidateId = id;
    this.stats.candidatesStarted++;
    return { type: "llm-start", candidateId: id, transcript };
  }

  private headStart(candidate: Candidate, now: number): number {
    return (this.finalizedAt ?? now) - candidate.startedAt;
  }

  private interruptActive(): SpeculationAction[] {
    if (this.activeCandidateId === null) return [];
    const active = this.candidates.get(this.activeCandidateId);
    if (!active || active.status !== "generating") return [];
    active.status = "interrupted";
    this.stats.candidatesInterrupted++;
    return [{ type: "llm-interrupt", candidateId: active.id }];
  }

  private committedText(candidate: Candidate): string {
    const cleaned = sanitizeThoughtCompletion(candidate.tokens, candidate.transcript);
    if (candidate.absorbedWords === 0) return cleaned;
    return cleaned.split(/\s+/).filter(Boolean).slice(candidate.absorbedWords).join(" ");
  }

  private hasCandidateFor(key: string): boolean {
    return this.findCandidate((c) => c.key === key && c.status !== "interrupted") !== null;
  }

  private findCandidate(predicate: (candidate: Candidate) => boolean): Candidate | null {
    for (const candidate of this.candidatesNewestFirst()) {
      if (predicate(candidate)) return candidate;
    }
    return null;
  }

  private candidatesNewestFirst(): Candidate[] {
    return [...this.candidates.values()].sort((a, b) => b.id - a.id);
  }
}
