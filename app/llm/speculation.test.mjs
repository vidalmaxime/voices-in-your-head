import assert from "node:assert/strict";
import test from "node:test";

import {
  SPECULATION,
  SpeculationCoordinator,
  absorbPredictedWords,
} from "./speculation.ts";

const FRAME = 1536; // 96 ms at 16 kHz

function speak(coordinator, { now, frames, probability = 0.9 }) {
  const actions = [];
  for (let i = 0; i < frames; i++) {
    now += 96;
    actions.push(...coordinator.onFrame({ now, probability, samples: FRAME, appended: true }));
  }
  return { now, actions };
}

function types(actions) {
  return actions.map((action) => action.type);
}

test("mid-speech refreshes wait for enough new audio and respect the interval", () => {
  const coordinator = new SpeculationCoordinator({ speculateCompletions: true });
  coordinator.beginUtterance({ now: 0 });

  // Four frames = 384 ms of audio: below the 400 ms minimum.
  let step = speak(coordinator, { now: 0, frames: 4 });
  assert.deepEqual(step.actions, []);

  // The fifth frame crosses the minimum and issues the first refresh.
  step = speak(coordinator, { now: step.now, frames: 1 });
  assert.deepEqual(types(step.actions), ["stt"]);
  assert.equal(step.actions[0].reason, "refresh");
  const first = step.actions[0].requestId;

  // Nothing else is issued while that request is in flight.
  step = speak(coordinator, { now: step.now, frames: 2 });
  assert.deepEqual(step.actions, []);

  coordinator.onSttResult({ requestId: first, transcript: "I think the reason", now: step.now });

  // Enough new audio arrives before the interval since the last send elapses.
  step = speak(coordinator, { now: step.now, frames: 3 });
  assert.deepEqual(step.actions, []);
  assert.equal(SPECULATION.sttRefreshMs, 700);
  // Sent at 480 ms; frames at 1056 and 1152 ms are still inside the interval,
  // the one at 1248 ms is not.
  step = speak(coordinator, { now: step.now, frames: 3 });
  assert.deepEqual(types(step.actions), ["stt"]);
  assert.equal(coordinator.statistics.sttRefreshes, 2);
});

test("a pause onset issues a transcription pass immediately", () => {
  const coordinator = new SpeculationCoordinator({ speculateCompletions: true });
  coordinator.beginUtterance({ now: 0 });
  let step = speak(coordinator, { now: 0, frames: 5 });
  const first = step.actions.find((action) => action.type === "stt");
  coordinator.onSttResult({ requestId: first.requestId, transcript: "hello there", now: step.now });

  // Well inside the refresh interval, but the first quiet frame still fires.
  step = speak(coordinator, { now: step.now, frames: 1, probability: 0.1 });
  assert.deepEqual(types(step.actions), ["stt"]);
  assert.equal(step.actions[0].reason, "pause");
  assert.equal(coordinator.statistics.sttPauseRefreshes, 1);
});

test("finalize reuses a transcript that already covers all speech", () => {
  const coordinator = new SpeculationCoordinator({ speculateCompletions: true });
  coordinator.beginUtterance({ now: 0 });
  let step = speak(coordinator, { now: 0, frames: 6 });
  const refresh = step.actions.find((action) => action.type === "stt");
  coordinator.onSttResult({ requestId: refresh.requestId, transcript: "the strange", now: step.now });

  step = speak(coordinator, { now: step.now, frames: 1, probability: 0.1 });
  const pause = step.actions[0];
  assert.equal(pause.reason, "pause");
  // The pause transcript starts a speculative completion right away.
  const started = coordinator.onSttResult({
    requestId: pause.requestId,
    transcript: "the strange thing about hearing my voice is.",
    now: step.now + 300,
  });
  assert.deepEqual(types(started), ["llm-start"]);

  // Endpoint fires 400 ms after the pause frame; more quiet frames arrived.
  step = speak(coordinator, { now: step.now + 300, frames: 3, probability: 0.1 });
  const actions = coordinator.finalize({ now: step.now });
  assert.deepEqual(types(actions), ["commit-transcript", "commit-completion"]);
  assert.equal(actions[0].transcript, "the strange thing about hearing my voice is");
  assert.equal(actions[1].candidateId, started[0].candidateId);
  assert.equal(actions[1].complete, false);
  assert.equal(actions[1].match, "exact");
  assert.equal(coordinator.statistics.transcriptHidden, true);
  assert.equal(coordinator.statistics.transcriptWaitMs, 0);
});

test("finalize waits for an in-flight pass that covers all speech", () => {
  const coordinator = new SpeculationCoordinator({ speculateCompletions: true });
  coordinator.beginUtterance({ now: 0 });
  let step = speak(coordinator, { now: 0, frames: 6 });
  const refresh = step.actions.find((action) => action.type === "stt");
  coordinator.onSttResult({ requestId: refresh.requestId, transcript: "the strange", now: step.now });
  step = speak(coordinator, { now: step.now, frames: 1, probability: 0.1 });
  const pause = step.actions[0];

  step = speak(coordinator, { now: step.now, frames: 4, probability: 0.1 });
  assert.deepEqual(coordinator.finalize({ now: step.now }), []);
  assert.equal(coordinator.currentPhase, "finalizing");

  const actions = coordinator.onSttResult({
    requestId: pause.requestId,
    transcript: "the strange thing is",
    now: step.now + 60,
  });
  assert.equal(actions[0].type, "commit-transcript");
  assert.equal(coordinator.statistics.transcriptHidden, false);
  assert.equal(coordinator.statistics.transcriptWaitMs, 60);
});

test("finalize requests a final pass when nothing covers the tail", () => {
  const coordinator = new SpeculationCoordinator({ speculateCompletions: true });
  coordinator.beginUtterance({ now: 0 });
  let step = speak(coordinator, { now: 0, frames: 6 });
  const refresh = step.actions.find((action) => action.type === "stt");
  coordinator.onSttResult({ requestId: refresh.requestId, transcript: "the", now: step.now });
  // More speech arrives after that result, and the endpoint fires without a pause pass.
  step = speak(coordinator, { now: step.now, frames: 3 });
  const actions = coordinator.finalize({ now: step.now });
  assert.deepEqual(types(actions), ["stt"]);
  assert.equal(actions[0].reason, "final");
  assert.equal(coordinator.ownsSttRequest(actions[0].requestId), true);

  const commit = coordinator.onSttResult({
    requestId: actions[0].requestId,
    transcript: "the whole thing",
    now: step.now + 400,
  });
  assert.equal(commit[0].type, "commit-transcript");
  assert.equal(coordinator.statistics.transcriptWaitMs, 400);
});

test("an exact speculative candidate is committed without regenerating", () => {
  const coordinator = new SpeculationCoordinator({ speculateCompletions: true });
  coordinator.beginUtterance({ now: 0 });
  let step = speak(coordinator, { now: 0, frames: 6 });
  const refresh = step.actions.find((action) => action.type === "stt");

  const started = coordinator.onSttResult({
    requestId: refresh.requestId,
    transcript: "the strange thing about talking to myself is",
    now: step.now,
  });
  assert.deepEqual(types(started), ["llm-start"]);
  const candidateId = started[0].candidateId;
  assert.deepEqual(coordinator.onLlmToken({ requestId: candidateId, text: "how often" }), []);
  coordinator.onLlmToken({ requestId: candidateId, text: " the answer arrives before the question" });
  coordinator.onLlmComplete({ requestId: candidateId, now: step.now + 500 });

  step = speak(coordinator, { now: step.now + 500, frames: 1, probability: 0.1 });
  const pause = step.actions[0];
  coordinator.onSttResult({
    requestId: pause.requestId,
    transcript: "The strange thing about talking to myself is,",
    now: step.now + 200,
  });

  const actions = coordinator.finalize({ now: step.now + 400 });
  assert.deepEqual(types(actions), ["commit-transcript", "commit-completion"]);
  assert.equal(actions[1].match, "exact");
  assert.equal(actions[1].complete, true);
  assert.equal(actions[1].text, "how often the answer arrives before the question");
  assert.equal(coordinator.statistics.completionWaitMs, 0);
  assert.equal(coordinator.statistics.candidatesStarted, 1);
});

test("a candidate that predicted the speaker's last words is used without them", () => {
  const coordinator = new SpeculationCoordinator({ speculateCompletions: true });
  coordinator.beginUtterance({ now: 0 });
  let step = speak(coordinator, { now: 0, frames: 6 });
  const refresh = step.actions.find((action) => action.type === "stt");
  const started = coordinator.onSttResult({
    requestId: refresh.requestId,
    transcript: "I think the reason this keeps",
    now: step.now,
  });
  const candidateId = started[0].candidateId;
  coordinator.onLlmToken({ requestId: candidateId, text: "happening is that nobody" });
  coordinator.onLlmToken({ requestId: candidateId, text: " writes anything down" });
  coordinator.onLlmComplete({ requestId: candidateId, now: step.now + 600 });

  step = speak(coordinator, { now: step.now + 600, frames: 1, probability: 0.1 });
  const pause = step.actions[0];
  coordinator.onSttResult({
    requestId: pause.requestId,
    transcript: "I think the reason this keeps happening is",
    now: step.now + 300,
  });
  const actions = coordinator.finalize({ now: step.now + 400 });
  const commit = actions.find((action) => action.type === "commit-completion");
  assert.equal(commit.match, "tolerant");
  assert.equal(commit.complete, true);
  assert.equal(commit.text, "that nobody writes anything down");
  // The pause transcript had started an exact candidate; a finished tolerant
  // one is preferred and the unfinished exact one is stopped.
  assert.deepEqual(types(actions), ["commit-transcript", "commit-completion", "llm-interrupt"]);
});

test("a dropped final pass is reissued", () => {
  const coordinator = new SpeculationCoordinator({ speculateCompletions: true });
  coordinator.beginUtterance({ now: 0 });
  const step = speak(coordinator, { now: 0, frames: 6 });
  const refresh = step.actions.find((action) => action.type === "stt");
  coordinator.onSttResult({ requestId: refresh.requestId, transcript: "the", now: step.now });
  const later = speak(coordinator, { now: step.now, frames: 3 });
  const [finalRequest] = coordinator.finalize({ now: later.now });
  const reissued = coordinator.onSttDropped({ requestId: finalRequest.requestId, now: later.now + 10 });
  assert.deepEqual(types(reissued), ["stt"]);
  assert.equal(reissued[0].reason, "final");
  assert.notEqual(reissued[0].requestId, finalRequest.requestId);
  assert.equal(coordinator.ownsSttRequest(reissued[0].requestId), true);
});

test("absorbPredictedWords rejects weak matches", () => {
  assert.equal(
    absorbPredictedWords({
      candidateTranscript: "I think the reason",
      finalTranscript: "I think the reason this",
      completion: "that nobody writes",
    }),
    null,
    "predicted words must match",
  );
  assert.equal(
    absorbPredictedWords({
      candidateTranscript: "I think the reason",
      finalTranscript: "I think the reason this keeps",
      completion: "this keeps going",
    }),
    null,
    "too little completion left after absorbing",
  );
  assert.equal(
    absorbPredictedWords({
      candidateTranscript: "I think",
      finalTranscript: "I think the reason this keeps happening today",
      completion: "the reason this keeps happening today is clear enough",
    }),
    null,
    "too many absorbed words",
  );
  assert.deepEqual(
    absorbPredictedWords({
      candidateTranscript: "I think the reason",
      finalTranscript: "I think the reason this keeps",
      completion: "this keeps going wrong every time",
    }),
    { text: "going wrong every time", absorbed: 2 },
  );
});

test("a stale speculative generation is interrupted and the final one streams", () => {
  const coordinator = new SpeculationCoordinator({ speculateCompletions: true });
  coordinator.beginUtterance({ now: 0 });
  let step = speak(coordinator, { now: 0, frames: 6 });
  const refresh = step.actions.find((action) => action.type === "stt");
  const started = coordinator.onSttResult({
    requestId: refresh.requestId,
    transcript: "when I finally have a",
    now: step.now,
  });
  const staleId = started[0].candidateId;

  step = speak(coordinator, { now: step.now, frames: 1, probability: 0.1 });
  const pause = step.actions[0];
  // The pause transcript displaces the stale generation.
  const displaced = coordinator.onSttResult({
    requestId: pause.requestId,
    transcript: "when I finally have a free afternoon I want to",
    now: step.now + 300,
  });
  assert.deepEqual(displaced, [{ type: "llm-interrupt", candidateId: staleId }]);
  assert.equal(coordinator.statistics.candidatesInterrupted, 1);

  const finalized = coordinator.finalize({ now: step.now + 400 });
  assert.deepEqual(types(finalized), ["commit-transcript"]);

  // The worker acknowledges the interrupt with a partial completion.
  const resumed = coordinator.onLlmComplete({ requestId: staleId, now: step.now + 450 });
  assert.deepEqual(types(resumed), ["llm-start", "commit-completion"]);
  const finalId = resumed[0].candidateId;
  assert.equal(resumed[0].transcript, "when I finally have a free afternoon I want to");
  assert.equal(resumed[1].complete, false);

  const update = coordinator.onLlmToken({ requestId: finalId, text: "take the train somewhere" });
  assert.deepEqual(update, [
    { type: "completion-update", candidateId: finalId, text: "take the train somewhere" },
  ]);
  coordinator.onLlmToken({ requestId: finalId, text: " I have never been" });
  const done = coordinator.onLlmComplete({ requestId: finalId, now: step.now + 1500 });
  assert.equal(done[0].type, "commit-completion");
  assert.equal(done[0].complete, true);
  assert.equal(done[0].text, "take the train somewhere I have never been");
  assert.equal(coordinator.statistics.match, "final");
  assert.equal(coordinator.statistics.completionWaitMs, 1100);
});

test("reset interrupts the active candidate and the next utterance waits for the worker", () => {
  const coordinator = new SpeculationCoordinator({ speculateCompletions: true });
  coordinator.beginUtterance({ now: 0 });
  let step = speak(coordinator, { now: 0, frames: 6 });
  const refresh = step.actions.find((action) => action.type === "stt");
  const [started] = coordinator.onSttResult({
    requestId: refresh.requestId,
    transcript: "maybe the part that",
    now: step.now,
  });

  const reset = coordinator.beginUtterance({ now: 5000 });
  assert.deepEqual(reset, [{ type: "llm-interrupt", candidateId: started.candidateId }]);
  assert.equal(coordinator.ownsLlmRequest(started.candidateId), true);

  step = speak(coordinator, { now: 5000, frames: 6 });
  const next = step.actions.find((action) => action.type === "stt");
  const queued = coordinator.onSttResult({
    requestId: next.requestId,
    transcript: "the second thought is",
    now: step.now,
  });
  assert.deepEqual(queued, [], "the worker is still busy with the interrupted run");

  const late = coordinator.onLlmComplete({ requestId: started.candidateId, now: step.now + 50 });
  assert.deepEqual(types(late), ["llm-start"]);
  assert.equal(late[0].transcript, "the second thought is");
  assert.equal(coordinator.ownsLlmRequest(started.candidateId), false);
});

test("transcription-only mode issues pause passes but no refreshes or completions", () => {
  const off = new SpeculationCoordinator({ speculateCompletions: false });
  off.beginUtterance({ now: 0 });
  let step = speak(off, { now: 0, frames: 30 });
  assert.deepEqual(step.actions, [], "no mid-speech refresh without completion speculation");
  step = speak(off, { now: step.now, frames: 1, probability: 0.1 });
  assert.deepEqual(types(step.actions), ["stt"]);
  assert.equal(step.actions[0].reason, "pause");
  assert.deepEqual(
    off.onSttResult({ requestId: step.actions[0].requestId, transcript: "a long enough sentence", now: step.now }),
    [],
  );

  const on = new SpeculationCoordinator({ speculateCompletions: true });
  on.beginUtterance({ now: 0 });
  step = speak(on, { now: 0, frames: 6 });
  const refresh = step.actions.find((action) => action.type === "stt");
  assert.deepEqual(on.onSttResult({ requestId: refresh.requestId, transcript: "so um", now: step.now }), []);
});

test("a pause onset during a running pass gets its own pass when that one returns", () => {
  const coordinator = new SpeculationCoordinator({ speculateCompletions: true });
  coordinator.beginUtterance({ now: 0 });
  let step = speak(coordinator, { now: 0, frames: 6 });
  const refresh = step.actions.find((action) => action.type === "stt");

  // The speaker stops while the refresh is still running.
  step = speak(coordinator, { now: step.now, frames: 2, probability: 0.1 });
  assert.deepEqual(step.actions, [], "nothing is issued while a pass is in flight");

  const actions = coordinator.onSttResult({
    requestId: refresh.requestId,
    transcript: "the strange thing about",
    now: step.now,
  });
  assert.equal(actions[0].type, "stt");
  assert.equal(actions[0].reason, "pause");
  assert.equal(actions[1].type, "llm-start");

  // That pass covers the whole utterance, so the endpoint can wait for it.
  step = speak(coordinator, { now: step.now, frames: 2, probability: 0.1 });
  assert.deepEqual(coordinator.finalize({ now: step.now }), []);
  const commit = coordinator.onSttResult({
    requestId: actions[0].requestId,
    transcript: "the strange thing about all of this",
    now: step.now + 50,
  });
  assert.equal(commit[0].type, "commit-transcript");
  assert.equal(coordinator.statistics.sttPauseRefreshes, 1);
});

test("speech resuming cancels a pending pause pass", () => {
  const coordinator = new SpeculationCoordinator({ speculateCompletions: true });
  coordinator.beginUtterance({ now: 0 });
  let step = speak(coordinator, { now: 0, frames: 6 });
  const refresh = step.actions.find((action) => action.type === "stt");
  step = speak(coordinator, { now: step.now, frames: 1, probability: 0.1 });
  step = speak(coordinator, { now: step.now, frames: 2 });
  const actions = coordinator.onSttResult({
    requestId: refresh.requestId,
    transcript: "the strange thing about",
    now: step.now,
  });
  assert.deepEqual(types(actions), ["llm-start"]);
});

test("pause mode starts one completion from the pause transcript and streams it at the endpoint", () => {
  const coordinator = new SpeculationCoordinator({ speculateCompletions: "pause" });
  assert.equal(coordinator.completions, "pause");
  coordinator.beginUtterance({ now: 0 });
  let step = speak(coordinator, { now: 0, frames: 30 });
  assert.deepEqual(step.actions, [], "no mid-speech refreshes in pause mode");

  step = speak(coordinator, { now: step.now, frames: 1, probability: 0.1 });
  assert.deepEqual(types(step.actions), ["stt"]);
  const pause = step.actions[0];
  assert.equal(pause.reason, "pause");

  // The pause pass returns 200 ms later, well before the 400 ms endpoint.
  const started = coordinator.onSttResult({
    requestId: pause.requestId,
    transcript: "the strange thing about hearing my voice is.",
    now: step.now + 200,
  });
  assert.deepEqual(types(started), ["llm-start"]);
  const candidateId = started[0].candidateId;
  coordinator.onLlmToken({ requestId: candidateId, text: "how often" });

  step = speak(coordinator, { now: step.now + 200, frames: 3, probability: 0.1 });
  const actions = coordinator.finalize({ now: step.now });
  assert.deepEqual(types(actions), ["commit-transcript", "commit-completion"]);
  assert.equal(actions[1].candidateId, candidateId);
  assert.equal(actions[1].match, "exact");
  assert.equal(actions[1].complete, false);
  assert.equal(actions[1].text, "how often");
  assert.equal(coordinator.statistics.transcriptHidden, true);
  assert.equal(coordinator.statistics.completionHeadStartMs, 288);

  coordinator.onLlmToken({ requestId: candidateId, text: " it answers first" });
  const done = coordinator.onLlmComplete({ requestId: candidateId, now: step.now + 500 });
  assert.equal(done[0].type, "commit-completion");
  assert.equal(done[0].complete, true);
  assert.equal(done[0].text, "how often it answers first");
  assert.equal(coordinator.statistics.completionWaitMs, 500);
});

test("pause mode stops a speculative completion as soon as speech resumes", () => {
  const coordinator = new SpeculationCoordinator({ speculateCompletions: "pause" });
  coordinator.beginUtterance({ now: 0 });
  let step = speak(coordinator, { now: 0, frames: 10 });
  step = speak(coordinator, { now: step.now, frames: 1, probability: 0.1 });
  const pause = step.actions[0];
  const [started] = coordinator.onSttResult({
    requestId: pause.requestId,
    transcript: "I think the reason",
    now: step.now + 150,
  });

  // A medium-confidence frame does not count as resumed speech.
  step = speak(coordinator, { now: step.now + 150, frames: 1, probability: 0.45 });
  assert.deepEqual(step.actions, []);
  // Confirmed speech does.
  step = speak(coordinator, { now: step.now, frames: 1, probability: 0.9 });
  assert.deepEqual(step.actions, [{ type: "llm-interrupt", candidateId: started.candidateId }]);
  assert.equal(coordinator.statistics.candidatesInterrupted, 1);

  // The next pause issues a new pass; its transcript waits for the worker to
  // acknowledge the interrupt, then starts a fresh candidate.
  step = speak(coordinator, { now: step.now, frames: 4 });
  step = speak(coordinator, { now: step.now, frames: 1, probability: 0.1 });
  const next = step.actions.find((action) => action.type === "stt");
  assert.equal(next.reason, "pause");
  const queued = coordinator.onSttResult({
    requestId: next.requestId,
    transcript: "I think the reason this keeps happening is",
    now: step.now + 200,
  });
  assert.deepEqual(queued, []);
  const resumed = coordinator.onLlmComplete({ requestId: started.candidateId, now: step.now + 220 });
  assert.deepEqual(types(resumed), ["llm-start"]);
  assert.equal(resumed[0].transcript, "I think the reason this keeps happening is");
  assert.equal(coordinator.statistics.candidatesStarted, 2);
});

test("pause mode does not complete a pause transcript that speech has overtaken", () => {
  const coordinator = new SpeculationCoordinator({ speculateCompletions: "pause" });
  coordinator.beginUtterance({ now: 0 });
  let step = speak(coordinator, { now: 0, frames: 10 });
  step = speak(coordinator, { now: step.now, frames: 1, probability: 0.1 });
  const pause = step.actions[0];
  // The speaker continues before the pass returns.
  step = speak(coordinator, { now: step.now, frames: 2 });
  const actions = coordinator.onSttResult({
    requestId: pause.requestId,
    transcript: "I think the reason",
    now: step.now,
  });
  assert.deepEqual(actions, [], "a stale pause transcript starts nothing");
  assert.equal(coordinator.statistics.candidatesStarted, 0);

  // The endpoint then needs a final pass since nothing covers the tail.
  step = speak(coordinator, { now: step.now, frames: 1, probability: 0.1 });
  assert.equal(step.actions[0]?.reason, "pause");
});

test("full mode keeps refresh candidates running through speech", () => {
  const coordinator = new SpeculationCoordinator({ speculateCompletions: "full" });
  assert.equal(coordinator.completions, "full");
  coordinator.beginUtterance({ now: 0 });
  let step = speak(coordinator, { now: 0, frames: 6 });
  const refresh = step.actions.find((action) => action.type === "stt");
  const [started] = coordinator.onSttResult({
    requestId: refresh.requestId,
    transcript: "when I finally have a",
    now: step.now,
  });
  step = speak(coordinator, { now: step.now, frames: 3, probability: 0.9 });
  assert.deepEqual(step.actions, [], "a refresh candidate is not stale while speech continues");
  assert.equal(coordinator.ownsLlmRequest(started.candidateId), true);
  assert.equal(coordinator.statistics.candidatesInterrupted, 0);
});
