import type { AudioPlayer } from "../tts/audio";

/** One synthesis destination at a time, with independently draining playback. */
export function createStaggeredChorus(
  createTrack: (index: number) => AudioPlayer,
  onEnded: (index: number) => void,
): AudioPlayer & { finishTrack(): void } {
  const players: AudioPlayer[] = [];
  const draining: Promise<void>[] = [];
  let active: AudioPlayer | null = null;
  let trackIndex = 0;
  let aborted = false;
  let closed = false;
  let closing: Promise<void> | null = null;

  const finishTrack = () => {
    const index = trackIndex++;
    if (!active) return;
    const player = active;
    active = null;
    // Flush short utterances immediately; do not wait for playback before
    // synthesizing the next voice.
    player.flush();
    draining.push(player.close().finally(() => onEnded(index)));
  };

  return {
    playChunk(samples) {
      if (aborted || closed || !samples.length) return;
      if (!active) {
        active = createTrack(trackIndex);
        players.push(active);
      }
      active.playChunk(samples);
    },
    finishTrack,
    async resume() { await active?.resume(); },
    close() {
      if (!closing) {
        closed = true;
        finishTrack();
        closing = Promise.all(draining).then(() => {});
      }
      return closing;
    },
    abort() {
      aborted = true;
      for (const player of players) player.abort();
    },
    flush() { active?.flush(); },
    get aborted() { return aborted; },
    get underrunCount() { return players.reduce((sum, player) => sum + player.underrunCount, 0); },
    get underrunMs() { return players.reduce((sum, player) => sum + player.underrunMs, 0); },
    get startedAt() { return players[0]?.startedAt ?? null; },
    get context() {
      if (!players[0]) throw new Error("Chorus has not started playback");
      return players[0].context;
    },
    toWav() { throw new Error("Staggered Chorus does not retain recordings"); },
  };
}
