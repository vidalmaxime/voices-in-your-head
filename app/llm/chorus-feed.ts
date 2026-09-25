/** A single-consumer stream of branches as the language worker finishes them. */
export class ChorusFeed<T> {
  private pending: T[] = [];
  private ended = false;
  private wake: (() => void) | null = null;

  push(value: T) {
    if (this.ended) return;
    this.pending.push(value);
    this.wake?.();
  }

  finish() {
    this.ended = true;
    this.wake?.();
  }

  async *read(signal?: AbortSignal): AsyncGenerator<T> {
    const abort = () => this.finish();
    signal?.addEventListener("abort", abort, { once: true });
    try {
      while (!signal?.aborted) {
        if (this.pending.length) {
          yield this.pending.shift()!;
        } else if (this.ended) {
          return;
        } else {
          await new Promise<void>(resolve => { this.wake = resolve; });
          this.wake = null;
        }
      }
    } finally {
      signal?.removeEventListener("abort", abort);
      this.pending = [];
      this.wake = null;
    }
  }
}
