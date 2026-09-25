/** Serial inference with cancellation for work that has not started yet. */
export class CompletionQueue {
  private tail: Promise<void> = Promise.resolve();
  private generation = 0;
  private pending = 0;

  interrupt() {
    this.generation++;
  }

  enqueue(task: () => Promise<void>, onCancelled: () => void = () => {}): Promise<void> {
    const generation = this.generation;
    this.pending++;
    const run = this.tail.then(async () => {
      try {
        if (generation !== this.generation) {
          onCancelled();
          return;
        }
        await task();
      } finally {
        this.pending--;
      }
    });
    this.tail = run.catch(() => {});
    return run;
  }

  /** Warmup must never accumulate behind real inference or another warmup. */
  enqueueWarmup(task: () => Promise<void>): Promise<void> {
    if (this.pending > 0) return Promise.resolve();
    return this.enqueue(task);
  }
}
