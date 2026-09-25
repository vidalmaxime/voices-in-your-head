export type SpeechTask = () => Promise<void>;

export class SpeechQueue {
  private tail: Promise<void> = Promise.resolve();

  enqueue(task: SpeechTask): Promise<void> {
    const run = this.tail.catch(() => {}).then(task);
    this.tail = run.catch(() => {});
    return run;
  }
}
