export const MODEL_RUNTIME_LOCK_NAME = "voices-in-your-head:model-runtime";

type RuntimeLockManager = {
  request(
    name: string,
    options: { mode: "exclusive"; ifAvailable: true },
    callback: (lock: object | null) => Promise<void> | void,
  ): Promise<void>;
};

export type ModelRuntimeLease = {
  release: () => void;
};

function getBrowserLockManager(): RuntimeLockManager | undefined {
  if (typeof navigator === "undefined" || !("locks" in navigator)) return undefined;
  return navigator.locks as unknown as RuntimeLockManager;
}

export async function acquireModelRuntimeLease(
  lockManager: RuntimeLockManager | undefined = getBrowserLockManager(),
): Promise<ModelRuntimeLease | null> {
  if (!lockManager) return null;

  let resolveReady!: (lease: ModelRuntimeLease | null) => void;
  let rejectReady!: (error: unknown) => void;
  let readySettled = false;
  const ready = new Promise<ModelRuntimeLease | null>((resolve, reject) => {
    resolveReady = resolve;
    rejectReady = reject;
  });
  let releaseHold!: () => void;
  const hold = new Promise<void>((resolve) => {
    releaseHold = resolve;
  });

  void lockManager.request(
    MODEL_RUNTIME_LOCK_NAME,
    { mode: "exclusive", ifAvailable: true },
    async (lock) => {
      if (!lock) {
        readySettled = true;
        resolveReady(null);
        return;
      }

      let released = false;
      readySettled = true;
      resolveReady({
        release: () => {
          if (released) return;
          released = true;
          releaseHold();
        },
      });
      await hold;
    },
  ).catch((error) => {
    if (!readySettled) rejectReady(error);
  });

  return ready;
}
