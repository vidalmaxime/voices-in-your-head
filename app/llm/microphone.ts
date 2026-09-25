type StoppableMediaStream = {
  getTracks(): Array<{ stop(): void }>;
};

export async function requestMicrophoneStream<T extends StoppableMediaStream>(
  request: () => Promise<T>,
  timeoutMs = 15000,
) {
  let timedOut = false;
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  const streamPromise = request().then((stream) => {
    if (timedOut) {
      for (const track of stream.getTracks()) track.stop();
    }
    return stream;
  });

  const timeoutPromise = new Promise<never>((_, reject) => {
    timeoutId = setTimeout(() => {
      timedOut = true;
      reject(new Error("Microphone permission timed out. Check the browser permission and retry."));
    }, timeoutMs);
  });

  try {
    return await Promise.race([streamPromise, timeoutPromise]);
  } finally {
    if (timeoutId !== null) clearTimeout(timeoutId);
  }
}
