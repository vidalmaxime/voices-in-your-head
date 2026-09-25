/** Mix chunked utterances from the same starting sample, preserving every tail. */
export function mixSpeechTracks(tracks: readonly (readonly Float32Array[])[]): Float32Array {
  const lengths = tracks.map(chunks => chunks.reduce((sum, chunk) => sum + chunk.length, 0));
  const activeCount = lengths.filter(length => length > 0).length;
  const mixed = new Float32Array(Math.max(0, ...lengths));
  if (!activeCount) return mixed;

  // Fixed headroom avoids clipping without changing volume as shorter voices end.
  for (const chunks of tracks) {
    let offset = 0;
    for (const chunk of chunks) {
      for (let i = 0; i < chunk.length; i++) {
        mixed[offset + i] += chunk[i] / activeCount;
      }
      offset += chunk.length;
    }
  }
  return mixed;
}
