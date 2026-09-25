export function normalizeTranscript(text: string) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9' ]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function cleanTranscript(text: string) {
  const words = text.trim().split(/\s+/).filter(Boolean);
  const cleaned: string[] = [];

  for (const word of words) {
    const normalized = normalizeTranscript(word);
    const previous = cleaned.at(-1);
    if (normalized && previous && normalizeTranscript(previous) === normalized) continue;
    cleaned.push(word);
  }

  return cleaned.join(" ");
}

export function wordErrorRate(reference: string, hypothesis: string) {
  const referenceWords = normalizeTranscript(reference).split(" ").filter(Boolean);
  const hypothesisWords = normalizeTranscript(hypothesis).split(" ").filter(Boolean);

  if (referenceWords.length === 0) {
    return hypothesisWords.length === 0 ? 0 : 1;
  }

  let previous = Array.from({ length: hypothesisWords.length + 1 }, (_, index) => index);
  for (let referenceIndex = 1; referenceIndex <= referenceWords.length; referenceIndex++) {
    const current = [referenceIndex];
    for (let hypothesisIndex = 1; hypothesisIndex <= hypothesisWords.length; hypothesisIndex++) {
      const substitutionCost =
        referenceWords[referenceIndex - 1] === hypothesisWords[hypothesisIndex - 1] ? 0 : 1;
      current[hypothesisIndex] = Math.min(
        previous[hypothesisIndex] + 1,
        current[hypothesisIndex - 1] + 1,
        previous[hypothesisIndex - 1] + substitutionCost,
      );
    }
    previous = current;
  }

  return previous[hypothesisWords.length] / referenceWords.length;
}
