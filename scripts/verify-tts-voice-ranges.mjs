import { pocketModelUrls } from "../app/tts/model-assets.ts";

// Checks that bundle.json's preset voice byte ranges slice voices.bin into
// whole PTVB records holding every flow-LM state tensor.
const { bundle: BUNDLE_URL, voices: VOICES_URL } = pocketModelUrls();

function readUint16(view, cursor) {
  const value = view.getUint16(cursor.offset, true);
  cursor.offset += 2;
  return value;
}

function readUint32(view, cursor) {
  const value = view.getUint32(cursor.offset, true);
  cursor.offset += 4;
  return value;
}

function readString(buffer, cursor, length) {
  const value = new TextDecoder().decode(
    new Uint8Array(buffer, cursor.offset, length),
  );
  cursor.offset += length;
  return value;
}

function parsePtvbVoiceSlice(buffer) {
  const view = new DataView(buffer);
  const cursor = { offset: 0 };
  const name = readString(buffer, cursor, readUint16(view, cursor));
  const tensorCount = readUint16(view, cursor);
  const tensors = [];

  for (let i = 0; i < tensorCount; i++) {
    const path = readString(buffer, cursor, readUint16(view, cursor));
    const dtypeCode = view.getUint8(cursor.offset++);
    const rank = view.getUint8(cursor.offset++);
    const shape = [];
    for (let dim = 0; dim < rank; dim++) {
      shape.push(readUint32(view, cursor));
    }
    const byteLength = readUint32(view, cursor);
    const dataOffset = cursor.offset;
    cursor.offset += byteLength;
    tensors.push({ path, dtypeCode, shape, byteLength, dataOffset });
  }

  if (cursor.offset !== buffer.byteLength) {
    throw new Error(
      `Voice slice ended at ${cursor.offset}, but buffer has ${buffer.byteLength} bytes`,
    );
  }

  return { name, tensorCount, tensors };
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Fetch failed for ${url}: ${response.status}`);
  }
  return response.json();
}

async function fetchVoiceSlice(name, range) {
  const response = await fetch(VOICES_URL, {
    headers: {
      Range: `bytes=${range.start}-${range.end - 1}`,
    },
  });
  if (response.status !== 206) {
    throw new Error(`Expected 206 Partial Content for ${name}, got ${response.status}`);
  }

  const expectedLength = range.end - range.start;
  const contentLength = Number(response.headers.get("content-length"));
  if (contentLength !== expectedLength) {
    throw new Error(
      `Expected ${expectedLength} bytes for ${name}, got Content-Length ${contentLength}`,
    );
  }

  const buffer = await response.arrayBuffer();
  if (buffer.byteLength !== expectedLength) {
    throw new Error(
      `Expected ${expectedLength} bytes for ${name}, got ${buffer.byteLength}`,
    );
  }
  return buffer;
}

const manifest = await fetchJson(BUNDLE_URL);
const ranges = manifest.voices ?? {};
const manifestVoices = manifest.predefined_voices;
if (!Array.isArray(manifestVoices)) {
  throw new Error("bundle.json did not contain predefined_voices");
}

const rangeNames = Object.keys(ranges);
const missing = manifestVoices.filter((voice) => !ranges[voice]);
const extra = rangeNames.filter((voice) => !manifestVoices.includes(voice));
if (missing.length > 0 || extra.length > 0) {
  throw new Error(
    `Voice range mismatch. Missing: ${missing.join(", ") || "none"}; extra: ${
      extra.join(", ") || "none"
    }`,
  );
}

const voiceName = process.argv[2] ?? "jean";
if (!ranges[voiceName]) {
  throw new Error(`Unknown voice '${voiceName}'. Known: ${rangeNames.join(", ")}`);
}

const stateNames = manifest.graphs.flow_lm_main.states.map((spec) => spec.name);
const slice = await fetchVoiceSlice(voiceName, ranges[voiceName]);
const parsed = parsePtvbVoiceSlice(slice);
if (parsed.name !== voiceName) {
  throw new Error(`Range for ${voiceName} parsed as ${parsed.name}`);
}
const tensorNames = parsed.tensors.map((tensor) => tensor.path);
const absent = stateNames.filter((name) => !tensorNames.includes(name));
if (absent.length > 0 || parsed.tensorCount !== stateNames.length) {
  throw new Error(
    `Expected ${stateNames.length} flow state tensors for ${voiceName}, got ${parsed.tensorCount} (missing: ${absent.join(", ") || "none"})`,
  );
}
if (!parsed.tensors.every((tensor) => tensor.shape.length > 0 && tensor.byteLength > 0)) {
  throw new Error(`Parsed ${voiceName} contained an invalid tensor`);
}

console.log(
  `Verified ${voiceName}: ${slice.byteLength} bytes, ${parsed.tensorCount} tensors, ${manifestVoices.length} manifest voices`,
);
