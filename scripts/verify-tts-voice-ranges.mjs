import { readFile } from "node:fs/promises";

const WORKER_PATH = new URL("../public/tts-onnx/inference-worker.js", import.meta.url);
const BASE_URL =
  "https://huggingface.co/spaces/KevinAHM/pocket-tts-web/resolve/main/onnx/english_2026-04";
const BUNDLE_URL = `${BASE_URL}/bundle.json`;
const VOICES_URL = `${BASE_URL}/voices.bin`;

function parseRanges(workerSource) {
  const tableMatch = workerSource.match(
    /const PREDEFINED_VOICE_RANGES = \{([\s\S]*?)\n\};/,
  );
  if (!tableMatch) {
    throw new Error("Could not find PREDEFINED_VOICE_RANGES in inference worker");
  }

  const ranges = {};
  const entryRegex = /(\w+): \{ start: (\d+), end: (\d+) \}/g;
  let match = entryRegex.exec(tableMatch[1]);
  while (match) {
    ranges[match[1]] = {
      start: Number(match[2]),
      end: Number(match[3]),
    };
    match = entryRegex.exec(tableMatch[1]);
  }

  if (Object.keys(ranges).length === 0) {
    throw new Error("No preset voice ranges were parsed");
  }
  return ranges;
}

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

const workerSource = await readFile(WORKER_PATH, "utf8");
const ranges = parseRanges(workerSource);
const manifest = await fetchJson(BUNDLE_URL);
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

const slice = await fetchVoiceSlice(voiceName, ranges[voiceName]);
const parsed = parsePtvbVoiceSlice(slice);
if (parsed.name !== voiceName) {
  throw new Error(`Range for ${voiceName} parsed as ${parsed.name}`);
}
if (parsed.tensorCount !== 12) {
  throw new Error(`Expected 12 tensors for ${voiceName}, got ${parsed.tensorCount}`);
}
if (!parsed.tensors.every((tensor) => tensor.shape.length > 0 && tensor.byteLength > 0)) {
  throw new Error(`Parsed ${voiceName} contained an invalid tensor`);
}

console.log(
  `Verified ${voiceName}: ${slice.byteLength} bytes, ${parsed.tensorCount} tensors, ${manifestVoices.length} manifest voices`,
);
