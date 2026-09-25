import { readFile } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

const DEBUG_AUDIO_FILES = new Set([
  "1.wav",
  "2.wav",
  "3.wav",
  "max-sample.wav",
  "short-sample.wav",
]);

export async function GET(request: Request) {
  if (process.env.NODE_ENV !== "development") {
    return new NextResponse(null, { status: 404 });
  }

  const name = new URL(request.url).searchParams.get("name") ?? "";
  if (!DEBUG_AUDIO_FILES.has(name)) {
    return NextResponse.json({ error: "Unknown debug audio fixture" }, { status: 404 });
  }

  let audio: Buffer<ArrayBuffer>;
  try {
    audio = await readFile(path.join(process.cwd(), "scripts", name));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      return NextResponse.json(
        { error: `Local audio fixture missing: add scripts/${name}` },
        { status: 404 },
      );
    }
    throw error;
  }
  return new NextResponse(audio, {
    headers: {
      "Cache-Control": "no-store",
      "Content-Type": "audio/wav",
    },
  });
}
