"use client";

import "./chorus-stage.css";
import { IM_Fell_English } from "next/font/google";
import { assetUrl } from "../base-path";

/** A seventeenth-century book face, its letters slightly uneven: right for
 * voices that are almost, not quite, someone's. */
const fell = IM_Fell_English({
  subsets: ["latin"],
  weight: "400",
  style: ["normal", "italic"],
  display: "swap",
});
/** The voices' typeface, for screens that share the field. */
export const chorusFont = fell;

/** Width / height of each painting, to place a caption at its mouth. */
export const CHARACTER_ASPECTS = [
  199 / 264, 198 / 215, 186 / 230, 223 / 252, 183 / 223, 208 / 248,
  164 / 263, 235 / 227, 209 / 216, 242 / 211, 174 / 247, 190 / 234,
  215 / 243, 240 / 222, 196 / 243, 226 / 269, 202 / 240, 210 / 210,
];
export const CHARACTER_COUNT = CHARACTER_ASPECTS.length;
/** Every character faces right; the open mouth sits here in its painting,
 * and the line starts just past the tip of the nose. */
const MOUTH = { x: 0.8, y: 0.56 };
/** Seconds per word: a little ahead of the synthesized voice, so the whole
 * line is on screen before the voice ends. */
const WORD_SECONDS = 0.2;

type Depth = "near" | "mid" | "far";
type Facing = "right" | "left";
type Slot = { x: number; y: number; depth: Depth; width: number; tilt: number; facing: Facing; reach: number };
type Rect = { x0: number; y0: number; x1: number; y1: number };

/** The stage is a fixed 16:9 box, so heights follow from widths. */
const STAGE_ASPECT = 16 / 9;
/** Figure widths, percent of the stage; the widest stand nearest. */
const WIDTH = { min: 6, max: 14 };
/** Clear space kept between any two figures or lines, percent of the stage. */
const GAP = 1.5;
/** Height of one caption line, percent of the stage, at the near size. */
const LINE_HEIGHT = 3.2;
/** Characters per percent of reach, at the near size: sets how a line wraps. */
const CHARS_PER_PERCENT = 1.6;
/** A typical line, for reserving room before the text is known. */
const TYPICAL_CHARS = 64;

function overlaps(a: Rect, b: Rect) {
  return a.x0 < b.x1 + GAP && a.x1 + GAP > b.x0 && a.y0 < b.y1 + GAP && a.y1 + GAP > b.y0;
}

/**
 * A seeded random source: every draw is a pure function of the seed, so a
 * render is stable for an utterance and different for the next.
 */
function random(seed: number) {
  let state = (seed * 2654435761 + 0x9e3779b9) >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Places `count` voices the way a crowd stands, not a grid: each figure
 * draws a size, stands lower on the stage the larger it is, and takes the
 * first spot, tried at random, where neither it nor the room for its line
 * meets anyone already placed. A figure that finds no room shrinks and tries
 * again. Every recording, with its new seed, is a new arrangement.
 */
function layout(count: number, seed: number): Slot[] {
  const next = random(seed + 7);
  const taken: Rect[] = [];
  const slots: Slot[] = [];
  for (let index = 0; index < Math.max(1, Math.min(count, 20)); index++) {
    let width = WIDTH.min + next() * (WIDTH.max - WIDTH.min);
    let placed: Slot | null = null;
    for (let attempt = 0; attempt < 160 && !placed; attempt++) {
      // Shrink as attempts run out, so a crowd still fits.
      if (attempt > 0 && attempt % 40 === 0) width = Math.max(4, width * 0.8);
      const height = (width / 100) * STAGE_ASPECT / 0.8;
      const depthShare = (width - WIDTH.min) / (WIDTH.max - WIDTH.min);
      // Nearer figures stand lower; a little jitter keeps rows from forming.
      const y = 24 + Math.max(0, Math.min(1, depthShare)) * 56 + (next() - 0.5) * 12;
      const x = width / 2 + 2 + next() * (96 - width);
      const facing: Facing = x > 62 && next() < 0.7 ? "left" : x > 88 ? "left" : "right";
      const scale = 0.6 + depthShare * 0.4;
      const noseX = facing === "right" ? x + 0.3 * width : x - 0.3 * width;
      const room = facing === "right" ? 97 - noseX : noseX - 3;
      const reach = Math.min(10 + next() * 14, room);
      if (reach < 7 || y - height < 3 || y > 84) continue;
      const lines = Math.ceil(TYPICAL_CHARS / (reach * CHARS_PER_PERCENT / scale));
      const mouthY = y - (1 - MOUTH.y) * height * 100;
      const figure: Rect = { x0: x - width / 2, y0: y - height * 100, x1: x + width / 2, y1: y };
      const caption: Rect = facing === "right"
        ? { x0: noseX, y0: mouthY - 1, x1: noseX + reach, y1: mouthY + lines * LINE_HEIGHT * scale }
        : { x0: noseX - reach, y0: mouthY + 2, x1: noseX, y1: mouthY + 3 + lines * LINE_HEIGHT * scale };
      if (taken.some((rect) => overlaps(rect, figure) || overlaps(rect, caption))) continue;
      taken.push(figure, caption);
      const depth: Depth = width >= 11 ? "near" : width >= 8 ? "mid" : "far";
      placed = { x, y, depth, width, tilt: -(1 + next() * 3), facing, reach };
    }
    slots.push(placed ?? { x: 50, y: 60, depth: "mid", width: 6, tilt: -2, facing: "right", reach: 12 });
  }
  return slots;
}

/** Deterministic shuffle so a render is a pure function of the branch ids. */
function shuffled(n: number, seed: number) {
  let state = (seed * 2654435761 + 0x9e3779b9) >>> 0;
  const next = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const items = Array.from({ length: n }, (_, i) => i);
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(next() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
  return items;
}

/**
 * The character and place of the utterance's n-th voice among `count`;
 * `seed` is its first branch id.
 */
export function chorusCast(seed: number, index: number, count: number) {
  const characters = shuffled(CHARACTER_COUNT, seed);
  const slots = layout(count, seed);
  return { character: characters[index % characters.length], ...slots[index % slots.length] };
}

/** Stereo position of a voice, so it sounds from where its character stands (-0.8 to 0.8). */
export function chorusVoicePan(seed: number, index: number, count: number) {
  return ((chorusCast(seed, index, count).x - 50) / 50) * 0.8;
}

export interface ChorusStageProps {
  /** What the speaker said; shown as a subtitle. */
  transcript: string;
  /** Muted instruction shown before the speaker's words arrive. */
  placeholder?: string;
  /** Branches of the current utterance; an empty list starts a new cast. */
  branches: { id: number; text: string }[];
  /** Branch ids whose voice is sounding right now. */
  speakingIds: number[];
  /** How many voices the utterance will have; branches arrive one by one. */
  voiceCount?: number;
  /** Branch ids whose voice has finished; their lines stay up. */
  spokenIds?: number[];
  /** True once every voice has spoken (or the Chorus was cut short): the
   * whole cast sinks back together. */
  finished?: boolean;
}

/**
 * The Chorus as a dark field: the whole cast surfaces out of black as the
 * utterance's branches arrive, each figure brightens while its voice sounds
 * and speaks its line word by word, the lines stay, and once everyone has
 * spoken the cast sinks back together, slowly. The cast is a pure function of
 * the utterance's branch ids, so every voice keeps its face and place, and
 * the next utterance, with new ids, reshuffles.
 */
export default function ChorusStage({ transcript, placeholder, branches, speakingIds, voiceCount = 0, spokenIds = [], finished = false }: ChorusStageProps) {
  const seed = branches[0]?.id ?? 0;
  const count = Math.max(voiceCount, branches.length);
  const live = branches.length > 0 && !finished;
  // A figure surfaces when its voice starts and stays until everyone has spoken.
  const up = (id: number) => live && (speakingIds.includes(id) || spokenIds.includes(id));

  return (
    <div className={`chorus-field ${fell.className}`} aria-label="Chorus">
    <div className="chorus-stage">
      {/* The scene creeps closer for as long as the cast is up. */}
      <div className="chorus-scene" data-live={live}>
      {branches.map(({ id }, index) => {
        const speaking = speakingIds.includes(id);
        const { character, x, y, depth, width, facing } = chorusCast(seed, index, count);
        const src = assetUrl(`/chorus/character-${String(character + 1).padStart(2, "0")}.png`);
        const style = { left: `${x}%`, top: `${y}%`, width: `${width}%`, zIndex: Math.round(y) };
        return (
          <div key={id} className="chorus-slot" data-open={up(id)} data-speaking={speaking} data-depth={depth} data-facing={facing} style={style}>
            <div className="chorus-figure">
              {/* eslint-disable-next-line @next/next/no-img-element -- static asset with alpha, sized by CSS */}
              <img src={src} alt="" draggable={false} />
              {/* eslint-disable-next-line @next/next/no-img-element -- the reflection */}
              <img src={src} alt="" draggable={false} aria-hidden />
            </div>
          </div>
        );
      })}
      <div className="chorus-captions">
        {branches.map(({ id, text }, index) => {
          const open = up(id);
          const { character, x, y, depth, width, tilt, facing, reach } = chorusCast(seed, index, count);
          // The figure's height as a share of the stage height, from its width.
          const height = (width / 100) * STAGE_ASPECT / CHARACTER_ASPECTS[character];
          const mouthX = x + (facing === "right" ? MOUTH.x - 0.5 : 0.5 - MOUTH.x) * width;
          // A line runs from the mouth and wraps within its reach, so its end
          // is always in view. A mirrored figure's line runs left and falls
          // away, so it diverges from the rising line of a figure it faces.
          const style = facing === "right"
            ? { left: `${mouthX}%`, top: `${y - (1 - MOUTH.y) * height * 100}%`, maxWidth: `${reach}%`, transform: `rotate(${tilt}deg)`, transformOrigin: "0 50%" }
            : { right: `${100 - mouthX}%`, top: `${y - (1 - MOUTH.y - 0.1) * height * 100}%`, maxWidth: `${reach}%`, textAlign: "right" as const, transform: `rotate(${2 * tilt}deg)`, transformOrigin: "100% 50%" };
          return (
            <div key={id} className="chorus-caption" data-open={open} data-depth={depth} role={open ? "status" : undefined} style={style}>
              {text.split(/\s+/).filter(Boolean).map((word, i) => (
                <span key={i} style={{ "--word-delay": `${(i * WORD_SECONDS).toFixed(2)}s` } as React.CSSProperties}>
                  {i > 0 ? " " : ""}
                  {word}
                </span>
              ))}
            </div>
          );
        })}
      </div>
      </div>
    </div>
      {(transcript || placeholder) && <p className="chorus-transcript" data-placeholder={!transcript}>{transcript || placeholder}</p>}
    </div>
  );
}
