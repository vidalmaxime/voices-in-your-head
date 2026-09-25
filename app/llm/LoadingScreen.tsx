"use client";

import "./chorus-stage.css";
import "./loading-screen.css";
import { assetUrl } from "../base-path";
import { CHARACTER_ASPECTS, CHARACTER_COUNT, chorusFont } from "./ChorusStage";
import { loadingLayout } from "./loading-layout";

export interface LoadProgressItem {
  file: string;
  progress: number;
  total: number;
}

export interface LoadingScreenProps {
  /** Files of the completion model, with bytes so far and in total. */
  llmItems: LoadProgressItem[];
  /** Files of the speech recogniser. */
  sttItems: LoadProgressItem[];
  /** Speech synthesis download, 0 to 1. */
  ttsProgress: number;
  ttsLoading: boolean;
  /** The pipeline's own status line; shown when it reports a problem. */
  message: string;
}

/** What a first load fetches, so the measure reads right before every file has announced itself. */
const EXPECTED_BYTES = 1_058e6;
const TTS_BYTES = 126e6;
/** The crowd's arrangement is the same every load: a place people come to know. */
const LOADING_CAST = loadingLayout(CHARACTER_ASPECTS);

function sum(items: LoadProgressItem[], key: "progress" | "total") {
  return items.reduce((total, item) => total + item[key], 0);
}

/**
 * The loading screen: the same black field as the Chorus, the cast's figures
 * surfacing one by one as the models arrive, a loading label,
 * and the count in megabytes.
 */
export default function LoadingScreen({ llmItems, sttItems, ttsProgress, message }: LoadingScreenProps) {
  const known = sum(llmItems, "total") + sum(sttItems, "total") + TTS_BYTES;
  const total = Math.max(EXPECTED_BYTES, known);
  const loaded = sum(llmItems, "progress") + sum(sttItems, "progress") + ttsProgress * TTS_BYTES;
  const fraction = Math.max(0, Math.min(1, loaded / total));
  const failed = /fail|error|cancel/i.test(message);
  const phase = failed ? message : "Loading...";
  // Figures surface as the load advances; the latest to arrive is in colour.
  const surfaced = Math.min(CHARACTER_COUNT, Math.floor(fraction * CHARACTER_COUNT + 0.5));
  const megabytes = (bytes: number) => Math.round(bytes / 1e6).toLocaleString("en-US");

  return (
    <div className={`chorus-field ${chorusFont.className}`} role="status" aria-live="polite" aria-label="Loading the voices">
      <div className="chorus-stage loading-stage">
        {LOADING_CAST.map(({ character, x, y, depth, width, facing }, index) => {
          const src = assetUrl(`/chorus/character-${String(character + 1).padStart(2, "0")}.png`);
          const style = { left: `${x}%`, top: `${y}%`, width: `${width}%`, zIndex: Math.round(y) };
          return (
            <div
              key={index}
              className="chorus-slot"
              data-open={index < surfaced}
              data-speaking={index === surfaced - 1}
              data-depth={depth}
              data-facing={facing}
              style={style}
            >
              <div className="chorus-figure">
                {/* eslint-disable-next-line @next/next/no-img-element -- static asset with alpha, sized by CSS */}
                <img src={src} alt="" draggable={false} />
                {/* eslint-disable-next-line @next/next/no-img-element -- the reflection */}
                <img src={src} alt="" draggable={false} aria-hidden />
              </div>
            </div>
          );
        })}
      </div>
      <div className="loading-words">
        {/* Turns the whole time, so a long warmup with no bytes moving still reads as alive. */}
        {!failed && <span className="loading-spinner" aria-hidden />}
        <p className="loading-words__phase">{phase}</p>
        {!failed && (
          <p className="loading-words__count">
            {megabytes(loaded)} of {megabytes(total)} MB
          </p>
        )}
      </div>
      <div className="loading-line" style={{ width: `${fraction * 100}%` }} />
    </div>
  );
}
