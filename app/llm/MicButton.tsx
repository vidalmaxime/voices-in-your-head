"use client";

import "./mic-button.css";
import { useState } from "react";
import { Loader2, Mic, Square } from "lucide-react";
import { assetUrl } from "../base-path";

export interface MicButtonProps {
  /** The microphone is open and the app is listening. */
  listening: boolean;
  /** The speaker is talking right now. */
  speaking: boolean;
  /** The voice detector is still loading. */
  loading?: boolean;
  /** The voices are being made or are speaking: not a moment to talk. */
  busy?: boolean;
  onClick: () => void;
}

/** The painted microphones in public/chorus, one per hover. */
const MICROPHONE_COUNT = 8;

/**
 * The microphone button: a black disc with a glyph until the cursor arrives,
 * when one of the painted microphones springs up over it, a different one
 * each time, and stays for as long as the app is listening.
 */
export default function MicButton({ listening, speaking, loading = false, busy = false, onClick }: MicButtonProps) {
  const [hovering, setHovering] = useState(false);
  const [face, setFace] = useState(0);
  const mouth = hovering || listening;
  const label = listening ? "Stop listening" : "Start listening";

  return (
      <button
        type="button"
        className="mic"
        data-mouth={mouth}
        data-listening={listening}
        data-speaking={listening && speaking && !busy}
        data-busy={busy}
        data-stop={listening && hovering && !busy}
        disabled={loading}
        aria-label={label}
        aria-busy={busy}
        aria-pressed={listening}
        onClick={onClick}
        onPointerEnter={() => {
          setHovering(true);
          if (!listening) setFace((index) => (index + 1) % MICROPHONE_COUNT);
        }}
        onPointerLeave={() => setHovering(false)}
      >
        {/* Sound rings out from a live microphone. */}
        <span className="mic__ripple" aria-hidden />
        <span className="mic__ripple" aria-hidden />
        <span className="mic__glyph" aria-hidden>
          {loading ? <Loader2 className="w-6 h-6 animate-spin" /> : <Mic className="w-6 h-6" strokeWidth={1.5} />}
        </span>
        {/* eslint-disable-next-line @next/next/no-img-element -- a painting, sized by CSS */}
        <img
          className="mic__mouth"
          src={assetUrl(`/chorus/mic-${String(face + 1).padStart(2, "0")}.png`)}
          alt=""
          draggable={false}
          aria-hidden
        />
        <span className="mic__stop" aria-hidden>
          <Square className="w-5 h-5 fill-current" />
        </span>
        <span className="mic__live" aria-hidden />
      </button>
  );
}
