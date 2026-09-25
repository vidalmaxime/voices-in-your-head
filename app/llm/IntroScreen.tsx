import "./intro.css";
import Image from "next/image";
import { useState, type CSSProperties } from "react";
import { assetUrl } from "../base-path";

/**
 * First-visit download, in MB, for the default model stack: the notebooks
 * 350M completion model (294), Parakeet CTC 0.6B Q4 (644), the Pocket TTS
 * core sessions (126), and one preset voice (about 7). The browser caches it.
 */
const FIRST_LOAD_MB = 1070;

// Face centres and soft mask radii, as percentages of the welcome artwork.
// Visit every face in a scattered order rather than sweeping along the rows.
const ECHO_FACES = [
  [24, 24, 10, 15], [47, 19, 11, 14], [70, 23, 10, 14],
  [88, 33, 10, 14], [16, 44, 10, 13], [35, 44, 10, 14],
  [54, 43, 9, 14], [67, 50, 9, 13], [85, 56, 10, 12],
  [19, 69, 10, 14], [41, 72, 10, 14], [58, 70, 8, 12],
  [74, 77, 9, 15], [88, 80, 8, 13],
] as const;

interface IntroScreenProps {
  onLoad: () => void;
  error: string | null;
}

export default function IntroScreen({ onLoad, error }: IntroScreenProps) {
  const [showDownloadInfo, setShowDownloadInfo] = useState(false);

  return (
    <section className="voices-intro" aria-label="Welcome to Voices in your head">
      <div className="intro-composition">
        <h1 className="intro-title">
          <Image
            src={assetUrl("/voices-title.png")}
            alt="Voices in your head"
            width={2172}
            height={724}
            priority
            sizes="(max-width: 767px) 90vw, 70vw"
          />
        </h1>
        <div className="intro-artwork">
          <Image
            src={assetUrl("/voices-intro-chorus-dark.png")}
            alt="Colorful painted faces with open mouths and overlapping echoes of their voices"
            width={1536}
            height={1024}
            priority
            sizes="calc(100vw - 48px)"
            className="intro-image"
          />
          {/* Softly masked copies let a single painted face leave an impression. */}
          {ECHO_FACES.map(([x, y, width, height], index) => (
            <div
              key={index}
              className="intro-echo"
              aria-hidden="true"
              style={{
                "--echo-x": `${x}%`,
                "--echo-y": `${y}%`,
                "--echo-width": `${width}%`,
                "--echo-height": `${height}%`,
                animationDelay: `${0.6 + ((index * 5) % ECHO_FACES.length) * 4}s`,
              } as CSSProperties}
            >
              <Image
                src={assetUrl("/voices-intro-chorus-dark.png")}
                alt=""
                width={1536}
                height={1024}
                sizes="calc(100vw - 48px)"
                className="intro-image"
                draggable={false}
              />
            </div>
          ))}
        </div>
      </div>
      <div className="intro-controls">
        <p className="max-w-lg text-center text-base leading-relaxed text-zinc-300">
          Start a thought. Hear what the voices have to say.
        </p>
        <div className="intro-action">
          <button
            onClick={onLoad}
            className="intro-begin"
          >
            Begin
          </button>
          <div
            className="intro-download"
            onMouseEnter={() => setShowDownloadInfo(true)}
            onMouseLeave={() => setShowDownloadInfo(false)}
          >
            <button
              type="button"
              className="intro-download__trigger"
              aria-label={`${(FIRST_LOAD_MB / 1000).toFixed(2)} GB download information`}
              aria-describedby={showDownloadInfo ? "intro-download-info" : undefined}
              onFocus={() => setShowDownloadInfo(true)}
              onBlur={() => setShowDownloadInfo(false)}
              onClick={() => setShowDownloadInfo(true)}
              onKeyDown={(event) => {
                if (event.key === "Escape") setShowDownloadInfo(false);
              }}
            >
              ({(FIRST_LOAD_MB / 1000).toFixed(2)} GB) ⓘ
            </button>
            {showDownloadInfo && (
              <span id="intro-download-info" role="tooltip" className="intro-download__tooltip">
                Everything runs in your browser. The initial download is cached for future visits.
              </span>
            )}
          </div>
        </div>
        {error && <p role="alert" className="max-w-sm text-center text-xs text-red-400">{error}</p>}
      </div>
    </section>
  );
}
