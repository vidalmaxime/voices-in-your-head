import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    "node_modules/**",
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    "public/**",
    "scripts/.venv/**",
    "app/tts/EventEmitter.js",
    "app/tts/PCMPlayerWorklet.js",
    "app/tts/sentencepiece-browser.js",
    "app/tts/sentencepiece.js",
    "app/tts/inference.ts",
    "app/tts/pocket-tts.ts",
  ]),
]);

export default eslintConfig;
