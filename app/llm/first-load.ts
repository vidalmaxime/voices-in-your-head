import { POCKET_FIRST_LOAD_BYTES } from "../tts/model-assets";

/**
 * First-visit download for the default model stack: the notebooks 350M
 * completion model, Parakeet CTC 0.6B Q4, and Pocket TTS with one preset
 * voice. The intro quotes it and the loading screen counts against it.
 */
export const LLM_FIRST_LOAD_BYTES = 294e6;
export const STT_FIRST_LOAD_BYTES = 644e6;
export const TTS_FIRST_LOAD_BYTES = POCKET_FIRST_LOAD_BYTES;
export const FIRST_LOAD_BYTES = LLM_FIRST_LOAD_BYTES + STT_FIRST_LOAD_BYTES + TTS_FIRST_LOAD_BYTES;
