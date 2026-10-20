// Pocket TTS english_18m_refl_2026-10, exported by scripts/pocket-onnx/export.py.
// Pinned to a repo revision so cached files never mix two exports.
export const POCKET_MODEL_REPO = "maxime/pocket-tts-18m-onnx";
export const POCKET_MODEL_REVISION = "9b874ef8e0be4a95a352a0fe0455e064e75404e0";
export const POCKET_MODEL_NAME = "english_18m_refl_2026-10";
export const POCKET_MODEL_BASE = `https://huggingface.co/${POCKET_MODEL_REPO}/resolve/${POCKET_MODEL_REVISION}`;
export const POCKET_MODEL_CACHE = "pocket-tts-models-v2";
/** First-load download: the core sessions, tokenizer and manifest (36 MB) plus one preset voice (4 MB). */
export const POCKET_FIRST_LOAD_BYTES = 40e6;

export function pocketModelUrls(base = POCKET_MODEL_BASE) {
  const root = `${base}/${POCKET_MODEL_NAME}`;
  return {
    mimi_encoder: `${root}/mimi_encoder.onnx`,
    text_conditioner: `${root}/text_conditioner.onnx`,
    flow_lm_main: `${root}/flow_lm_main_int8.onnx`,
    flow_lm_flow: `${root}/flow_lm_flow.onnx`,
    mimi_decoder: `${root}/mimi_decoder.onnx`,
    tokenizer: `${root}/tokenizer.model`,
    bundle: `${root}/bundle.json`,
    bos_before_voice: `${root}/bos_before_voice.npy`,
    voices: `${root}/voices.bin`,
  };
}

export function pocketPrefetchUrls() {
  const { mimi_encoder, voices, ...core } = pocketModelUrls();
  // The encoder and large voice bundle remain lazy; only core assets are needed.
  void mimi_encoder; void voices;
  return Object.values(core);
}
