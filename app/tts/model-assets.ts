export const POCKET_MODEL_BASE = "https://huggingface.co/spaces/KevinAHM/pocket-tts-web/resolve/main";
export const POCKET_MODEL_CACHE = "pocket-tts-models-v1";

export function pocketModelUrls(base = POCKET_MODEL_BASE) {
  const root = `${base}/onnx/english_2026-04`;
  return {
    mimi_encoder: `${root}/mimi_encoder_int8.onnx`,
    text_conditioner: `${root}/text_conditioner_int8.onnx`,
    flow_lm_main: `${root}/flow_lm_main_int8.onnx`,
    flow_lm_flow: `${root}/flow_lm_flow_int8.onnx`,
    mimi_decoder: `${root}/mimi_decoder_int8.onnx`,
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
