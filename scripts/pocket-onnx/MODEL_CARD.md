---
license: cc-by-4.0
language:
- en
base_model:
- kyutai/pocket-tts
pipeline_tag: text-to-speech
library_name: onnxruntime
tags:
- tts
- onnx
- onnxruntime-web
- voice-cloning
---

# Pocket TTS `english_18m_refl_2026-10`, ONNX

An ONNX export of [Kyutai's Pocket TTS](https://huggingface.co/kyutai/pocket-tts)
`english_18m_refl_2026-10` (18.2M parameters, reward fine-tuned, voice cloning), for
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) on WebGPU or WASM.

The weights are Kyutai's, released under CC-BY-4.0. This repository only converts them.

## Use policy

Kyutai's terms apply: use of the model must comply with all applicable laws and must not
result in, involve or facilitate any illegal, harmful, deceptive, fraudulent or unauthorized
activity. Prohibited uses include, without limitation, voice impersonation or cloning without
explicit and lawful consent; misinformation, disinformation or deception (including fake news,
fraudulent calls, or presenting generated content as genuine recordings of real people or
events); and the generation of unlawful, harmful, libelous, abusive, harassing,
discriminatory, hateful or privacy-invasive content.

## Files

Everything for one model lives under `english_18m_refl_2026-10/`:

| File | Inputs → outputs |
| --- | --- |
| `mimi_encoder.onnx` | `audio` [1, 1, N] (24 kHz, N a multiple of 1920) → `conditioning` [1, T, 512], including the speaker projection |
| `text_conditioner.onnx` | `token_ids` int64 [1, T] → `embeddings` [1, T, 512] |
| `flow_lm_main.onnx`, `flow_lm_main_int8.onnx` | `sequence` [1, S, 32] (NaN = BOS), `text_embeddings` [1, T, 512], `state_*` → `conditioning` [1, 64], `eos_logit` [1, 1], `out_state_*` |
| `flow_lm_flow[_int8].onnx` | drifting head: `c` [1, 64], `x` [1, 32] noise × √temperature → `latent` [1, 32] |
| `mimi_decoder[_int8].onnx` | `latent` [1, N, 32], `state_*` → `audio` [1, 1, N × 1920], `out_state_*` |
| `tokenizer.model` | SentencePiece tokenizer |
| `bos_before_voice.npy` | [1, 1, 512], prepended to an encoded voice prompt |
| `voices.bin` | PTVB1 bundle of voice-conditioned flow LM states for the preset voices |
| `bundle.json` | Manifest: state names, shapes and dtypes, dimensions, flow type, temperature, EOS settings, voice byte ranges, file hashes |

The flow LM keeps one KV cache per layer, `state_cache_<i>` [2, 1, L, 8, 64], plus a shared
`state_offset`. L is a dynamic dimension: allocate the voice prompt, the text and the frames
you intend to generate, zero-filled. The Mimi decoder keeps a 50-step sliding window per
transformer layer and its convolution buffers, all listed in `bundle.json`.

Sampling follows upstream: temperature 0.3, EOS when `eos_logit > -1.0`, ignored for the first
6 frames, then 3 more frames (5 for prompts of 4 words or fewer).

## How it was made

`scripts/pocket-onnx/export.py` in
[voices-in-your-head](https://github.com/vidalmaxime/voices-in-your-head) traces export-friendly
re-implementations of the streaming modules (explicit state tensors instead of in-place
caches) with the upstream weights and encodes the preset voices' source recordings through
the exported graphs. `validate.py` checks every graph against PyTorch with onnxruntime: the
fp32 graphs agree to about 1e-5, and the preset voice states match Kyutai's published ones.

`flow_lm_main_int8.onnx` quantizes the transformer's attention and FFN weights (per channel),
like upstream's `quantize=True`; over a teacher-forced run it stays within upstream's own
int8 error. The sampler head and Mimi are only provided in fp32, as upstream never quantizes
them.
