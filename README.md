# Voices in Your Head

Start a thought, then pause. Let the voices continue your thought.

Speech recognition, text generation, and voice synthesis run locally in the browser. Models download once and are cached for future visits.

## Run

Requires a WebGPU-capable browser, microphone access, and a recent Node.js version.

```sh
npm install
npm run dev
```

Open [localhost:3000/voices](http://localhost:3000/voices). The initial model download is about **1.07 GB**; cloning may load additional assets.

Production: `npm run build` then `npm run start`.

## How it works

```text
Microphone → Silero VAD → Parakeet → LFM2.5 → Pocket TTS → Web Audio
```

| Stage | Model / runtime | Code |
| --- | --- | --- |
| Speech recognition | Parakeet CTC 0.6B, Q4, WebGPU with WASM fallback | `app/llm/whisper-worker.ts` |
| Thought completion | LFM2.5-350M fine-tuned on Simone Weil and Albert Camus, Q4, WebGPU | `app/llm/worker.ts` |
| Voice synthesis | Kyutai Pocket TTS, INT8 ONNX, WASM | `public/tts-onnx/inference-worker.js` |
| Playback | Streamed 24 kHz audio with stereo positioning | `app/tts/audio.ts`, `app/llm/staggered-chorus.ts` |

Next.js and React provide the interface; Transformers.js loads recognition and completion models in separate workers. `app/llm/LLMPage.tsx` coordinates capture, loading, generation, and playback.

- **Pauses:** Silero detects speech and triggers completion after roughly 480 ms of quiet. Listening is suppressed during playback to avoid hearing the app itself.
- **Completions:** the model continues your sentence, with a 32-token cap and cleanup of repetition and chatbot-style output.
- **Voices:** Pocket encodes a recording of up to 15 seconds and reuses its voice conditioning. Synthesis runs sequentially, while chorus tracks can overlap in playback.
- **Memory:** optional conversation context retains up to eight turns in session memory; Reset clears it.
- **Loading:** model sessions initialize sequentially with overlapping asset prefetch. Browser caches avoid repeat downloads, and a Web Lock prevents duplicate model stacks in same-origin tabs.

The public model is [maxime/personal-notebooks-350m](https://huggingface.co/maxime/personal-notebooks-350m), trained on English translations of Weil's and Camus's notebooks. See its model repository for license terms. Source texts and training data are not published here.

## Fine-tuning

Both notebook and personal-notes workflows use **LiquidAI/LFM2.5-350M** with rank-8 LoRA. Training runs locally on Apple Silicon with MLX-LM and Transformers, outside the browser. Download the base model first:

```sh
hf download LiquidAI/LFM2.5-350M config.json generation_config.json tokenizer.json tokenizer_config.json chat_template.jinja model.safetensors --local-dir data/models/lfm2.5-350m-base
```

For personal Markdown notes:

```sh
python scripts/finetune/prepare_dataset.py --notes /path/to/notes
mlx_lm.lora -c scripts/finetune/lora-config.yaml
```

Preparation uses the runtime prompt and Liquid tokenizer, preserves complete sentence tails, and enforces output/context limits. Use `--exclude` patterns to omit copied material. Outputs go to ignored `data/personal-lfm/`. The personal LFM model needs a new training run before use.

For the public notebook workflow, supply the source EPUBs under `data/`, run `scripts/finetune/prepare_notebooks.py`, then train with `scripts/finetune/notebooks-lora.yaml`. Use `evaluate_notebooks.py` to compare base and tuned generations.

Export personal weights using the Liquid4All `onnx-export` checkout at `data/tools/onnx-export`, with its own environment prepared via `uv sync --no-dev`:

```sh
mlx_lm.fuse --model data/models/lfm2.5-350m-base --adapter-path data/personal-lfm/adapters --save-path data/personal-lfm/fused
data/tools/onnx-export/.venv/bin/python scripts/finetune/restore_lfm_hf_layout.py --base data/models/lfm2.5-350m-base --fused data/personal-lfm/fused
data/tools/onnx-export/.venv/bin/lfm2-export data/personal-lfm/fused --output-dir data/personal-lfm --output-name personal-350m --precision q4
```

**Do not skip convolution-layout restoration** between MLX fusion and ONNX export. Inspect generated continuations and check for memorization before using an export.

Copy `data/personal-lfm/exports/personal-350m/` into `public/models/personal-350m/`, keeping configs, tokenizer, and ONNX graph/external weights together. Open `/voices?debugLlmModel=personal-350m` on localhost. Personal weights remain local and unpublished. For notebook exports, use the corresponding `data/notebooks/` paths and `notebooks-350m` name.
