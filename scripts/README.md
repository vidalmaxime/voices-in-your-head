# Development scripts

Run these tools from the repository root:

- `finetune/`: dataset preparation, training configurations, evaluation, and export repair for the workflows in the [project README](../README.md). Keep source texts, datasets, adapters, and model exports in ignored `data/` directories.
- `pocket-onnx/`: export Kyutai's Pocket TTS (`english_18m_refl_2026-10`) to the five ONNX graphs, `bundle.json` and `voices.bin` that `public/tts-onnx/inference-worker.js` runs, and check them against PyTorch. See below.
- `verify-tts-voice-ranges.mjs`: validate the published `bundle.json` voice ranges against `voices.bin` with `npm run verify:tts-voices`.

## Pocket TTS ONNX export

Needs Python 3.10+ and access to the gated [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts) weights (accept the terms, then `hf auth login`). The preset voices are encoded from their [kyutai/tts-voices](https://huggingface.co/kyutai/tts-voices) recordings, so download those first (`hf download kyutai/tts-voices <files> --local-dir <dir>`; `pocket_tts.utils.utils._ORIGINS_OF_PREDEFINED_VOICES` lists them).

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv -r scripts/pocket-onnx/requirements.txt
.venv/bin/python scripts/pocket-onnx/export.py --out build/english_18m_refl_2026-10 --voices-dir <tts-voices dir>
.venv/bin/python scripts/pocket-onnx/validate.py --bundle build/english_18m_refl_2026-10 --voices-dir <tts-voices dir> \
    --reference-voices <dir of upstream embeddings/*.safetensors>
```

`--weights` and `--tokenizer` take local copies instead of the config's `hf://` paths. Publish the output folder under `english_18m_refl_2026-10/` in the Hugging Face repo named in `app/tts/model-assets.ts` (with `MODEL_CARD.md` as the repo README), then pin `POCKET_MODEL_REVISION` to that commit.

## Tests

Run the dataset preparation tests without downloading models:

```sh
python3 -m unittest discover -s scripts/finetune -p 'test_*.py'
```
