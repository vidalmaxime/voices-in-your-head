# Development scripts

Run these tools from the repository root:

- `finetune/`: dataset preparation, training configurations, evaluation, and export repair for the workflows in the [project README](../README.md). Keep source texts, datasets, adapters, and model exports in ignored `data/` directories.
- `verify-tts-voice-ranges.mjs`: validate preset voice ranges with `npm run verify:tts-voices`.

## Tests

Run the dataset preparation tests without downloading models:

```sh
python3 -m unittest discover -s scripts/finetune -p 'test_*.py'
```
