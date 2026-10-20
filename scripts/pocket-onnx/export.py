"""Export a Pocket TTS model to the five ONNX graphs the web worker runs.

    uv run --with-requirements scripts/pocket-onnx/requirements.txt \
        python scripts/pocket-onnx/export.py --out build/pocket-onnx/english_18m_refl_2026-10

Writes mimi_encoder.onnx (with speaker_proj), text_conditioner.onnx,
flow_lm_main.onnx and flow_lm_main_int8.onnx, flow_lm_flow.onnx,
mimi_decoder.onnx, tokenizer.model, bos_before_voice.npy, voices.bin (PTVB1)
and bundle.json, the manifest the worker reads before creating sessions.

Only the flow LM backbone gets an int8 variant, quantized the way upstream's
`quantize=True` does it: the attention and FFN weights of the transformer.
Upstream never quantizes the sampler head or Mimi, and int8 Mimi audio was
measurably off (max |diff| > 0.1 on a 0.86 peak), so those stay fp32.

Preset voices are encoded from their source wavs through the exported fp32
graphs (mimi_encoder -> BOS -> flow_lm_main prefill), so voices.bin always
matches the graphs it ships with.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from graphs import FlowLMFlow, FlowLMMain, MimiDecoder, MimiEncoder, TextConditioner  # noqa: E402

DEFAULT_MODEL = "english_18m_refl_2026-10"
OPSET = 17

# Voice names shipped in voices.bin, mapped to upstream's source recordings.
PRESET_VOICES = ["alba", "azelma", "cosette", "eponine", "fantine", "javert", "jean", "marius"]

# prepare_text_prompt: <= 4 words -> 3 frames after EOS, else 1; the stream adds 2.
FRAMES_AFTER_EOS = {"max_short_words": 4, "short": 5, "long": 3}


def load_tts_model(model: str, weights: str | None, tokenizer: str | None):
    from pocket_tts.models.tts_model import TTSModel
    from pocket_tts.utils.config import CONFIGS_DIR, load_config

    config_path = CONFIGS_DIR / f"{model}.yaml"
    config = load_config(config_path)
    if weights:
        config.weights_path = weights
    if tokenizer:
        config.flow_lm.lookup_table.tokenizer_path = tokenizer
    tts = TTSModel._from_pydantic_config_with_weights(
        config,
        temp=config.default_temperature,
        sampler_decode_steps=1,
        noise_clamp=None,
        eos_threshold=config.default_eos_threshold,
        origin=config_path,
    )
    if not tts.has_voice_cloning:
        raise SystemExit(
            "Loaded the weights without voice cloning (no speaker_proj). Accept the terms at "
            "https://huggingface.co/kyutai/pocket-tts and log in with `hf auth login`."
        )
    tts.eval()
    return tts, config


def export_graph(module, args, path: Path, input_names, output_names, dynamic_axes):
    module.eval()
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        torch.onnx.export(
            module,
            tuple(args),
            str(path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    import onnx

    onnx.checker.check_model(str(path))
    print(f"  {path.name}: {path.stat().st_size / 1e6:.1f} MB")


# Transformer attention (in_proj, out_proj) and FFN (linear1, linear2) MatMuls,
# upstream's RECOMMENDED_CONFIG; input_linear, z_proj and the EOS head stay fp32.
QUANTIZED_MATMUL = re.compile(r"/(in_proj|out_proj|linear1|linear2)(_\d+)?/MatMul$")


def quantize_backbone(src: Path, dst: Path):
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    nodes = [
        n.name
        for n in onnx.load(str(src)).graph.node
        if n.op_type == "MatMul" and QUANTIZED_MATMUL.search(n.name)
    ]
    if not nodes:
        raise RuntimeError(f"No transformer MatMuls found to quantize in {src.name}")
    quantize_dynamic(
        str(src),
        str(dst),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
        nodes_to_quantize=nodes,
        per_channel=True,
    )
    print(f"  {dst.name}: {dst.stat().st_size / 1e6:.1f} MB ({len(nodes)} MatMuls)")


def export_all(tts, out: Path) -> dict:
    flow_lm, mimi = tts.flow_lm, tts.mimi
    ldim = flow_lm.ldim
    d_model = flow_lm.dim
    z_dim = flow_lm.z_proj.out_features if flow_lm.z_proj is not None else d_model
    graphs: dict = {}

    print("text_conditioner")
    export_graph(
        TextConditioner(flow_lm),
        [torch.tensor([[1, 2, 3, 4]], dtype=torch.long)],
        out / "text_conditioner.onnx",
        ["token_ids"],
        ["embeddings"],
        {"token_ids": {1: "tokens"}, "embeddings": {1: "tokens"}},
    )
    graphs["text_conditioner"] = {"inputs": ["token_ids"], "outputs": ["embeddings"]}

    print("mimi_encoder")
    frame_size = mimi.frame_size
    export_graph(
        MimiEncoder(mimi, flow_lm),
        [torch.randn(1, 1, frame_size * 12) * 0.1],
        out / "mimi_encoder.onnx",
        ["audio"],
        ["conditioning"],
        {"audio": {2: "samples"}, "conditioning": {1: "frames"}},
    )
    graphs["mimi_encoder"] = {"inputs": ["audio"], "outputs": ["conditioning"]}

    print("flow_lm_main")
    main = FlowLMMain(flow_lm)
    capacity = 24
    states = [spec.initial(capacity) for spec in main.states]
    state_in = [spec.input_name for spec in main.states]
    state_out = [spec.output_name for spec in main.states]
    dyn = {"sequence": {1: "seq"}, "text_embeddings": {1: "text"}}
    for spec in main.states:
        dyn[spec.input_name] = dict(spec.dynamic_axes)
        dyn[spec.output_name] = dict(spec.dynamic_axes)
    export_graph(
        main,
        [torch.randn(1, 1, ldim), torch.randn(1, 3, d_model), *states],
        out / "flow_lm_main.onnx",
        ["sequence", "text_embeddings", *state_in],
        ["conditioning", "eos_logit", *state_out],
        dyn,
    )
    graphs["flow_lm_main"] = {
        "inputs": ["sequence", "text_embeddings", *state_in],
        "outputs": ["conditioning", "eos_logit", *state_out],
        "states": [spec.manifest() for spec in main.states],
    }

    print("flow_lm_flow")
    flow = FlowLMFlow(flow_lm)
    if flow.flow_type == "drifting":
        flow_args, flow_in = [torch.randn(1, z_dim), torch.randn(1, ldim)], ["c", "x"]
        flow_out = ["latent"]
    else:
        flow_args = [torch.randn(1, z_dim), torch.zeros(1, 1), torch.ones(1, 1), torch.randn(1, ldim)]
        flow_in, flow_out = ["c", "s", "t", "x"], ["flow_dir"]
    export_graph(flow, flow_args, out / "flow_lm_flow.onnx", flow_in, flow_out, {})
    graphs["flow_lm_flow"] = {"inputs": flow_in, "outputs": flow_out}

    print("mimi_decoder")
    decoder = MimiDecoder(mimi, flow_lm)
    dec_states = [spec.initial() for spec in decoder.states]
    dec_in = [spec.input_name for spec in decoder.states]
    dec_out = [spec.output_name for spec in decoder.states]
    export_graph(
        decoder,
        [torch.randn(1, 2, ldim), *dec_states],
        out / "mimi_decoder.onnx",
        ["latent", *dec_in],
        ["audio", *dec_out],
        {"latent": {1: "frames"}, "audio": {2: "samples"}},
    )
    graphs["mimi_decoder"] = {
        "inputs": ["latent", *dec_in],
        "outputs": ["audio", *dec_out],
        "states": [spec.manifest() for spec in decoder.states],
    }

    print("int8")
    quantize_backbone(out / "flow_lm_main.onnx", out / "flow_lm_main_int8.onnx")

    return {
        "graphs": graphs,
        "dims": {
            "latent": ldim,
            "conditioning": z_dim,
            "text_embedding": d_model,
            "kv_heads": main.states[1].shape[3],
            "head_dim": main.states[1].shape[4],
            "layers": len(main.states) - 1,
        },
        "flow_type": flow.flow_type,
        "decoder_steps_per_latent": decoder.steps_per_latent,
        "frame_size": frame_size,
    }


# ----------------------------------------------------------------------------
# Preset voices
# ----------------------------------------------------------------------------


def load_voice_wav(path: Path, sample_rate: int) -> np.ndarray:
    """Mono float32 at the model rate, ending on a pause, padded to whole frames."""
    from pocket_tts.data.audio import audio_read
    from pocket_tts.data.audio_utils import convert_audio, end_on_pause

    audio, sr = audio_read(path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = convert_audio(audio, sr, sample_rate, 1)
    audio = end_on_pause(audio, sample_rate)
    return audio.numpy().astype(np.float32)


def pad_to_frames(audio: np.ndarray, frame_size: int) -> np.ndarray:
    extra = (-audio.shape[-1]) % frame_size
    return np.pad(audio, [(0, 0)] * (audio.ndim - 1) + [(0, extra)])


def encode_voice(sessions: dict, bundle: dict, audio: np.ndarray, bos: np.ndarray) -> dict:
    """Voice-conditioned flow LM state, compact (cache length == prompt frames)."""
    audio = pad_to_frames(audio, bundle["frame_size"])[None]  # [1, 1, N]
    (cond,) = sessions["mimi_encoder"].run(None, {"audio": audio})
    prompt = np.concatenate([bos, cond], axis=1) if bos is not None else cond
    frames = prompt.shape[1]
    feeds = {
        "sequence": np.zeros((1, 0, bundle["dims"]["latent"]), np.float32),
        "text_embeddings": prompt.astype(np.float32),
    }
    for spec in bundle["graphs"]["flow_lm_main"]["states"]:
        shape = [frames if d == -1 else d for d in spec["shape"]]
        feeds[spec["name"]] = np.zeros(shape, np.int64 if spec["dtype"] == "int64" else np.float32)
    session = sessions["flow_lm_main"]
    outputs = dict(zip([o.name for o in session.get_outputs()], session.run(None, feeds)))
    return {spec["name"]: outputs[spec["output"]] for spec in bundle["graphs"]["flow_lm_main"]["states"]}


def write_ptvb(path: Path, voices: dict[str, dict[str, np.ndarray]]) -> dict[str, dict]:
    """PTVB1: magic, u32 voice count, then per voice a name and its tensors.

    Returns each voice's byte range so the worker can fetch one voice with a
    Range request.
    """
    ranges = {}
    with path.open("wb") as f:
        f.write(b"PTVB1")
        f.write(struct.pack("<I", len(voices)))
        for name, tensors in voices.items():
            start = f.tell()
            encoded = name.encode()
            f.write(struct.pack("<H", len(encoded)) + encoded)
            f.write(struct.pack("<H", len(tensors)))
            for tensor_name, array in tensors.items():
                tensor_bytes = tensor_name.encode()
                if array.dtype == np.float32:
                    code = 0
                elif array.dtype == np.int64:
                    code = 1
                else:
                    raise ValueError(f"Unsupported dtype {array.dtype} for {tensor_name}")
                data = np.ascontiguousarray(array).tobytes()
                f.write(struct.pack("<H", len(tensor_bytes)) + tensor_bytes)
                f.write(struct.pack("<BB", code, array.ndim))
                f.write(struct.pack(f"<{array.ndim}I", *array.shape))
                f.write(struct.pack("<I", len(data)))
                f.write(data)
            ranges[name] = {"start": start, "end": f.tell()}
    return ranges


def build_voices(out: Path, bundle: dict, voices_dir: Path, sample_rate: int) -> dict:
    import onnxruntime as ort

    from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

    sessions = {
        name: ort.InferenceSession(str(out / f"{name}.onnx"), providers=["CPUExecutionProvider"])
        for name in ("mimi_encoder", "flow_lm_main")
    }
    bos = np.load(out / "bos_before_voice.npy") if bundle["insert_bos_before_voice"] else None
    voices = {}
    for name in PRESET_VOICES:
        origin = _ORIGINS_OF_PREDEFINED_VOICES[name].removeprefix("hf://kyutai/tts-voices/")
        wav = voices_dir / origin.split("@")[0]
        audio = load_voice_wav(wav, sample_rate)
        voices[name] = encode_voice(sessions, bundle, audio, bos)
        frames = int(voices[name]["state_offset"][0])
        print(f"  {name}: {audio.shape[-1] / sample_rate:.1f}s -> {frames} frames")
    return write_ptvb(out / "voices.bin", voices)


# ----------------------------------------------------------------------------


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="pocket_tts/config/<model>.yaml")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weights", help="local model.safetensors (default: the config's hf:// path)")
    parser.add_argument("--tokenizer", help="local tokenizer.model (default: the config's hf:// path)")
    parser.add_argument("--voices-dir", type=Path, required=True,
                        help="local copy of the kyutai/tts-voices files behind the preset voices")
    args = parser.parse_args()

    torch.manual_seed(0)
    args.out.mkdir(parents=True, exist_ok=True)
    tts, config = load_tts_model(args.model, args.weights, args.tokenizer)
    flow_lm = tts.flow_lm

    exported = export_all(tts, args.out)

    from pocket_tts.utils.config import CONFIGS_DIR, load_config
    from pocket_tts.utils.utils import download_if_necessary

    # Recorded in bundle.json: the published source, not a local --weights copy.
    published_weights = load_config(CONFIGS_DIR / f"{args.model}.yaml").weights_path

    shutil.copyfile(
        download_if_necessary(str(config.flow_lm.lookup_table.tokenizer_path)),
        args.out / "tokenizer.model",
    )
    if flow_lm.insert_bos_before_voice:
        np.save(args.out / "bos_before_voice.npy", flow_lm.bos_before_voice.detach().float().numpy())

    bundle = {
        "format": "pocket-tts-onnx",
        "version": 2,
        "model": args.model,
        "weights": published_weights,
        "sample_rate": config.mimi.sample_rate,
        "frame_rate": config.mimi.frame_rate,
        "frame_size": exported["frame_size"],
        "dims": exported["dims"],
        "flow": {"type": exported["flow_type"], "decode_steps": 1},
        "temperature": tts.temp,
        "noise_clamp": None,
        "eos": {
            "threshold": tts.eos_threshold,
            "min_frames": tts._MIN_FRAMES_BEFORE_EOS,
            "frames_after_eos": FRAMES_AFTER_EOS,
        },
        "max_gen": {
            "tokens_per_second": tts._TOKENS_PER_SECOND_ESTIMATE,
            "padding_seconds": tts._GEN_SECONDS_PADDING,
        },
        "decoder_steps_per_latent": exported["decoder_steps_per_latent"],
        "insert_bos_before_voice": bool(flow_lm.insert_bos_before_voice),
        "graphs": exported["graphs"],
    }

    print("voices")
    ranges = build_voices(args.out, bundle, args.voices_dir, config.mimi.sample_rate)
    bundle["predefined_voices"] = list(ranges)
    bundle["voices"] = ranges

    bundle["files"] = {
        p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(args.out.iterdir())
        if p.is_file() and p.name != "bundle.json"
    }
    (args.out / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    print(f"wrote {args.out / 'bundle.json'}")


if __name__ == "__main__":
    main()
