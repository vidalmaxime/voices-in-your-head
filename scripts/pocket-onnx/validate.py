"""Check the exported ONNX graphs against the PyTorch model with onnxruntime.

    uv run --with-requirements scripts/pocket-onnx/requirements.txt \
        python scripts/pocket-onnx/validate.py --bundle build/pocket-onnx/english_18m_refl_2026-10 \
        --voices-dir <kyutai/tts-voices copy> [--reference-voices <dir of upstream voice .safetensors>]

Every graph is compared on the same inputs (fp32 tightly; the int8 flow LM
against the error of upstream's own int8 quantization), the
flow LM is teacher-forced through a voice prompt, text prompt and generated
frames, and one sentence is generated end to end with shared noise. The
generated wavs are written next to the bundle under validation/.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import scipy.io.wavfile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export import DEFAULT_MODEL, load_tts_model, load_voice_wav, pad_to_frames  # noqa: E402

from pocket_tts.modules.stateful_module import increment_steps, init_states  # noqa: E402
from pocket_tts.quantization import RECOMMENDED_CONFIG, apply_dynamic_int8  # noqa: E402

INT8_SLACK = 1.5
TEXT = "Hello world. I am Kyutai's Pocket TTS, running in your browser."

failures: list[str] = []


def check(label: str, ref, got, atol: float, rtol: float = 0.0):
    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    if ref.shape != got.shape:
        failures.append(f"{label}: shape {got.shape} != {ref.shape}")
        print(f"  FAIL {label}: shape {got.shape} != {ref.shape}")
        return
    err = np.abs(ref - got)
    max_err = float(err.max()) if err.size else 0.0
    scale = float(np.abs(ref).max()) if ref.size else 0.0
    ok = bool(np.all(err <= atol + rtol * np.abs(ref)))
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: max |diff| {max_err:.2e} (|ref| max {scale:.2e})")
    if not ok:
        failures.append(f"{label}: max |diff| {max_err:.2e}")


def check_relative(label: str, ref, got, bound: float):
    err = float(np.abs(np.asarray(ref, np.float64) - np.asarray(got, np.float64)).max())
    ok = err <= bound
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: max |diff| {err:.2e} (allowed {bound:.2e})")
    if not ok:
        failures.append(f"{label}: max |diff| {err:.2e} > {bound:.2e}")


def session(bundle_dir: Path, name: str) -> ort.InferenceSession:
    return ort.InferenceSession(str(bundle_dir / f"{name}.onnx"), providers=["CPUExecutionProvider"])


def run(sess: ort.InferenceSession, feeds: dict) -> dict:
    return dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))


def zero_states(specs, dynamic: int = 0) -> dict:
    return {
        s["name"]: np.zeros(
            [dynamic if d == -1 else d for d in s["shape"]],
            np.int64 if s["dtype"] == "int64" else np.float32,
        )
        for s in specs
    }


def advance(outputs: dict, specs) -> dict:
    return {s["name"]: outputs[s["output"]] for s in specs}


class TorchFlowLM:
    """Upstream streaming flow LM, stepped manually so each output can be compared."""

    def __init__(self, tts, capacity: int):
        self.flow_lm = tts.flow_lm
        self.state = init_states(self.flow_lm, batch_size=1, sequence_length=capacity)

    @torch.no_grad()
    def step(self, sequence: torch.Tensor, text_embeddings: torch.Tensor):
        m = self.flow_lm
        seq = torch.where(torch.isnan(sequence), m.bos_emb, sequence)
        out = m.backbone(m.input_linear(seq), text_embeddings, seq, model_state=self.state)
        # With an empty sequence backbone keeps every position ([:, -0:]); the
        # last one is still what the graph returns.
        out = out.to(torch.float32)[:, -1]
        eos = m.out_eos(out)
        increment_steps(m, self.state, increment=sequence.shape[1] + text_embeddings.shape[1])
        return out, eos


def validate_graphs(tts, bundle_dir: Path, bundle: dict, voices_dir: Path):
    flow_lm, mimi = tts.flow_lm, tts.mimi
    ldim = bundle["dims"]["latent"]
    main_specs = bundle["graphs"]["flow_lm_main"]["states"]
    dec_specs = bundle["graphs"]["mimi_decoder"]["states"]
    rng = np.random.default_rng(0)

    print("text_conditioner")
    tc = session(bundle_dir, "text_conditioner")
    tokens = flow_lm.conditioner.prepare(TEXT)
    ref_text = flow_lm.conditioner(tokens).detach().numpy()
    got_text = run(tc, {"token_ids": tokens.numpy()})["embeddings"]
    check("embeddings", ref_text, got_text, atol=1e-6)

    print("mimi_encoder")
    enc = session(bundle_dir, "mimi_encoder")
    from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES

    wav = voices_dir / _ORIGINS_OF_PREDEFINED_VOICES["alba"].removeprefix("hf://kyutai/tts-voices/")
    audio = pad_to_frames(load_voice_wav(wav, bundle["sample_rate"]), bundle["frame_size"])
    with torch.no_grad():
        ref_cond = tts._encode_audio(torch.from_numpy(audio)[None]).numpy()
    got_cond = run(enc, {"audio": audio[None]})["conditioning"]
    check("conditioning", ref_cond, got_cond, atol=2e-3, rtol=2e-3)
    short = audio[..., : bundle["frame_size"] * 7]
    with torch.no_grad():
        ref_short = tts._encode_audio(torch.from_numpy(short)[None]).numpy()
    check("conditioning (7 frames)", ref_short, run(enc, {"audio": short[None]})["conditioning"],
          atol=2e-3, rtol=2e-3)

    bos = np.load(bundle_dir / "bos_before_voice.npy") if bundle["insert_bos_before_voice"] else None
    prompt = np.concatenate([bos, ref_cond], axis=1) if bos is not None else ref_cond

    # int8 is held to upstream's own quantization: its error against fp32 may
    # be at most INT8_SLACK times that of `TTSModel.load_model(quantize=True)`.
    quantized = copy.deepcopy(tts)
    apply_dynamic_int8(quantized.flow_lm, RECOMMENDED_CONFIG)
    capacity = prompt.shape[1] + tokens.shape[1] + 20 + 7  # deliberately not the export size
    empty_seq = np.zeros((1, 0, ldim), np.float32)
    empty_text = np.zeros((1, 0, bundle["dims"]["text_embedding"]), np.float32)
    for variant in ("flow_lm_main", "flow_lm_main_int8"):
        int8 = variant.endswith("int8")
        print(variant)
        sess = session(bundle_dir, variant)
        ref = TorchFlowLM(tts, capacity)
        upstream_int8 = TorchFlowLM(quantized, capacity) if int8 else None
        states = zero_states(main_specs, capacity)
        # Worst error over the whole run, for this graph and for upstream int8.
        worst = {"conditioning": [0.0, 0.0], "eos_logit": [0.0, 0.0]}

        def compare(label, seq, text, report):
            nonlocal states
            ref_c, ref_eos = ref.step(torch.from_numpy(seq), torch.from_numpy(text))
            out = run(sess, {"sequence": seq, "text_embeddings": text, **states})
            states = advance(out, main_specs)
            pairs = {"conditioning": (ref_c.numpy(), out["conditioning"])}
            if seq.shape[1]:
                pairs["eos_logit"] = (ref_eos.numpy(), out["eos_logit"])
            if int8:
                q_c, q_eos = upstream_int8.step(torch.from_numpy(seq), torch.from_numpy(text))
                upstream = {"conditioning": q_c.numpy(), "eos_logit": q_eos.numpy()}
                for key, (r, g) in pairs.items():
                    worst[key][0] = max(worst[key][0], float(np.abs(r - g).max()))
                    worst[key][1] = max(worst[key][1], float(np.abs(r - upstream[key]).max()))
            elif report:
                for key, (r, g) in pairs.items():
                    check(f"{label} {key}", r, g, atol=2e-4, rtol=2e-4)
            return ref_c

        compare("voice prompt", empty_seq, prompt.astype(np.float32), True)
        compare("text prompt", empty_seq, ref_text.astype(np.float32), True)
        latent = np.full((1, 1, ldim), np.nan, np.float32)
        for step in range(20):
            ref_c = compare(f"step {step}", latent, empty_text, step in (0, 1, 19))
            with torch.no_grad():
                noise = torch.from_numpy(rng.standard_normal((1, ldim)).astype(np.float32))
                latent = flow_lm.flow_net(ref_c, noise).numpy()[:, None]
        if int8:
            for key, (err, upstream_err) in worst.items():
                check_relative(f"{key} over 22 calls (upstream int8: {upstream_err:.2e})",
                               0.0, err, INT8_SLACK * upstream_err + 1e-3)
        expected_offset = prompt.shape[1] + tokens.shape[1] + 20
        if int(states["state_offset"][0]) != expected_offset:
            failures.append(f"{variant}: offset {states['state_offset'][0]} != {expected_offset}")

    print("flow_lm_flow")
    sess = session(bundle_dir, "flow_lm_flow")
    c = rng.standard_normal((1, bundle["dims"]["conditioning"])).astype(np.float32)
    x = (rng.standard_normal((1, ldim)) * bundle["temperature"] ** 0.5).astype(np.float32)
    with torch.no_grad():
        ref_latent = flow_lm.flow_net(torch.from_numpy(c), torch.from_numpy(x)).numpy()
    check("latent", ref_latent, run(sess, {"c": c, "x": x})["latent"], atol=1e-5, rtol=1e-5)

    print("mimi_decoder")
    latents = rng.standard_normal((1, 13, ldim)).astype(np.float32)
    chunks = [(0, 1), (1, 5), (5, 6), (6, 13)]
    with torch.no_grad():
        mimi_state = init_states(mimi, batch_size=1, sequence_length=13 * 16 + 16)
        x = torch.from_numpy(latents) * flow_lm.emb_std + flow_lm.emb_mean
        ref_audio = []
        for start, end in chunks:
            ref_audio.append(mimi.decode_from_latent(x[:, start:end], mimi_state).numpy())
            increment_steps(mimi, mimi_state, increment=16 * (end - start))
    sess = session(bundle_dir, "mimi_decoder")
    states = zero_states(dec_specs)
    got = []
    for start, end in chunks:
        out = run(sess, {"latent": latents[:, start:end], **states})
        states = advance(out, dec_specs)
        got.append(out["audio"])
    check("audio (chunks 1/4/1/7)", np.concatenate(ref_audio, axis=-1), np.concatenate(got, axis=-1), atol=2e-4)


def validate_voices(bundle_dir: Path, bundle: dict, reference_dir: Path):
    """voices.bin against upstream's precomputed voice states."""
    import safetensors.numpy

    print("voices.bin vs upstream voice states")
    data = (bundle_dir / "voices.bin").read_bytes()
    for name, rng in bundle["voices"].items():
        ref_path = reference_dir / f"{name}.safetensors"
        if not ref_path.exists():
            continue
        ref = safetensors.numpy.load_file(str(ref_path))
        tensors = parse_ptvb_slice(data[rng["start"] : rng["end"]])
        ref_offset = int(ref["transformer.layers.0.self_attn/offset"].reshape(-1)[0]) \
            if "transformer.layers.0.self_attn/offset" in ref else None
        got_offset = int(tensors["state_offset"][0])
        print(f"  {name}: {got_offset} frames (upstream {ref_offset})")
        if ref_offset == got_offset:
            ref_cache = ref["transformer.layers.0.self_attn/cache"][:, :, :got_offset]
            check(f"{name} layer 0 cache", ref_cache, tensors["state_cache_0"], atol=5e-3, rtol=5e-3)


def parse_ptvb_slice(buf: bytes) -> dict[str, np.ndarray]:
    import struct

    off = 0
    (n,) = struct.unpack_from("<H", buf, off); off += 2 + n
    (count,) = struct.unpack_from("<H", buf, off); off += 2
    tensors = {}
    for _ in range(count):
        (n,) = struct.unpack_from("<H", buf, off); off += 2
        name = buf[off : off + n].decode(); off += n
        code, rank = struct.unpack_from("<BB", buf, off); off += 2
        shape = struct.unpack_from(f"<{rank}I", buf, off); off += 4 * rank
        (length,) = struct.unpack_from("<I", buf, off); off += 4
        dtype = np.float32 if code == 0 else np.int64
        tensors[name] = np.frombuffer(buf[off : off + length], dtype=dtype).reshape(shape)
        off += length
    assert off == len(buf), (off, len(buf))
    return tensors


def generate(tts, bundle_dir: Path, bundle: dict, voice_state: dict, int8: bool, noises) -> np.ndarray:
    """The worker's loop, in Python: text prompt, AR frames with EOS handling, streaming decode."""
    main = session(bundle_dir, "flow_lm_main_int8" if int8 else "flow_lm_main")
    flow = session(bundle_dir, "flow_lm_flow")
    dec = session(bundle_dir, "mimi_decoder")
    tc = session(bundle_dir, "text_conditioner")
    main_specs = bundle["graphs"]["flow_lm_main"]["states"]
    dec_specs = bundle["graphs"]["mimi_decoder"]["states"]
    ldim = bundle["dims"]["latent"]

    tokens = tts.flow_lm.conditioner.prepare(TEXT).numpy()
    max_frames = int(np.ceil((tokens.shape[1] / bundle["max_gen"]["tokens_per_second"]
                              + bundle["max_gen"]["padding_seconds"]) * bundle["frame_rate"]))
    voice_frames = int(voice_state["state_offset"][0])
    capacity = voice_frames + tokens.shape[1] + max_frames
    states = {}
    for spec in main_specs:
        value = voice_state[spec["name"]]
        if -1 in spec["shape"]:
            padded = np.zeros([capacity if d == -1 else d for d in spec["shape"]], np.float32)
            padded[:, :, : value.shape[2]] = value
            value = padded
        states[spec["name"]] = value
    text = run(tc, {"token_ids": tokens})["embeddings"]
    out = run(main, {"sequence": np.zeros((1, 0, ldim), np.float32), "text_embeddings": text, **states})
    states = advance(out, main_specs)

    words = len(TEXT.split())
    fae = bundle["eos"]["frames_after_eos"]
    frames_after_eos = fae["short"] if words <= fae["max_short_words"] else fae["long"]
    empty_text = np.zeros((1, 0, bundle["dims"]["text_embedding"]), np.float32)
    latent = np.full((1, 1, ldim), np.nan, np.float32)
    latents, eos_step = [], None
    for step in range(max_frames):
        out = run(main, {"sequence": latent, "text_embeddings": empty_text, **states})
        states = advance(out, main_specs)
        if out["eos_logit"][0, 0] > bundle["eos"]["threshold"] and eos_step is None \
                and step >= bundle["eos"]["min_frames"]:
            eos_step = step
        if eos_step is not None and step >= eos_step + frames_after_eos:
            break
        x = (noises[step] * bundle["temperature"] ** 0.5).astype(np.float32)
        latent = run(flow, {"c": out["conditioning"], "x": x})["latent"][:, None]
        latents.append(latent)
    dec_states = zero_states(dec_specs)
    audio = []
    for i in range(0, len(latents), 4):
        chunk = np.concatenate(latents[i : i + 4], axis=1)
        out = run(dec, {"latent": chunk, **dec_states})
        dec_states = advance(out, dec_specs)
        audio.append(out["audio"])
    print(f"  {'int8' if int8 else 'fp32'}: {len(latents)} frames, EOS at {eos_step}")
    return np.concatenate(audio, axis=-1)[0, 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--weights")
    parser.add_argument("--tokenizer")
    parser.add_argument("--voices-dir", type=Path, required=True)
    parser.add_argument("--reference-voices", type=Path)
    args = parser.parse_args()

    torch.manual_seed(0)
    bundle = json.loads((args.bundle / "bundle.json").read_text())
    tts, _ = load_tts_model(args.model, args.weights, args.tokenizer)

    validate_graphs(tts, args.bundle, bundle, args.voices_dir)
    if args.reference_voices:
        validate_voices(args.bundle, bundle, args.reference_voices)

    print("end to end (voice: alba)")
    data = (args.bundle / "voices.bin").read_bytes()
    alba = bundle["voices"]["alba"]
    voice_state = parse_ptvb_slice(data[alba["start"] : alba["end"]])
    noises = np.random.default_rng(1).standard_normal((1000, 1, bundle["dims"]["latent"]))
    out_dir = args.bundle / "validation"
    out_dir.mkdir(exist_ok=True)
    for int8 in (False, True):
        audio = generate(tts, args.bundle, bundle, voice_state, int8, noises)
        if not np.all(np.isfinite(audio)) or np.abs(audio).max() < 1e-3:
            failures.append(f"end to end {'int8' if int8 else 'fp32'}: silent or non-finite audio")
        path = out_dir / f"alba_{'int8' if int8 else 'fp32'}.wav"
        scipy.io.wavfile.write(path, bundle["sample_rate"], audio.astype(np.float32))
        print(f"  wrote {path} ({audio.shape[0] / bundle['sample_rate']:.2f}s)")

    if failures:
        print(f"\n{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
