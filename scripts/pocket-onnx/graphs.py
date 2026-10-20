"""Export-friendly forward passes for the five Pocket TTS ONNX graphs.

Upstream's streaming modules keep their state in dicts they mutate in place,
read the KV-cache offset with `.item()` and slice the cache to its valid
length. None of that survives tracing. These wrappers reuse the upstream
modules' weights but thread every piece of state through explicit tensors:

* flow LM self-attention writes new keys/values into a zero-initialised,
  fixed-capacity cache and attends over the whole cache with a causal mask.
  The capacity is a dynamic dimension chosen by the caller.
* the Mimi decoder transformer runs at 200 Hz (16 steps per latent) with a
  50-step context, so it keeps a sliding window of the last 50 keys/values
  plus a step counter for positions.
* streaming convolutions return their new `previous` / `partial` buffers
  instead of writing them in place. Zero-size buffers are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d
from pocket_tts.modules.seanet import SEANetResnetBlock
from pocket_tts.modules.transformer import StreamingTransformerLayer


# ----------------------------------------------------------------------------
# Attention
# ----------------------------------------------------------------------------


def _project_qkv(layer: StreamingTransformerLayer, x: torch.Tensor, rope_offset: torch.Tensor):
    sa = layer.self_attn
    b, t, _ = x.shape
    d = sa.dim_per_head
    projected = sa.in_proj(x)
    q = projected[..., : sa.attn_dim].reshape(b, t, sa.num_heads, d)
    kv = projected[..., sa.attn_dim :].reshape(b, t, 2, sa.num_kv_heads, d)
    k, v = torch.unbind(kv, dim=2)
    q, k = sa.rope(q, k, offset=rope_offset)
    return q, k, v


def _attend(layer: StreamingTransformerLayer, q, keys, values, mask):
    """q [B,T,H,D]; keys/values [B,L,Hkv,D]; mask [T,L] bool."""
    sa = layer.self_attn
    q = q.transpose(1, 2)
    keys = keys.transpose(1, 2)
    values = values.transpose(1, 2)
    if sa.num_kv_heads != sa.num_heads:
        rep = sa.num_heads // sa.num_kv_heads
        keys = keys.repeat_interleave(rep, dim=1)
        values = values.repeat_interleave(rep, dim=1)
    x = F.scaled_dot_product_attention(q, keys, values, mask[None, None], dropout_p=0.0)
    b, h, t, d = x.shape
    x = x.transpose(1, 2).reshape(b, t, h * d)
    return sa.out_proj(x)


def _layer_tail(layer: StreamingTransformerLayer, x_orig: torch.Tensor, update: torch.Tensor):
    x = x_orig + layer.layer_scale_1(update)
    return layer._ff_block(x)


def linear_cache_layer(
    layer: StreamingTransformerLayer, x: torch.Tensor, cache: torch.Tensor, offset: torch.Tensor
):
    """One transformer layer over a fixed-capacity KV cache.

    cache: [2, B, L, Hkv, D], zeros where nothing was written. offset: int64 [1],
    the absolute position of x[:, 0]. New keys/values are written to
    [offset, offset + T); every slot is attended with a causal mask, so
    unwritten (zero) slots past the queries are masked out.
    """
    b, t, _ = x.shape
    h = layer.norm1(x)
    q, k, v = _project_qkv(layer, h, offset)
    steps = torch.arange(t, device=x.device, dtype=torch.long)
    pos_q = offset + steps  # [T]
    new_kv = torch.stack([k, v], dim=0)  # [2, B, T, Hkv, D]
    # Write [offset, offset + T) as Gather + Where rather than ScatterElements,
    # which onnxruntime-web's WebGPU backend may not implement: every cache
    # slot gathers its would-be new entry and keeps it only inside the range.
    capacity = cache.shape[2]
    pos_k = torch.arange(capacity, device=x.device, dtype=torch.long)
    rel = pos_k - offset
    written = (rel >= 0) & (rel < t)
    gathered = torch.index_select(new_kv, 2, rel.clamp(0, t - 1))
    cache = torch.where(written.view(1, 1, capacity, 1, 1), gathered, cache)
    mask = pos_k[None, :] <= pos_q[:, None]
    update = _attend(layer, q, cache[0], cache[1], mask)
    return _layer_tail(layer, x, update), cache


def window_cache_layer(
    layer: StreamingTransformerLayer,
    x: torch.Tensor,
    window: torch.Tensor,
    step: torch.Tensor,
    context: int,
):
    """One transformer layer with a sliding window of the last `context` steps.

    window: [2, B, W, Hkv, D] holding positions [step - W, step); negative
    positions (before the stream started) are masked. Returns the last W
    keys/values of the concatenation as the next window.
    """
    b, t, _ = x.shape
    w = window.shape[2]
    h = layer.norm1(x)
    q, k, v = _project_qkv(layer, h, step)
    full = torch.cat([window, torch.stack([k, v], dim=0)], dim=2)  # [2, B, W + T, Hkv, D]
    pos_q = step + torch.arange(t, device=x.device, dtype=torch.long)
    pos_k = step - w + torch.arange(full.shape[2], device=x.device, dtype=torch.long)
    delta = pos_q[:, None] - pos_k[None, :]
    mask = (pos_k[None, :] >= 0) & (delta >= 0) & (delta < context)
    update = _attend(layer, q, full[0], full[1], mask)
    return _layer_tail(layer, x, update), full[:, :, -w:]


def stateless_layer(layer: StreamingTransformerLayer, x: torch.Tensor, context: int | None):
    """Full-sequence causal layer (no cache), with a mask built from the dynamic length."""
    t = x.shape[1]
    h = layer.norm1(x)
    zero = torch.zeros(1, dtype=torch.long, device=x.device)
    q, k, v = _project_qkv(layer, h, zero)
    pos = torch.arange(t, device=x.device, dtype=torch.long)
    delta = pos[:, None] - pos[None, :]
    mask = delta >= 0
    if context is not None:
        mask = mask & (delta < context)
    update = _attend(layer, q, k, v, mask)
    return _layer_tail(layer, x, update)


# ----------------------------------------------------------------------------
# Convolutions
# ----------------------------------------------------------------------------


def conv_history(conv: StreamingConv1d) -> int:
    return conv._effective_kernel_size - conv._stride


def conv_step(conv: StreamingConv1d, x: torch.Tensor, previous: torch.Tensor | None):
    """Streaming Conv1d with constant padding. Returns (y, next previous)."""
    assert conv.pad_mode == "constant", conv.pad_mode
    if previous is None:
        return conv.conv(x), None
    x = torch.cat([previous, x], dim=-1)
    return conv.conv(x), x[..., -previous.shape[-1] :]


def convtr_step(convtr: StreamingConvTranspose1d, x: torch.Tensor, partial: torch.Tensor | None):
    """Streaming ConvTranspose1d. Returns (y, next partial)."""
    y = convtr.convtr(x)
    if partial is None:
        return y, None
    pt = partial.shape[-1]
    y = torch.cat([y[..., :pt] + partial, y[..., pt:]], dim=-1)
    tail = y[..., -pt:]
    bias = convtr.convtr.bias
    if bias is not None:
        tail = tail - bias[:, None]
    return y[..., :-pt], tail


# ----------------------------------------------------------------------------
# State bookkeeping
# ----------------------------------------------------------------------------


@dataclass
class StateSpec:
    """One recurrent tensor of a graph: input `state_<name>`, output `out_state_<name>`."""

    name: str
    shape: list[int]  # initial shape; -1 marks a dynamic dimension
    dtype: str  # "float32" | "int64"
    dynamic_axes: dict[int, str]

    @property
    def input_name(self) -> str:
        return f"state_{self.name}"

    @property
    def output_name(self) -> str:
        return f"out_state_{self.name}"

    def initial(self, dynamic_size: int = 0) -> torch.Tensor:
        shape = [dynamic_size if d == -1 else d for d in self.shape]
        dtype = torch.long if self.dtype == "int64" else torch.float32
        return torch.zeros(shape, dtype=dtype)

    def manifest(self) -> dict:
        return {
            "name": self.input_name,
            "output": self.output_name,
            "shape": self.shape,
            "dtype": self.dtype,
            "dynamic": {str(k): v for k, v in self.dynamic_axes.items()},
        }


# ----------------------------------------------------------------------------
# Graphs
# ----------------------------------------------------------------------------


class TextConditioner(nn.Module):
    """token_ids int64 [1, T] -> embeddings [1, T, d_model]."""

    def __init__(self, flow_lm):
        super().__init__()
        self.conditioner = flow_lm.conditioner

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.conditioner(token_ids)


class MimiEncoder(nn.Module):
    """audio [1, 1, N] (N a multiple of the frame size) -> conditioning [1, T, d_model].

    SEANet encoder, encoder transformer, downsampling to 12.5 Hz and the flow
    LM's speaker projection, i.e. upstream's TTSModel._encode_audio.
    """

    def __init__(self, mimi, flow_lm):
        super().__init__()
        self.mimi = mimi
        self.speaker_proj_weight = flow_lm.speaker_proj_weight
        self.context = mimi.encoder_transformer.transformer.context

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        mimi = self.mimi
        x = audio
        for layer in mimi.encoder.model:
            if isinstance(layer, StreamingConv1d):
                x = self._conv(layer, x)
            elif isinstance(layer, SEANetResnetBlock):
                v = x
                for sub in layer.block:
                    v = self._conv(sub, v) if isinstance(sub, StreamingConv1d) else sub(v)
                x = x + v
            else:
                x = layer(x)

        proj = mimi.encoder_transformer
        z = x.transpose(1, 2)
        if proj.input_proj is not None:
            z = proj.input_proj(z)
        for layer in proj.transformer.layers:
            z = stateless_layer(layer, z, self.context)
        z = proj.output_projs[0](z).transpose(1, 2)

        if mimi.encoder_frame_rate != mimi.frame_rate:
            down = mimi.downsample.conv
            history = conv_history(down)
            assert down.pad_mode == "replicate"
            if history:
                z = torch.cat([z[..., :1].expand(-1, -1, history), z], dim=-1)
            z = down.conv(z)
        latents = z.transpose(1, 2).to(torch.float32)
        return F.linear(latents, self.speaker_proj_weight)

    @staticmethod
    def _conv(conv: StreamingConv1d, x: torch.Tensor) -> torch.Tensor:
        history = conv_history(conv)
        if history:
            x = F.pad(x, (history, 0))
        return conv.conv(x)


class FlowLMMain(nn.Module):
    """Flow LM backbone with a fixed-capacity KV cache per layer.

    Inputs: sequence [1, S, ldim] (NaN rows mean BOS), text_embeddings
    [1, T, d_model], state_offset int64 [1], state_cache_<i> [2, 1, L, H, D].
    Outputs: conditioning [1, z_dim] and eos_logit [1, 1] for the last
    position, then the updated states.
    """

    def __init__(self, flow_lm):
        super().__init__()
        self.flow_lm = flow_lm
        layers = flow_lm.transformer.layers
        sa = layers[0].self_attn
        self.states = [StateSpec("offset", [1], "int64", {})] + [
            StateSpec(
                f"cache_{i}",
                [2, 1, -1, sa.num_kv_heads, sa.dim_per_head],
                "float32",
                {2: "cache_len"},
            )
            for i in range(len(layers))
        ]

    def forward(self, sequence, text_embeddings, offset, *caches):
        m = self.flow_lm
        sequence = torch.where(torch.isnan(sequence), m.bos_emb, sequence)
        x = torch.cat([text_embeddings, m.input_linear(sequence)], dim=1)
        next_caches = []
        for layer, cache in zip(m.transformer.layers, caches):
            x, cache = linear_cache_layer(layer, x, cache, offset)
            next_caches.append(cache)
        x = m.out_norm(x)[:, -1].to(torch.float32)
        if m.z_proj is not None:
            x = m.z_proj(x)
        eos_logit = m.out_eos(x)
        return (x, eos_logit, offset + text_embeddings.shape[1] + sequence.shape[1], *next_caches)


class FlowLMFlow(nn.Module):
    """Sampler head.

    drifting: (c [1, z_dim], x [1, ldim] noise already scaled by sqrt(temp)) -> latent.
    lsd: (c, s [1, 1], t [1, 1], x) -> flow_dir, integrated by the caller.
    """

    def __init__(self, flow_lm):
        super().__init__()
        self.flow_net = flow_lm.flow_net
        self.flow_type = flow_lm.flow_type
        if self.flow_type not in ("drifting", "lsd"):
            raise ValueError(f"Unsupported flow type for export: {self.flow_type}")

    def forward(self, c, *args):
        return self.flow_net(c, *args)


class MimiDecoder(nn.Module):
    """latent [1, N, ldim] (flow LM space) -> audio [1, 1, N * frame_size], with states.

    De-normalises with the flow LM's emb_std/emb_mean, projects through the
    quantizer, upsamples to the encoder frame rate, runs the decoder
    transformer with a sliding window, then the SEANet decoder.
    """

    def __init__(self, mimi, flow_lm):
        super().__init__()
        self.mimi = mimi
        self.register_buffer("emb_std", flow_lm.emb_std.detach().clone().float())
        self.register_buffer("emb_mean", flow_lm.emb_mean.detach().clone().float())
        tr = mimi.decoder_transformer.transformer
        self.context = tr.context
        sa = tr.layers[0].self_attn

        self.states: list[StateSpec] = [StateSpec("step", [1], "int64", {})]
        self._conv_slots: dict[int, str] = {}

        def add_buffer(module, channels, length, prefix):
            if length <= 0:
                return
            name = f"{prefix}_{len(self.states)}"
            self.states.append(StateSpec(name, [1, channels, length], "float32", {}))
            self._conv_slots[id(module)] = name

        if mimi.encoder_frame_rate != mimi.frame_rate:
            up = mimi.upsample.convtr
            add_buffer(up, up.convtr.out_channels, up._kernel_size - up._stride, "upsample")
        for i in range(len(tr.layers)):
            self.states.append(
                StateSpec(
                    f"window_{i}",
                    [2, 1, self.context, sa.num_kv_heads, sa.dim_per_head],
                    "float32",
                    {},
                )
            )
        for layer in mimi.decoder.model:
            convs = layer.block if isinstance(layer, SEANetResnetBlock) else [layer]
            for conv in convs:
                if isinstance(conv, StreamingConv1d):
                    add_buffer(conv, conv.conv.in_channels, conv_history(conv), "conv")
                elif isinstance(conv, StreamingConvTranspose1d):
                    add_buffer(
                        conv, conv.convtr.out_channels, conv._kernel_size - conv._stride, "convtr"
                    )
        self.steps_per_latent = int(mimi.encoder_frame_rate / mimi.frame_rate)

    def forward(self, latent, *state_tensors):
        state = {spec.name: t for spec, t in zip(self.states, state_tensors)}
        nxt: dict[str, torch.Tensor] = {}
        mimi = self.mimi

        x = latent * self.emb_std + self.emb_mean
        x = mimi.quantizer(x.transpose(1, 2))
        if mimi.encoder_frame_rate != mimi.frame_rate:
            x = self._convtr(mimi.upsample.convtr, x, state, nxt)

        proj = mimi.decoder_transformer
        z = x.transpose(1, 2)
        if proj.input_proj is not None:
            z = proj.input_proj(z)
        step = state["step"]
        for i, layer in enumerate(proj.transformer.layers):
            z, nxt[f"window_{i}"] = window_cache_layer(
                layer, z, state[f"window_{i}"], step, self.context
            )
        nxt["step"] = step + z.shape[1]
        x = proj.output_projs[0](z).transpose(1, 2)

        for layer in mimi.decoder.model:
            if isinstance(layer, StreamingConv1d):
                x = self._conv(layer, x, state, nxt)
            elif isinstance(layer, StreamingConvTranspose1d):
                x = self._convtr(layer, x, state, nxt)
            elif isinstance(layer, SEANetResnetBlock):
                v = x
                for sub in layer.block:
                    v = self._conv(sub, v, state, nxt) if isinstance(sub, StreamingConv1d) else sub(v)
                x = x + v
            else:
                x = layer(x)
        return (x, *[nxt[spec.name] for spec in self.states])

    def _conv(self, conv, x, state, nxt):
        name = self._conv_slots.get(id(conv))
        y, previous = conv_step(conv, x, state[name] if name else None)
        if name:
            nxt[name] = previous
        return y

    def _convtr(self, convtr, x, state, nxt):
        name = self._conv_slots.get(id(convtr))
        y, partial = convtr_step(convtr, x, state[name] if name else None)
        if name:
            nxt[name] = partial
        return y
