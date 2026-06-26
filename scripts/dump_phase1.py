"""dump_phase1.py - Dump PyTorch reference tensors for aria's Phase 1 parity tests.

Run with the sa3-sf-api venv so stable_audio_tools imports work, e.g.:

  SA3_MODEL=~/.cache/huggingface/hub/models--stabilityai--stable-audio-3-small-music/snapshots/<snap>
  .../sa3-sf-api/.venv/bin/python dump_phase1.py "$SA3_MODEL" out_dir

Currently dumps:
  number_cond/seconds_<i>.atns   reference seconds_total embeddings [768]
"""
import os
import sys
import json
import math
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(__file__))
from parity_io import save_atns

# Shared with tests/test_number_cond.c -- keep in sync.
SECONDS_VALUES = [0.0, 5.0, 15.0, 30.0, 47.3, 95.0, 200.0, 384.0]


def dump_number_cond(model_dir, out_dir):
    from stable_audio_tools.models.conditioners import NumberConditioner

    cfg = json.load(open(os.path.join(model_dir, "model_config.json")))
    # locate the seconds_total conditioner config
    cond_cfgs = cfg["model"]["conditioning"]["configs"]
    sec = next(c for c in cond_cfgs if c["id"] == "seconds_total")
    cond_dim = cfg["model"]["conditioning"].get("cond_dim", 768)
    min_val = sec["config"].get("min_val", 0)
    max_val = sec["config"].get("max_val", 384)
    fft = sec["config"].get("fourier_features_type", "expo")

    nc = NumberConditioner(output_dim=cond_dim, min_val=min_val, max_val=max_val,
                           fourier_features_type=fft).eval()

    # load the Linear weights (embedder.embedding.1) from the model safetensors
    st_path = os.path.join(model_dir, "model.safetensors")
    with safe_open(st_path, framework="pt") as f:
        w = f.get_tensor("conditioner.conditioners.seconds_total.embedder.embedding.1.weight")
        b = f.get_tensor("conditioner.conditioners.seconds_total.embedder.embedding.1.bias")
    nc.embedder.embedding[1].weight.data.copy_(w)
    nc.embedder.embedding[1].bias.data.copy_(b)
    W = w.float().cpu().numpy()  # [768, 256]
    B = b.float().cpu().numpy()  # [768]

    def canonical(s):
        # ExpoFourierFeatures computed in float64 (stable at large args), cast to
        # f32, then the float32 Linear -- matches aria_number_embed in C. torch's
        # all-float32 path is rounding noise at high freq*duration; see dump log.
        t = min((max(s, min_val) - min_val) / (max_val - min_val), 1.0)
        half = 128
        ramp = np.linspace(0.0, 1.0, half)
        freqs = np.exp(ramp * (math.log(10000.0) - math.log(0.5)) + math.log(0.5))
        args = t * freqs * 2 * math.pi
        fourier = np.concatenate([np.cos(args), np.sin(args)]).astype(np.float32)  # [256]
        return (fourier @ W.T + B).astype(np.float32)  # [768]

    d = os.path.join(out_dir, "number_cond")
    os.makedirs(d, exist_ok=True)
    worst_torch_dev = 0.0
    with torch.no_grad():
        for i, s in enumerate(SECONDS_VALUES):
            ref = canonical(s)
            save_atns(os.path.join(d, f"seconds_{i}.atns"), ref)
            t_emb, _mask = nc([s], device="cpu")  # torch float32 path, [1,1,768]
            dev = float(np.max(np.abs(t_emb.squeeze().float().cpu().numpy() - ref)))
            worst_torch_dev = max(worst_torch_dev, dev)
    print(f"dumped {len(SECONDS_VALUES)} seconds_total embeddings to {d} "
          f"(min={min_val} max={max_val} type={fft} dim={cond_dim}); "
          f"canonical(f64)-vs-torch(f32) worst dev = {worst_torch_dev:.3e}")


def dump_ops(out_dir):
    """Op-level references with seeded synthetic inputs, driving the real
    stable_audio_tools ops. Inputs are dumped too, so C consumes identical data."""
    import torch.nn.functional as F
    from stable_audio_tools.models.transformer import (
        RotaryEmbedding, apply_rotary_pos_emb, FeedForward,
    )

    d = os.path.join(out_dir, "ops")
    os.makedirs(d, exist_ok=True)

    def save(name, t):
        save_atns(os.path.join(d, name), t.detach().float().cpu().numpy())

    # --- RoPE (rotate-half) : H=2, N=6, D=64, rot_dim=64 ---
    torch.manual_seed(0)
    H, N, D, rot = 2, 6, 64, 64
    q = torch.randn(H, N, D)
    rotary = RotaryEmbedding(rot)
    freqs, _ = rotary.forward_from_seq_len(N)
    q_rot = apply_rotary_pos_emb(q, freqs)
    save("rope_q_in.atns", q)
    save("rope_q_out.atns", q_rot)

    # --- attention (SDPA) : H=2, Nq=4, Nk=6, D=8 ---
    torch.manual_seed(1)
    H, Nq, Nk, Dh = 2, 4, 6, 8
    qa = torch.randn(H, Nq, Dh)
    ka = torch.randn(H, Nk, Dh)
    va = torch.randn(H, Nk, Dh)
    out = F.scaled_dot_product_attention(qa, ka, va, is_causal=False)
    save("attn_q.atns", qa)
    save("attn_k.atns", ka)
    save("attn_v.atns", va)
    save("attn_out.atns", out)

    # --- GLU/SwiGLU FeedForward : dim=64, mult=4 -> inner=256 ---
    torch.manual_seed(2)
    dim, mult, Nf = 64, 4, 5
    ff = FeedForward(dim, mult=mult, glu=True, zero_init_output=False, no_bias=False).eval()
    x = torch.randn(1, Nf, dim)
    with torch.no_grad():
        y = ff(x)
    save("ff_x.atns", x.squeeze(0))
    save("ff_y.atns", y.squeeze(0))
    save("ff_Win.atns", ff.ff[0].proj.weight)   # [2*inner, dim]
    save("ff_bin.atns", ff.ff[0].proj.bias)     # [2*inner]
    save("ff_Wout.atns", ff.ff[2].weight)       # [dim, inner]
    save("ff_bout.atns", ff.ff[2].bias)         # [dim]
    print(f"dumped op references (rope/attn/ff) to {d}")


if __name__ == "__main__":
    model_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "parity_dumps"
    os.makedirs(out_dir, exist_ok=True)
    dump_number_cond(model_dir, out_dir)
    dump_ops(out_dir)
