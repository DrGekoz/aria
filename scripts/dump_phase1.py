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


if __name__ == "__main__":
    model_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "parity_dumps"
    os.makedirs(out_dir, exist_ok=True)
    dump_number_cond(model_dir, out_dir)
