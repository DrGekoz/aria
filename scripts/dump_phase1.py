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


def dump_dit_block(model_dir, out_dir):
    """Load DiT block 0's real weights into a TransformerBlock and dump a
    seeded forward (inputs + output). C loads the same weights from the model
    and must reproduce the output -- validates both name mapping and the block."""
    from stable_audio_tools.models.transformer import TransformerBlock, RotaryEmbedding

    dim, num_heads, head_dim = 1024, 16, 64
    block = TransformerBlock(
        dim, dim_heads=head_dim, cross_attend=True, dim_context=dim,
        global_cond_dim=dim, local_add_cond_dim=257,
        norm_type="rms_norm", norm_kwargs={"force_fp32": True},
        attn_kwargs={"qk_norm": "rms", "differential": False},
        ff_kwargs={"mult": 4.0},
    ).eval()

    prefix = "model.model.transformer.layers.0."
    sd = {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith(prefix):
                sd[k[len(prefix):]] = f.get_tensor(k)
    missing, unexpected = block.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "rope" not in m]  # rope buffer is recomputed
    if missing:
        print(f"  WARNING block0 missing keys: {missing}")

    torch.manual_seed(3)
    S, Sc = 40, 257
    x = torch.randn(1, S, dim)
    context = torch.randn(1, Sc, dim)
    global_cond = torch.randn(1, 6 * dim)  # already passed through global_cond_embedder
    rope = RotaryEmbedding(max(head_dim // 2, 32)).forward_from_seq_len(S)
    with torch.no_grad():
        out = block(x, context=context, global_cond=global_cond,
                    rotary_pos_emb=rope, local_add_cond=None)

    d = os.path.join(out_dir, "dit")
    os.makedirs(d, exist_ok=True)
    save_atns(os.path.join(d, "block0_x.atns"), x.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "block0_context.atns"), context.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "block0_global.atns"), global_cond.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "block0_out.atns"), out.squeeze(0).float().cpu().numpy())
    print(f"dumped DiT block-0 reference to {d} (S={S}, Sc={Sc})")


def dump_schedule(out_dir):
    """LogSNR schedule reference (build_schedule + LogSNRShift) for fixed params."""
    from stable_audio_tools.inference.sampling import build_schedule, LogSNRShift
    d = os.path.join(out_dir, "sampler")
    os.makedirs(d, exist_ok=True)
    shift = LogSNRShift(anchor_length=2000, anchor_logsnr=-6.2, rate=1.0, logsnr_end=2.0)
    for steps, seq_len in [(8, 323), (8, 1293), (16, 512)]:
        sched = build_schedule(steps=steps, sigma_max=1.0, dist_shift=shift,
                               effective_seq_len=seq_len, include_endpoint=True)
        save_atns(os.path.join(d, f"sched_s{steps}_L{seq_len}.atns"),
                  sched.float().cpu().numpy())
    print(f"dumped LogSNR schedules to {d}")


def dump_dit_full(model_dir, out_dir):
    """Full DiT denoiser_forward: instantiate the real DiffusionTransformer from
    the config, load model.model.* weights, run _forward on seeded synthetic
    conditioning (bypasses T5Gemma). C must reproduce the velocity output."""
    from stable_audio_tools.models.dit import DiffusionTransformer

    cfg = json.load(open(os.path.join(model_dir, "model_config.json")))
    dcfg = cfg["model"]["diffusion"]["config"]
    objective = cfg["model"]["diffusion"].get("diffusion_objective", "rf_denoiser")
    dit = DiffusionTransformer(diffusion_objective=objective, **dcfg).eval()

    prefix = "model.model."
    sd = {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith(prefix):
                sd[k[len(prefix):]] = f.get_tensor(k)
    missing, unexpected = dit.load_state_dict(sd, strict=False)
    missing = [x for x in missing if "rope" not in x and "inv_freq" not in x]
    if missing:
        print(f"  WARNING dit missing keys: {missing[:8]}{'...' if len(missing) > 8 else ''}")

    torch.manual_seed(5)
    T, n_cond = 16, 257
    x = torch.randn(1, dcfg["io_channels"], T)
    t = torch.tensor([0.3])
    cross = torch.randn(1, n_cond, dcfg["cond_token_dim"])
    glob = torch.randn(1, dcfg["global_cond_dim"])
    with torch.no_grad():
        out = dit._forward(x, t, cross_attn_cond=cross, global_embed=glob)

    d = os.path.join(out_dir, "dit")
    os.makedirs(d, exist_ok=True)
    save_atns(os.path.join(d, "full_x.atns"), x.squeeze(0).float().cpu().numpy())       # [C,T]
    save_atns(os.path.join(d, "full_cross.atns"), cross.squeeze(0).float().cpu().numpy())  # [257,768]
    save_atns(os.path.join(d, "full_global.atns"), glob.squeeze(0).float().cpu().numpy())  # [768]
    save_atns(os.path.join(d, "full_out.atns"), out.squeeze(0).float().cpu().numpy())     # [C,T]
    save_atns(os.path.join(d, "full_t.atns"), np.array([0.3], dtype=np.float32))
    print(f"dumped full DiT denoiser reference to {d} (T={T})")


def _load_decoder_ae(model_dir):
    from safetensors.torch import load_file
    from stable_audio_tools.models.autoencoders import create_autoencoder_from_config
    cfg = json.load(open(os.path.join(model_dir, "model_config.json")))
    ae_cfg = cfg["model"]["pretransform"]["config"]
    ae = create_autoencoder_from_config(
        {"model": ae_cfg, "sample_rate": cfg["sample_rate"]}).eval()
    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    prefix = "pretransform.model."
    ae_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = ae.load_state_dict(ae_sd, strict=False)
    missing = [x for x in missing if "encoder" not in x and "inv_freq" not in x]
    if missing:
        print(f"  WARNING decoder missing keys: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    # neutralize noise so the reference is deterministic (matches the C path)
    for mod in ae.modules():
        if hasattr(mod, "noise_regularize"):
            mod.noise_regularize = False
        if hasattr(mod, "mask_noise"):
            mod.mask_noise = 0.0
    return ae


def dump_decoder(model_dir, out_dir):
    """Staged taae_v2 decode reference (softnorm -> SAMEDecoder -> unpatch)."""
    ae = _load_decoder_ae(model_dir)
    d = os.path.join(out_dir, "dec")
    os.makedirs(d, exist_ok=True)

    orig_randn_like = torch.randn_like
    torch.randn_like = lambda x, *a, **k: torch.zeros_like(x)  # belt-and-suspenders determinism
    try:
        torch.manual_seed(0)
        T = 16
        latent = torch.randn(1, 256, T)
        with torch.no_grad():
            z = ae.bottleneck.decode(latent)        # [1,256,T]
            dec = ae.decoder(z)                     # [1,512,16T]
            audio = ae.pretransform.decode(dec)     # [1,2,4096T]
    finally:
        torch.randn_like = orig_randn_like

    save_atns(os.path.join(d, "latent.atns"), latent.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "after_softnorm.atns"), z.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "after_decoder.atns"), dec.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "audio.atns"), audio.squeeze(0).float().cpu().numpy())
    print(f"dumped taae decoder stages to {d} (T={T})")


def dump_taae_block(model_dir, out_dir):
    """Standalone transformers[0] block of the decoder resampling stack (best-effort,
    for localization). Skipped if the block can't be constructed standalone."""
    try:
        from stable_audio_tools.models.transformer import TransformerBlock
        from safetensors.torch import load_file
        blk = TransformerBlock(768, dim_heads=64, cross_attend=False, norm_type="dyt",
                               attn_kwargs={"qk_norm": "dyt", "differential": True},
                               add_rope=True).eval()
        sd = load_file(os.path.join(model_dir, "model.safetensors"))
        prefix = "pretransform.model.decoder.layers.3.transformers.0."
        bsd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)
               and "inv_freq" not in k}
        blk.load_state_dict(bsd, strict=False)
        torch.manual_seed(7)
        N = 34
        x = torch.randn(1, N, 768)
        with torch.no_grad():
            out = blk(x)
        d = os.path.join(out_dir, "dec")
        os.makedirs(d, exist_ok=True)
        save_atns(os.path.join(d, "block_x.atns"), x.squeeze(0).float().cpu().numpy())
        save_atns(os.path.join(d, "block_out.atns"), out.squeeze(0).float().cpu().numpy())
        print(f"dumped taae block-0 reference to {d} (N={N})")
    except Exception as e:
        print(f"  (skipping standalone taae block dump: {type(e).__name__}: {e})")


if __name__ == "__main__":
    model_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "parity_dumps"
    os.makedirs(out_dir, exist_ok=True)
    dump_number_cond(model_dir, out_dir)
    dump_ops(out_dir)
    dump_dit_block(model_dir, out_dir)
    dump_dit_full(model_dir, out_dir)
    dump_schedule(out_dir)
    dump_taae_block(model_dir, out_dir)
    dump_decoder(model_dir, out_dir)
