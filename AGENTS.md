# AGENTS.md — working guide for aria.c

Conventions and ground rules for anyone (human or AI agent) working in this
repository. Read this before making changes. Keep it short and follow it.

## What this is

`aria.c` is a from-scratch, **dependency-free C inference runtime** for audio
diffusion models. First target: **Stable Audio 3** (small-music + medium). It is
designed to be **modular** so other audio diffusion models (e.g. AceStep 1.5)
plug in as new model modules without touching the kernels, sampler, or I/O.

It is the audio sibling of two reference runtimes (study them, don't copy
blindly): `../sa3-sf-api/references/iris.c` (FLUX/Z-Image image diffusion — the
diffusion architecture + modularity model; **no CUDA**) and
`../sa3-sf-api/references/ds4` (DeepSeek LLMs — the CUDA backend, int8/q4
quantization with dequant-on-use, mmap + SSD weight streaming, compile-time
backend selection).

## Golden rules

1. **Dependency-free by default; optional accelerated backends behind build
   flags.** The default build (`make`) uses only standard C11 + libm + OpenMP and
   must always work and stay the reference. Its CPU GEMM is a hand-rolled packed
   AVX2/FMA microkernel (near-MKL, no BLAS dependency). The only optional backend
   is `-DARIA_CUDA` (cuda + cuBLAS); the pure-C path must stay compiled in. We
   borrow only kernels, never a framework — no BLAS, no libsndfile, no JSON libs,
   no curl, no ONNX/LibTorch/ggml-as-a-dep.
2. **CPU-first.** The CPU path is the primary, highest-quality target and must
   always build and pass with `make` / `make test` (no GPU). CUDA is an optional
   accelerator.
3. **TDD / parity-gated.** Every non-trivial op or component ships with a test.
   Numerical components are validated against the **real** PyTorch reference
   (`stable_audio_tools`) via the `.atns` parity harness (see below). No
   "looks right" — prove it against a dump.
4. **Simplicity first.** Minimum code that solves the problem. No speculative
   abstractions, no configurability nobody asked for, no error handling for
   impossible states. If 200 lines could be 50, write 50.
5. **Surgical changes.** Touch only what the task needs. Match surrounding
   style. Don't refactor or reformat adjacent code. Remove only orphans *your*
   change created.
6. **Config-driven, not hardcoded.** Read dims from `model_config.json` when a
   value is available; don't bake in magic numbers that the config already
   carries.

## Build & test

```sh
make                       # CPU build (pure C) -> ./aria + build/libaria.a
make test                  # hermetic unit tests (no model, no GPU). MUST stay green.
make parity ARIA_MODEL=<model_dir>   # parity tests vs PyTorch dumps (needs the venv)
make cuda CUDA_ARCH=sm_86  # CUDA build (Phase 3); sm_61 for the GT 1030
make clean
```

- `make test` must never require a model, network, or GPU. Parity tests that
  need a model live behind `ARIA_MODEL`/`ARIA_DUMPS` env vars and **SKIP**
  cleanly when unset, so they don't break the hermetic suite.
- `PYTHON` defaults to `../sa3-sf-api/.venv/bin/python` (the parity reference
  venv with `stable_audio_tools`).
- Build must be warning-clean under `-Wall -Wextra`.

## Repository layout

```
src/
  aria.h / aria.c            public API + orchestrator (load, registry, generate)
  aria_model.h               internal model-module vtable + aria_ctx struct
  aria_model_sa3.c           Stable Audio 3 module (detect/load/generate, device routing)
  aria_sa3_config.c          model_config.json -> aria_sa3_config
  aria_sa3_dit.{c,h}         SA3 DiT forward (standard + differential attention)
  aria_taae.{c,h}            shared taae_v2 transformer block (small + medium kernels)
  aria_sa3_dec.{c,h}         taae_v2 decoder (small-music) + medium decoder decls
  aria_sa3_dec_medium.c      taae_v2 decoder (medium: sliding-window + sinusoidal FF)
  aria_sa3_enc.c             taae_v2 encoder (audio->latent, for continue/inpaint)
  aria_t5enc.c               T5Gemma text encoder
  aria_tokenizer.c           tokenizer (exported binary loaded at runtime)
  aria_cond.{c,h}            conditioners (number, timestep, assembly, inpaint local)
  aria_sampler.{c,h}         RNG (xoshiro256**) + LogSNR schedule + pingpong
  aria_arena.{c,h}           bump allocator (hot-path scratch reuse)
  aria_ops.h                 backend-agnostic op surface
  aria_cpu.c                 CPU kernels (AVX2/FMA + OpenMP)
  aria_cuda.cu               CUDA backend (device-resident DiT + decoders, cuBLAS)
  aria_quant.{c,h}           Q8 / asymmetric-Q4 pack + dequant-on-use
  aria_safetensors.*         mmap weight loader
  aria_wav.*                 WAV I/O
  aria_json.*                tiny config reader
  aria_parity.*              .atns reader for parity tests
  main.c                     CLI
examples/generate.c          minimal external program linking libaria.a
tests/                       unit + parity tests
scripts/                     export_tokenizer.py, dump_phase1.py (PyTorch reference dumps)
tools/aria_quantize.c        offline DiT quantizer (model.safetensors -> .aria overlay)
```

The CUDA backend (`aria_cuda.cu`), Q8/Q4 quantization, continue/inpaint, and the
medium model are all implemented (see [STATUS.md](STATUS.md)); the pure-C `make`
build stays the default and the always-green reference.

## Architecture (and how to add a model)

Five layers: **CLI → public API (`aria.h`) → orchestrator (`aria.c`) → model
modules → op library (`aria_ops.h`) + backends (`aria_cpu.c` / `aria_cuda.cu`)**.

A model is one `aria_model_module` vtable (`aria_model.h`):
`detect / load / unload / generate`. The registry is the array `g_registry[]` in
`aria.c`. **To add a model** (e.g. AceStep 1.5): write `aria_dit_<model>.c` (+ AE
/ encoder modules as needed) built from the shared op surface, expose an
`aria_module_<model>` descriptor, and append it to `g_registry[]`. Do **not**
modify kernels, the sampler, quantization, or I/O to fit a new model — extend the
op surface generically if a primitive is genuinely missing.

## Naming conventions

- `aria_` prefix for shared/public identifiers.
- `_sa3` postfix for Stable-Audio-3-specific internals; future models use their
  own postfix (`_acestep`, ...).
- Public API functions route by model internally and take **no** model postfix.
- Component namespaces kept as-is: `safetensors_*`, and `.gamma` is the
  on-disk name for RMSNorm weights in SA3.

## Precision & quantization

- Default: **fp16 storage + fp32 compute**; activations fp32 (SA3 norms set
  `force_fp32`). The op library computes in fp32.
- Numerically ill-conditioned scalar ops (e.g. the Fourier duration/timestep
  embeddings, whose `arg` reaches ~6e4) are computed in **double** in C — cheap
  and stable. Don't bit-match PyTorch's float32 noise there; gate against an
  f64-canonical reference.
- int8 (Q8, per-row scale) then q4 (block) are added in Phase 4 with
  dequant-on-use inside gemm (ds4 pattern), validated op-by-op before trust.

## Parity methodology (how we prove correctness)

- `.atns` format (`scripts/parity_io.py` writer, `src/aria_parity.*` reader):
  `"ATNS" | u32 version | u32 ndim | u32 dtype(0=f32) | i64 shape[] | f32 data`.
- Dump references from the **real** `stable_audio_tools` modules
  (`scripts/dump_phase1.py`), not a Python reimplementation — except where a
  computation is ill-conditioned in f32 (then dump an f64-canonical value and
  log the deviation from torch).
- **Gate with atol + rtol**: `pass if maxabsdiff <= atol + rtol * max|ref|`.
  Float32 matmul accumulation order legitimately differs from numpy/BLAS by
  ~1e-3·max|ref|; a fixed tiny atol will false-fail. Typical: `atol≈3e-4`,
  `rtol≈2e-3` for linear-heavy ops; tighten for elementwise.
- For RNG-sensitive paths (sampler), inject identical noise from Python into the
  C path so you validate the *math* independently of the RNG.
- Determinism: tests must be deterministic. Seed explicitly.

## Hardware targets

- **CPU (primary):** i7-class, AVX2 + FMA + F16C, OpenMP. Must be correct and the
  reference for parity.
- **CUDA (secondary):** RTX 3070 (sm_86, fp16 tensor cores) is the realistic GPU
  target; GT 1030 (sm_61, 2 GB, slow fp16) is an extreme low-VRAM edge handled
  with fp16-storage + SSD weight streaming. CUDA toolkit 11.2.

## Don'ts

- Don't commit model weights, `.safetensors`, `.atns` dumps, audio, or `build/`.
- Don't add dependencies (see Golden rule 1).
- Don't download gated weights without explicit user approval.
- Don't leave `make test` red or introduce compiler warnings.
- Don't hardcode dims that `model_config.json` provides.

## Quick SA3 reference (small-music)

Latent 256-D @ ~10.7 Hz, stereo 44.1 kHz, downsample 4096×. DiT: embed_dim 1024,
depth 20, 16 heads, head_dim 64, FFN GLU (`ff.0.proj[8192,1024]`→`ff.2[1024,4096]`),
fused `to_qkv[3072,1024]`, QK-RMSNorm `q/k_norm.gamma[64]`, RMSNorm weights named
`.gamma`, adaLN `to_scale_shift_gate[6144]`, 64 memory tokens, zero-init inpaint
MLP `to_local_embed.0[1024,257]`. Conditioning: cross-attn `[prompt | seconds]`,
global adaLN from `seconds_total`, local-additive `[inpaint_mask | masked_input]`.
Autoencoder: `taae_v2`, patch 256, stride 16, differential attention + DynamicTanh
+ softnorm bottleneck. Objective `rf_denoiser`; sampler **pingpong** (8 steps,
cfg 1.0): `x ← (1−t_next)·(x − t·v) + t_next·noise`. Text encoder: T5Gemma-b-b
(12-layer Gemma2-style, 768-d, RoPE θ=10000, logit softcap 50, query_pre_attn
scalar 64); weights + tokenizer cached under the model's `t5gemma-b-b-ul2/`.
Always confirm against the live `model_config.json`.
