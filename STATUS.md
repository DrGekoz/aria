# aria.c — Implementation Status

_Snapshot of what actually works today. Forward-looking task detail lives in [ROADMAP.md](ROADMAP.md);
note that the ROADMAP's per-task checkboxes for the foundational epics (E1–E6) lag reality — the
work landed before the checkbox habit. This file is the source of truth for "what's done"._

## TL;DR

A from-scratch, dependency-free **C runtime for Stable Audio 3 (small-music)**: full
**text → audio** inference, parity-verified against PyTorch/`stable-audio-tools`, on **CPU**
(AVX2/FMA + OpenMP) and **CUDA** (device-resident DiT, cuBLAS tensor-core GEMMs). Milestones
**M1 (end-to-end), M2 (self-contained text), M4 (CUDA generation)** reached.

## What works (verified end-to-end)

- **Pipeline:** T5Gemma encoder → conditioning (cross-attn + global adaLN + seconds) → DiT
  denoise (8-step pingpong) → taae_v2 decode → stereo 44.1 kHz WAV.
- **Conditioning inputs:** text prompt (BPE tokenizer + T5Gemma encoder, all in C) · precomputed
  `[256,768]` embedding (`--prompt-embed`) · unconditional (`--uncond`); `seconds_total`
  duration conditioner.
- **Backends**
  - **CPU:** register-blocked AVX2/FMA GEMM (175–280 GFLOP/s) + OpenMP + reusable arena
    workspace; optional BLAS (`make blas`). Pure-C fallback always compiled in.
  - **CUDA (`--device cuda`):** device-resident DiT — weights uploaded once (fp16), denoise loop
    on the GPU, cuBLAS GemmEx (tensor cores) for linears, batched attention; encoder/decoder on
    CPU; auto-fallback to CPU if no device / insufficient VRAM.
- **CLI (`aria`):** `-m <model>` `-p "prompt"` | `--prompt-embed` | `--uncond`, `-d <seconds>`,
  `-s <steps>`, `--seed`, `--device cpu|cuda`, `--bench N` (warm timing), `-o <out.wav>`;
  `ARIA_PROFILE=1` for per-stage timing; `--info`, `--list-tensors`.
- **Tests:** per-op + per-component + end-to-end parity (`test_ops/arena/wav/config/sampler/
  schedule/number_cond/attn/dit/dit_full/dec/t5enc/tokenizer/e2e`) + `test_cuda` (CUDA-vs-CPU op
  parity) + `make bench`. CPU↔GPU audio agree to ~0.6 % rel-RMS (fp16-weight level).

## Performance (10 s clip, 8 steps)

| backend | time | notes |
|---|---|---|
| CPU 8-core (dev box) | ~9 s | AVX2 microkernel + arena |
| CPU 20-core (RTX box) | 9.4 s | |
| **RTX 3070 warm** | **1.28 s** | dit 0.42 + decode 0.76 (CPU) + setup 0.09 |
| GT 1030 (2 GB) | slower than CPU | correctness-only (fits via fp16; weak Pascal) |
| _ref: stable-audio-tools (3070, warm)_ | _0.36 s_ | gap **3.6×** (was 13× before the perf pass) |

PyTorch CPU reference is ~5.6 s warm; aria CPU is competitive once one-time weight fault-in is
excluded.

## Precision / quantization (current)

**Hardcoded per backend — no `--precision` flag yet, no quantization.** CPU = fp32;
CUDA = fp16 weights / fp32-accumulate. Selectable precision + Q8/Q4 are **E9** (E9.0 = the
`--precision` interface for the fp variants, E9.1–E9.4 = quant).

## Source map (`src/`)

- **Core:** `aria.{c,h}` (orchestrator + public API), `aria_model_sa3.c` (SA3 module), `main.c` (CLI).
- **Ops/backends:** `aria_ops.h` (op surface), `aria_cpu.c` (CPU kernels + AVX2 GEMM),
  `aria_cuda.cu` (CUDA backend + device-resident DiT), `aria_gpu.h`, `aria_arena.{c,h}` (scratch arena).
- **Model components:** `aria_sa3_dit.{c,h}` (DiT + request context), `aria_sa3_dec.{c,h}`
  (taae decoder), `aria_t5enc.{c,h}` (T5Gemma encoder), `aria_tokenizer.{c,h}` (BPE),
  `aria_cond.c` (number/timestep), `aria_sampler.{c,h}` (RNG + LogSNR + pingpong).
- **Infra:** `aria_safetensors.{c,h}` (mmap loader), `aria_wav.{c,h}`, `aria_json.{c,h}`,
  `aria_sa3_config.c`, `aria_parity.{c,h}`.

## Milestones

| | status |
|---|---|
| **M1** end-to-end text→audio (precomputed emb + injected noise) == Python | ✅ |
| **M2** self-contained text (T5Gemma encoder in C) | ✅ |
| **M4** CUDA end-to-end generation | ✅ |
| M3 continue / inpaint | ⬜ |
| M5 quantized (Q4) | ⬜ |
| M6 medium model | ⬜ |
| M7 steering (TasteSteer) | ⬜ |

## Epics — status

| epic | status | notes |
|---|---|---|
| **E0** Scaffolding & infra | ✅ | repo, mmap loader, WAV, op surface, parity harness |
| **E1** Conditioning | ✅ (E1.6 ⬜) | number/timestep/global-adaLN/cross-attn done; local-add inpaint cond ⬜ (rides on E7) |
| **E2** Core ops | ✅ + perf | rope/attn/qk-norm/cross/softcap/differential/conv1d/GLU done; **E2.9 + E2.9b ✅** (AVX2 GEMM, arena/KV-cache); E2.9c, E2.10 ⬜ |
| **E3** T5Gemma encoder + tokenizer | ✅ | BPE tokenizer + 12-layer Gemma2 encoder + conditioner; `--prompt-embed` |
| **E4** SA3 DiT forward | ✅ (E4.5 ⬜) | block-0 + full 20-block parity; differential-attn path for medium ⬜ |
| **E5** taae_v2 decoder | ✅ | softnorm + chunked resampling + unpatch, full latent→audio parity |
| **E6** Sampler + end-to-end | ✅ (E6.5 ⬜) | LogSNR + xoshiro + pingpong + e2e WAV; CFG/`--cfg` flag ⬜ (base-checkpoint only) |
| **E7** continue / inpaint | ⬜ | taae **encoder**, inpaint mask, local-add cond, `--continue`/`--inpaint` |
| **E8** CUDA backend | ✅ E8.1–E8.5b · ⬜ E8.5c/E8.6 | scaffold, all op kernels, device-resident DiT, perf pass (4.73→1.28 s); GPU decoder + SSD streaming ⬜ |
| **E9** Precision & Quantization | ⬜ | E9.0 `--precision` (fp32/fp16/bf16), E9.1–E9.4 Q8/Q4 + `aria-quantize` |
| **E10** medium model | ⬜ | embed 1536 / depth 24 / differential DiT attn |
| **E11** Release polish (v1.0.0) | ⬜ | API/install finalize, CLI UX, Philox RNG, docs, CI |
| **E12** Steering (TasteSteer) | ⬜ | latent / DiT-residual / cond-space hooks + `--steer` |
| **E13** Batch / server | ✅ E13.3 · ⬜ E13.1/2 | enc/dec arena done (concurrency-clean); request-parallel + batched-forward APIs ⬜ |

## Known gaps / notes

- **Text prompts need setup** on a fresh box: the gated T5Gemma weights (`download_model.sh`) +
  exported tokenizer (`scripts/export_tokenizer.py`). `--uncond` / `--prompt-embed` work without it.
- **GT 1030 (2 GB)** is correctness-validation only (fits via fp16; slower than the CPU). Real GPU
  speed is the **RTX 3070** (sm_86).
- Not implemented: continue/inpaint (E7), quantization (E9), medium (E10), steering (E12),
  batch/server APIs (E13.1/2), SSD streaming (E8.6).
