# aria.c — Implementation Status

_Snapshot of what actually works today. Forward-looking task detail lives in [ROADMAP.md](ROADMAP.md);
note that the ROADMAP's per-task checkboxes for the foundational epics (E1–E6) lag reality — the
work landed before the checkbox habit. This file is the source of truth for "what's done"._

## TL;DR

A from-scratch, dependency-free **C runtime for Stable Audio 3 (small-music)**: full
**text → audio** inference **+ continue/inpaint**, parity-verified against PyTorch/`stable-audio-tools`,
on **CPU** (AVX2/FMA + OpenMP) and **CUDA** (device-resident DiT, cuBLAS tensor-core GEMMs).
Milestones **M1 (end-to-end), M2 (self-contained text), M3 (continue/inpaint), M4 (CUDA
generation), M5 (Q4 quantization — medium fits low VRAM), M6 (medium model end-to-end)** reached.
Both **small-music and medium** generate end-to-end on **CPU and GPU** — medium adds a
differential DiT + a sliding-window/sinusoidal taae decoder, both device-resident on CUDA
(RTX 3070: 11.4 s → 2.2 s).

## What works (verified end-to-end)

- **Pipeline:** T5Gemma encoder → conditioning (cross-attn + global adaLN + seconds) → DiT
  denoise (8-step pingpong) → taae_v2 decode → stereo 44.1 kHz WAV.
- **Continue / inpaint:** taae_v2 **encoder** (audio→latent) + inpaint mask (audio→latent
  nearest-interp) + local-additive cond (`[mask | latent·mask]`) → `--continue <wav>` (extend a
  clip) / `--inpaint <wav> --from <s> --to <s>` (regenerate a region). End-to-end parity vs
  `generate_diffusion_cond_inpaint` (latent 6.6e-4, audio 2.4e-4); kept regions preserved. CPU DiT.
- **Conditioning inputs:** text prompt (BPE tokenizer + T5Gemma encoder, all in C) · precomputed
  `[256,768]` embedding (`--prompt-embed`) · unconditional (`--uncond`); `seconds_total`
  duration conditioner.
- **Backends**
  - **CPU:** register-blocked AVX2/FMA GEMM (175–280 GFLOP/s) + OpenMP + reusable arena
    workspace; optional BLAS (`make blas`). Pure-C fallback always compiled in.
  - **CUDA (`--device cuda`):** device-resident **DiT + decoder** — weights uploaded once (fp16),
    denoise loop *and* taae decode on the GPU, cuBLAS GemmEx (tensor cores) for linears, batched
    attention; T5Gemma encoder on CPU; `--device auto` picks the GPU when an sm_70+ card fits,
    else CPU.
- **CLI (`aria`):** `-m <model>` `-p "prompt"` | `--prompt-embed` | `--uncond`, `-d <seconds>`,
  `-s <steps>`, `--seed`, `--device auto|cpu|cuda`, `--precision fp32|q8|q4`,
  `--rng xoshiro|torch` (torch = PyTorch-matched noise), `--bench N` (warm timing),
  `-o <out.wav>`; per-step progress on a TTY;
  `--continue <wav>` / `--inpaint <wav> --from <s> --to <s>` (continue/inpaint);
  `ARIA_PROFILE=1` for per-stage timing; `--info`, `--list-tensors`.
- **Tests:** per-op + per-component + end-to-end parity (`test_ops/arena/wav/config/sampler/
  quant/schedule/number_cond/attn/dit/dit_full/quant_dit/dec/enc/inpaint/inpaint_e2e/t5enc/
  tokenizer/e2e`) + `test_cuda` (CUDA-vs-CPU op parity) + `make bench`. CPU↔GPU audio agree to
  ~0.6 % rel-RMS (fp16-weight level).

## Performance (10 s clip, 8 steps)

| backend | time | notes |
|---|---|---|
| CPU 8-core (dev box) | ~5 s | packed outer-product GEMM (490–540 GFLOP/s, ~2× the old dot-product) + SIMD RMSNorm + arena |
| CPU 20-core (RTX box) | 9.4 s | |
| **RTX 3070 warm fp16** | **0.29 s** | dit 0.16 + decode 0.03 (GPU) + setup 0.10 — full pipeline device-resident · **1464 MB VRAM** |
| RTX 3070 warm **q8** | 0.36 s | dit 0.27 · **1142 MB VRAM (−22 %)** · 2.45 % velocity — weights packed in VRAM, dequant-on-use |
| RTX 3070 warm **q4** | 0.35 s | dit 0.22 · **1080 MB VRAM (−26 %)** · 9.3 % velocity (asym + Q8 attention) |
| **medium** RTX 3070 (full GPU) | **2.2 s** | DiT 0.36 s (CPU 9.1 s, 26×) + **GPU decode 0.05 s** (CPU 2.0 s, 40×); CPU total 11.4 s → 2.2 s (5.2×); fp16 3.95 GB → **q4 2.84 GB**; GPU-vs-CPU audio 0.76 % |
| GT 1030 (2 GB) | slower than CPU | correctness-only (fits via fp16; weak Pascal) |

_GPU quant (q8/q4) costs ~20 % time for a 22–26 % VRAM cut — tensor cores keep the GEMM fast; the
saving is what lets the larger medium model fit. Tuned q4 is the smallest VRAM (below q8) at near
q8 fidelity. (Benchmarked on the RTX 3070, sm_86.)_
| _ref: stable-audio-tools (3070, warm)_ | _0.36 s_ | **aria is on par / ahead** (was 13× slower originally) |

PyTorch CPU reference is ~5.6 s warm; aria CPU is competitive once one-time weight fault-in is
excluded.

## Precision / quantization (current)

**`--precision fp32|q8|q4`** selects the CPU DiT weight precision (E9.0/E9.1/E9.3). The block
GEMMs dispatch through an `aria_qweight` overlay (`aria_quant.{c,h}`): fp32 keeps the zero-copy
mmap path (bit-identical), **q8** = per-row int8 (4.0× smaller: 1.6 GB → 361 MB; **2.45 %
velocity rel-RMS** — high fidelity), **q4** = a workflow-tuned mixed recipe: **asymmetric
(zero-point) int4 on the FFN + Q8 on the attention projections** (which feed qk-rmsnorm + softmax,
where error is amplified). That brings q4 from 29.8 % (uniform symmetric) to **9.3 %** velocity —
near q8 — at 300 MB (4.8×, still smaller than q8). `test_quant`
(hermetic) + `test_quant_dit` (model-level, 13 % gate) gate it. **On the GPU (E9.4)** `--precision
q8|q4` keeps the DiT block weights **packed in VRAM** and dequantizes on use into a reused fp16
scratch before the tensor-core GEMM, so resident VRAM shrinks — the path that lets medium fit.
Verified: GPU q4 is finite and matches the CPU q4 path (1.06 % rel-RMS, fp16-dequant level).
**Offline packing (E9.2):** `make quantize` → `aria-quantize <model> <out.aria> [q8|q4]` writes a
packed DiT overlay; `aria --load-quant <out.aria>` loads it (bit-identical to on-the-fly, md5-checked).
**Medium fit (M5):** the medium model (embed 1536, depth 24) loads + runs + **fits** on the RTX 3070
at q4 (GPU 2.84 GB fp16 → **1.91 GB q4**); the infra/quantization handle its scale. A *correct*
medium run still needs differential DiT attention (E4.5 → M6) — medium's `to_qkv` is `[5·ed,ed]`. **Caveat:** CPU quant is a *footprint* win, not
speed (scalar dequant GEMM ~6–8× slower than the AVX2 f32 kernel); the GPU speed win wants tensor
cores (RTX 3070, sm_86 — bench there). fp16/bf16 fall back to fp32/fp16 for now.

## Source map (`src/`)

- **Core:** `aria.{c,h}` (orchestrator + public API), `aria_model_sa3.c` (SA3 module), `main.c` (CLI).
- **Ops/backends:** `aria_ops.h` (op surface), `aria_cpu.c` (CPU kernels + AVX2 GEMM),
  `aria_quant.{c,h}` (dtypes + Q8/Q4 pack + dequant-on-use GEMM + `aria_qweight` dispatch),
  `aria_cuda.cu` (CUDA backend + device-resident DiT), `aria_gpu.h`, `aria_arena.{c,h}` (scratch arena).
- **Model components:** `aria_sa3_dit.{c,h}` (DiT + request context), `aria_sa3_dec.{c,h}`
  (taae decoder), `aria_sa3_enc.{c,h}` (taae encoder), `aria_taae.{c,h}` (shared resampling
  block + chunk pass), `aria_t5enc.{c,h}` (T5Gemma encoder), `aria_tokenizer.{c,h}` (BPE),
  `aria_cond.{c,h}` (number/timestep + inpaint mask/local-cond), `aria_sampler.{c,h}` (RNG + LogSNR + pingpong).
- **Infra:** `aria_safetensors.{c,h}` (mmap loader), `aria_wav.{c,h}`, `aria_json.{c,h}`,
  `aria_sa3_config.c`, `aria_parity.{c,h}`.

## Milestones

| | status |
|---|---|
| **M1** end-to-end text→audio (precomputed emb + injected noise) == Python | ✅ |
| **M2** self-contained text (T5Gemma encoder in C) | ✅ |
| **M3** continue / inpaint | ✅ |
| **M4** CUDA end-to-end generation | ✅ |
| **M5** quantized (Q4) — medium fits low VRAM | ✅ |
| **M6** medium model end-to-end | ✅ |
| M7 steering (TasteSteer) | ⬜ |

## Epics — status

| epic | status | notes |
|---|---|---|
| **E0** Scaffolding & infra | ✅ | repo, mmap loader, WAV, op surface, parity harness |
| **E1** Conditioning | ✅ | number/timestep/global-adaLN/cross-attn + **E1.6** local-add inpaint cond hook (now exercised with a real mask via **E7.3**) |
| **E2** Core ops | ✅ + perf | rope/attn/qk-norm/cross/softcap/differential/conv1d/GLU; **E2.9/9b/9c ✅** (AVX2 GEMM tuned 3×3, arena/KV-cache), **E2.10 ✅** (`--device auto`); E2.9c-pack/E2.10-mps deferred |
| **E3** T5Gemma encoder + tokenizer | ✅ | BPE tokenizer + 12-layer Gemma2 encoder + conditioner; `--prompt-embed` |
| **E4** SA3 DiT forward | ✅ | block-0 + full parity; **E4.5 ✅** differential attention (medium), synthetic 7.7e-7 |
| **E5** taae_v2 decoder | ✅ | softnorm + chunked resampling + unpatch, full latent→audio parity |
| **E6** Sampler + end-to-end | ✅ (E6.5 ⬜) | LogSNR + xoshiro + pingpong + e2e WAV; CFG/`--cfg` flag ⬜ (base-checkpoint only) |
| **E7** continue / inpaint | ✅ | taae **encoder** (parity 9e-4), inpaint mask + local-add cond (exact), `--continue`/`--inpaint`, e2e parity (audio 2.4e-4); CPU DiT (GPU local-cond ⬜) |
| **E8** CUDA backend | ✅ E8.1–E8.5d · ⬜ E8.5e/E8.6 | scaffold, all op kernels, device-resident **DiT + decoder**, profile-guided kernels (4.73→**0.29 s** warm — **on par with / ahead of PyTorch**); last micro-opts (E8.5e) + SSD streaming ⬜ |
| **E9** Precision & Quantization | ✅ E9.1–E9.4 · 🟡 E9.0 | `aria_quant` (Q8/Q4 pack + dequant GEMM) + `--precision` seam on **CPU and GPU**; q8 2.45 %, **q4 fidelity-tuned to 9.3 %** (asym int4 + Q8 attention); GPU packs weights in VRAM (dequant-on-use); offline **`aria-quantize`** + `--load-quant`. **M5 reached** (medium fits the 3070 at q4). 🟡 E9.0 fp16/bf16 CPU storage still falls back to fp32 |
| **E10** medium model | ✅ E10.1/E10.2 (CPU) | **medium end-to-end on CPU** — differential DiT (E4.5) + medium decoder (`aria_sa3_dec_medium`: sliding-window + sinusoidal FF); e2e parity audio 1.7e-2. GPU/quant for medium = perf follow-up |
| **E11** Release polish (v1.0.0) | ⬜ | API/install finalize, CLI UX, Philox RNG, docs, CI |
| **E12** Steering (TasteSteer) | ⬜ | latent / DiT-residual / cond-space hooks + `--steer` |
| **E13** Batch / server | ✅ E13.3 · ⬜ E13.1/2 | enc/dec arena done (concurrency-clean); request-parallel + batched-forward APIs ⬜ |

## Known gaps / notes

- **Text prompts need setup** on a fresh box: the gated T5Gemma weights (`download_model.sh`) +
  exported tokenizer (`scripts/export_tokenizer.py`). `--uncond` / `--prompt-embed` work without it.
- **GT 1030 (2 GB)** is correctness-validation only (fits via fp16; slower than the CPU). Real GPU
  speed is the **RTX 3070** (sm_86).
- **Continue/inpaint runs on the CPU DiT** — the device DiT has no local-cond path yet (follow-up).
  Init WAV must already be at the model sample rate (44.1 kHz); no resampler yet.
- **Quantization**: q8 usable, q4 coarse; on CPU it's a footprint win not speed (scalar dequant
  GEMM). On GPU the weights are packed in VRAM (dequant-on-use) — the medium-fits path — but the
  speed win needs tensor cores (verify on the RTX 3070). q4 fidelity tuning pending.
- Not implemented: medium (E10), steering (E12), batch/server APIs (E13.1/2), SSD streaming
  (E8.6), GPU inpaint, offline `aria-quantize` (E9.2).
