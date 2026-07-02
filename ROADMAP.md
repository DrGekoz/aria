# ROADMAP — aria.c to v1.0.0

Path from the current scaffold to a functional **v1.0.0** first release: Stable
Audio 3 (small-music + medium), text→audio + continue/inpaint, CPU **and** CUDA,
int8/q4 quantization, as a library + CLI — all parity-gated.

## Legend

- Status: `✅` done · `🟡` in progress · `⬜` todo
- Priority: **P0** critical path to first audio · **P1** required for v1.0.0 ·
  **P2** polish/optimization
- `deps:` task IDs that must land first · `∥` = no dependencies, startable now /
  parallel-safe within its group
- **Verify:** the concrete check that closes the task (a test, a parity gate, a
  build, or a manual run). No task is "done" without its verify passing.

## Milestones

| ID | Milestone | Closes at |
|----|-----------|-----------|
| **M1** ✅ | First audio: `text→audio` (precomputed embeddings + injected noise) matches Python | E6.4 |
| **M2** ✅ | Self-contained text: T5Gemma encoder in C | E3.4 + E3.5 |
| **M3** ✅ | continue / inpaint working | E7.4 |
| **M4** ✅ | CUDA end-to-end generation | E8.5 |
| **M5** ✅ | Quantized (Q4) — medium fits low VRAM | E9.4 |  ← Q4 mechanism (CPU+GPU), fidelity-tuned (9.3%), offline `aria-quantize`; medium loads/runs/**fits** on the 3070 at q4 (1.91 GB). A *correct* medium run needs differential DiT attn (E4.5 → M6). |
| **M6** ✅ | medium model end-to-end | E10.2 |  ← differential DiT + medium decoder (sliding-window/sinusoidal); e2e parity latent 7.3e-3 / audio 1.7e-2; runs on **CPU and GPU** (GPU 0.54 s ≈ stable-audio-tools) |
| **M7** 🟡 | Steering: DiT-residual arm done (add+project, CPU+GPU, graph-resident, CLI/server); latent + cond arms open | E12.7 ✅ (residual); E12.2/E12.4 open |
| **v1.0.0** | Release checklist green | E11.7 |

## Status snapshot

Milestones **M1–M6** reached (**M6: medium model end-to-end**, now on **CPU *and*
GPU** — differential DiT + medium decoder; e2e parity audio 1.7e-2). Done: **E0–E7**
(full `text→audio` **+ continue/inpaint** pipeline — all parity-verified), **E2.9/E2.9b
+ E11.4** (CPU **packed outer-product GEMM ~460–622 GFLOP/s, near MKL** + arena/KV-cache;
the optional BLAS backend was removed as redundant), **E8.1–E8.5e** (CUDA backend +
device-resident DiT *and* medium decoder + on-device cross-K/V + fp16 tensor-core
attention + fused sliding-window decoder attention — **small-music 0.19 s / medium
0.54 s warm**, ≈ stable-audio-tools and 1.8× faster on small), **E9.0–E9.4**
(precision/quant: `--precision` fp32/q8/q4 on CPU+GPU, **fidelity-tuned q4 9.3 %**,
offline `aria-quantize`), **E11** (release polish: install/example/CI/progress/torch-RNG),
**E13.3** (concurrency-clean arenas). Memory: host RSS on the GPU path dropped 7.3 →
0.44 GB (`madvise` after upload). Build + hermetic + parity all green; benchmarks vs
stable-audio-tools in [BENCHMARKS.md](BENCHMARKS.md), full snapshot in [STATUS.md](STATUS.md).
**Since then (2026-07):** E12 residual steering shipped end-to-end — additive **and**
projection ops, CPU+GPU, **graph-resident** (steer kernels captured in the CUDA graph via
device-side effective scales, bf00c9e), `--steer` CLI + `--batch` resident jobs (6d6d76e) +
**`aria-server`** HTTP binary (c238392; warm steered gen **0.185 s over HTTP** on the 3070);
exact-math GPU wins (QK-alpha fold + softmax-f16 fusion, 42962e0: 60 s GPU −10–12 %);
steering-path correctness fixes (d430899). Efficiency methodology re-based (in-process
`--bench` warm on both sides — see BENCHMARKS.md). **Open: E12.2/E12.4 steering arms,
E13.1/E13.2 true concurrency, the E15–E18 forward plan below, and the `v1.0.0` tag.**

## Critical path to M1 (first audio) — ✅ reached

`E1.2 → {E1.3, E1.4, E1.5}` · `E2.1,E2.2,E2.3,E2.4,E2.8` · `E4.1→E4.2→E4.3→E4.4`
· `E5.1→E5.2→E5.3→E5.4` · `E6.1,E6.3 → E6.4` — all done; M1 (e2e) and M2 (T5Gemma in
C) reached. The active front is now **E7** (continue/inpaint), **E9** (precision/
quant), **E8.5c** (GPU decoder), **E10** (medium), **E12** (steering).

---

## E0 — Scaffolding & infrastructure ✅

- ✅ **E0.1** Repo + Makefile (cpu/cuda/test/parity). Verify: `make`, `make test`.
- ✅ **E0.2** Safetensors mmap loader. Verify: loads 685-tensor SA3, `--list-tensors`.
- ✅ **E0.3** WAV I/O (PCM16/24/32 + f32). Verify: `test_wav` round-trip.
- ✅ **E0.4** Op surface + CPU kernels (linear, matmul, rmsnorm, dyn_tanh, silu, gelu_tanh, silu_gate, softmax, elementwise). Verify: `test_ops`.
- ✅ **E0.5** Flat JSON config reader. Verify: reads model_type/sr/channels.
- ✅ **E0.6** Model-module registry + SA3 detect/validate. Verify: `--info`.
- ✅ **E0.7** CLI (info / list-tensors / wav-roundtrip). Verify: runs on cached model.
- ✅ **E0.8** Parity harness (`.atns` dumper + C reader + atol/rtol gate). Verify: `make parity`.

## E1 — Conditioning

- ✅ **E1.1** NumberConditioner `seconds_total` (ExpoFourier f64 + Linear). Verify: `test_number_cond` parity.
- ✅ **E1.2** `∥` **P0** Nested JSON config parser (object navigation: `model.diffusion.config.*`, `model.pretransform.*`, `conditioning.configs[]`). deps: — · Verify: unit test on cached `model_config.json` returns embed_dim=1024, depth=20, num_heads=16, latent 256, downsample 4096.
- ✅ **E1.3** **P0** Timestep features (`expo`) + `to_timestep_embed` MLP. deps: E1.1, E1.2 · Verify: parity of timestep embedding for fixed t (dump from dit.py).
- ✅ **E1.4** **P0** Global adaLN: `to_global_embed` MLP on (timestep+seconds) → `global_cond[1024]`; per-layer `to_scale_shift_gate` add → 6 chunks. deps: E1.2, E1.3 · Verify: parity of global_cond + block-0 modulation params.
- ✅ **E1.5** **P0** Cross-attn cond pack: `to_cond_embed` MLP on concat`[prompt 256×768 | seconds 1×768]` → `[257×1024]` + mask. deps: E1.2 · Verify: parity of cross_attn_cond.
- ✅ **E1.6** **P1** Local-additive inpaint cond hook. Per-block `to_local_embed` MLP (257→1024→1024) loaded; `aria_sa3_dit_req_set_local(local_raw[T,257])` projects + left-pads past the 64 memory tokens, and each block adds it after cross-attention (matches `_apply_local_conditioning`). NULL-safe: plain text→audio is bit-identical (verified, same md5). Active path exercised by `test_local` (synthetic cond shifts the velocity, no NaNs). Full inpaint parity (real mask/masked_input) is E7.3. deps: E1.2.

## E2 — Core ops (shared by DiT / T5Gemma / taae)

- ✅ **E2.1** `∥` **P0** RoPE op (θ-param, per-head q/k rotation). deps: — · Verify: parity vs reference rope on random q + known positions.
- ✅ **E2.2** `∥` **P0** Scaled-dot-product attention (multi-head, non-causal, optional additive mask). deps: — · Verify: parity vs torch SDPA on random q,k,v.
- ✅ **E2.3** **P0** QK-RMSNorm in attention (per-head rmsnorm of q,k with `.gamma[64]`). deps: E2.2 · Verify: parity on self-attn with qk_norm.
- ✅ **E2.4** **P0** Cross-attention variant (kv from context, split `to_q`/`to_kv`). deps: E2.2 · Verify: parity.
- ✅ **E2.5** `∥` **P1** Attn logit softcapping (`tanh(s/50)*50`, T5Gemma). deps: E2.2 · Verify: parity.
- ✅ **E2.6** **P1** Differential attention (q,k base/diff split; `attn(base) − λ·attn(diff)`). deps: E2.2 · Verify: parity (taae + medium DiT). *High-risk — confirm formula from transformer.py.*
- ✅ **E2.7** `∥` **P0** conv1d (weight-normalized; kernel/stride/pad). deps: — · Verify: parity vs torch Conv1d.
- ✅ **E2.8** `∥` **P0** GLU/SiLU-gated FFN helper (`ff.0.proj[2·inner]` → silu-gate → `ff.2`). deps: — · Verify: parity vs a DiT FFN.
- ✅ **E2.9** **P1** Fast GEMM. Done: blocked+omp-simd (6.6× over naive) → **register-blocked AVX2 microkernel** (MR×NR ymm tiles + ILP, cache-blocked, K%8 + edge tails; `make bench` shows 175–280 GFLOP/s vs the loop's ~30–40), dependency-free with a pure-C fallback for non-AVX2. End-to-end generation 17 s → 10.5 s (~1.6×); compute now ~competitive with PyTorch CPU once the one-time weight fault-in is excluded. (Later superseded by the **packed outer-product GEMM** at ~460–622 GFLOP/s — see E11.4 — which made the once-optional `-DARIA_BLAS` backend redundant; it has since been removed to keep the build fully dependency-free.) Verified: `test_ops` + e2e/t5enc parity green.
- ✅ **E2.9b** **P1** Workspace reuse + step-invariant caching. New `aria_arena` (bump allocator, save/restore) replaces the per-block/per-step `malloc`/`free` churn in the DiT hot path (no re-faulting large scratch 160×/gen); `aria_ff_glu`/`aria_attention` take optional caller scratch. Split immutable model (`aria_sa3_dit`, weights) from a per-request context (`aria_sa3_dit_req`: arena + cached `cross_ed`, RoPE tables, `to_global_embed(seconds)`, and **per-block cross-attention K/V** projected once instead of every step). `req_begin`/`step`/`end`; one-shot `aria_sa3_dit_forward` wraps them. End-to-end 11.1 s → 9.1 s (~19%, min-of-N) with much tighter variance; bit-identical output, all parity green. The req/model split is the foundation for the batch API (E11).
- ✅ **E2.9c** **P2** Further CPU GEMM. Tuned the AVX2 register tile **MR×NR → 3×3** (3+1+9 = 13 ymm) via the new `make bench-sweep`: matches/beats 4×3 (which saturates all 16 ymm) with register headroom; bit-identical output (tiling doesn't change the per-element dot). The other items were measured **low-ROI and deferred**: weight-packing would trade the zero-copy mmap weights (halved RSS — a deliberate feature) for ~10–15 %, and the per-head attention is memory-bound (≤10 % of a ~20 % slice). The arena-for-encoder/decoder part already shipped in E13.3. Net: the CPU GEMM is near its practical sweet spot (~220 GFLOP/s); the real speed headroom is the GPU (E8). Verify: `make bench-sweep` + `test_ops`/hermetic green.
- ✅ **E2.10** **P2** Hardware-adaptive backend selection. `--device auto` (the new default) on a CUDA build picks the GPU when it's worth it — a present, **sm_70+** device (`aria_cuda_recommended()`; weaker cards like the GT 1030 sm_61 are slower than the AVX2 CPU) that also fits the model (create returns NULL on OOM → CPU). `--device cuda` forces any device; `--device cpu` never; CPU-only builds treat auto as cpu. deps: E2.9, E8 · Verified: GT 1030 → auto=CPU, cuda=forced; 3070 → auto=GPU.

## E3 — T5Gemma text encoder + tokenizer (parallel track, off critical path)

- ✅ **E3.1** `∥` **P1** Tokenizer: load BPE (tokenizer.json) or SentencePiece (tokenizer.model); encode + Gemma template. deps: — · Verify: token-id parity vs HF tokenizer on prompt set.
- ✅ **E3.2** `∥` **P1** T5Gemma weight load: filter `encoder.*`, name-map 12 layers. deps: E1.2 · Verify: shapes/counts match config.
- ✅ **E3.3** **P1** Encoder forward: embed `·√d_model`, RMSNorm, self-attn (bidirectional + RoPE + softcap + query_pre_attn_scalar 64), GeGLU `gelu_tanh`. deps: E2.1, E2.2, E2.5, E3.2 · Verify: per-layer + final hidden parity vs `T5GemmaEncoderModel`.
- ✅ **E3.4** **P1** Conditioner wrapper → `[1,256,768]` + mask + learned padding embedding. deps: E3.1, E3.3 · Verify: parity vs `T5GemmaConditioner.forward`. **← M2 (with E3.5)**
- ✅ **E3.5** **P0** Precomputed-embedding path (CLI `--prompt-embed <file>`). deps: E1.5 · Verify: generate from dumped embedding == Python (unblocks M1 before E3.3).

## E4 — SA3 DiT forward

- ✅ **E4.1** **P0** DiT weight load + name map (pre/postprocess conv, to_cond/global/timestep embed, 20 layers, memory tokens). deps: E1.2 · Verify: all tensors found, shapes logged.
- ✅ **E4.2** **P0** Input path: `preprocess_conv` + project to embed_dim + prepend 64 memory tokens + RoPE positions. deps: E4.1, E2.7, E2.1 · Verify: pre-block hidden parity.
- ✅ **E4.3** **P0** DiT block forward (one block): adaLN(6) → self-attn(qk-rms,rope)+`σ(1−gate)`+res → cross-attn+res → local-add(NULL-safe) → FFN GLU+gate+res. deps: E2.1–2.4, E2.8, E1.4, E1.5 · Verify: **block-0 parity** vs Python (keystone op test).
- ✅ **E4.4** **P0** Full DiT: 20 blocks + final norm + `postprocess` → velocity `[256,T]`. deps: E4.2, E4.3 · Verify: full `denoiser_forward(x,t,cond)` parity.
- ✅ **E4.5** **P1** Differential attention path for medium (config-gated by `attn_kwargs.differential`). Both self- AND cross-attention: `to_qkv` is `[5·ed,ed]` (q,k,v,q_diff,k_diff), cross `to_q [2·ed]` (q,q_diff), `to_kv [3·ed]` (k,k_diff,v); `out = attn(q,k,v) − attn(q_diff,k_diff,v)` (no λ — medium has no `feat_scale`), RMS qk-norm on base+diff, RoPE on self only, `v` shared. The request caches `cross_kd` (k_diff post k_norm). deps: E2.6, E4.4 · Verify: `test_dit_diff` (synthetic differential block, random weights) parity **7.7e-7**; small-music (non-diff) unchanged (block0 5.5e-4, full 8.3e-4).

## E5 — taae_v2 autoencoder decoder

- ✅ **E5.1** **P0** Decoder weight load + name map (bottleneck, layers, resampling, conv mapping). deps: E1.2 · Verify: tensors found/shaped.
- ✅ **E5.2** **P0** Softnorm bottleneck inverse (running_std/scaling_factor + bias). deps: E5.1 · Verify: parity.
- ✅ **E5.3** **P0** TransformerResamplingBlock (decode): chunked attention (chunk 32, midpoint shift), learnable `new_tokens`, DynamicTanh, differential attn, depth 6. deps: E2.6, E2.7 · Verify: one-block then stack parity. *Highest-risk task.*
- ✅ **E5.4** **P0** Residual upsampler (stride 16) + conv mapping + unpatch (256-sample) → stereo audio. deps: E2.7, E5.3 · Verify: full decoder latent→audio parity vs `pretransform.decode`.

## E6 — Sampler + end-to-end (KEYSTONE)

- ✅ **E6.1** **P0** LogSNR schedule (linspace + LogSNRShift anchor −6.2/end 2.0, t[0]=σ_max). deps: E1.2 · Verify: schedule parity vs `build_schedule`.
- ✅ **E6.2** `∥` **P0** RNG: xoshiro256** + Gaussian. deps: — · Verify: distribution stats; Philox parity later (E11.3).
- ✅ **E6.3** **P0** Pingpong loop `x ← (1−t_next)(x − t·v) + t_next·noise`. deps: E4.4, E6.1 · Verify: with **injected Python noise**, latent parity vs `sample_flow_pingpong`.
- ✅ **E6.4** **P0** End-to-end `text→audio` (precomputed emb + injected noise): cond → init noise → sampler → decoder → WAV. deps: E1.4, E1.5, E3.5, E4.4, E5.4, E6.3 · Verify: C WAV vs Python WAV (waveform MSE under tol). **← M1**
- ⬜ **E6.5** **P2** CFG + `--cfg` CLI flag: cond/uncond two-pass guidance `v = v_uncond + cfg·(v_cond − v_uncond)` for `cfg_scale > 1` (≈2× cost/step; runs the denoiser twice with a null/empty prompt). **Only worthwhile on BASE checkpoints** — the post-trained/distilled small-music is tuned for `cfg = 1.0` (CFG off), so `cfg > 1` there is out-of-distribution and not recommended. deps: E6.3 · Verify: parity vs the `stable_audio_tools` CFG path on a base checkpoint. (`--seed` is already implemented.)

## E7 — continue / inpaint

- ✅ **E7.1** **P1** taae_v2 encoder (audio→latent): patchify (256-sample) → SAME encoder (pad to mult 32, WNConv1d mapping 512→768 k1, group-16 + 1 learned new_token → 17, two chunked S=34 transformer halves [0-2 unshifted, 3-5 midpoint-shift], take last of each 17-group, Linear 768→256) → softnorm fwd (`(x·scaling_factor + bias)/running_std`). The differential-attention block + chunk pass are now shared with the decoder via `aria_taae.{c,h}` (extracted; `test_dec` unchanged at 1.4e-4). Verify: staged `test_enc` parity vs `pretransform.encode` — patchify/softnorm exact, SAME encoder 8.9e-5, full encode 9e-4, zero-pad path 1.8e-2 (0.1% rel, softnorm-amplified). deps: E5.3, E2.7.
- ✅ **E7.2** **P1** Inpaint mask build. `aria_inpaint_mask_latent` nearest-interps an audio-space mask (1=keep, 0=inpaint) to latent length (`mask_lat[t]=mask_audio[(t·audio_len)/T]`, matching torch `F.interpolate(mode='nearest')`); `aria_inpaint_local_cond` builds `local[T,257] = [mask | latent·mask]` (channel 0 = mask, 1..256 = masked_input), matching generation.py. deps: E7.1 · Verify: `test_inpaint` mask/local_cond parity — both **exact (0.0)**.
- ✅ **E7.3** **P1** Local-additive cond live in DiT (E1.6 hook with a real mask). The DiT velocity with a live `local_add_cond` (built by E7.2) matches `dit._forward(..., local_add_cond=real)` at **2.8e-4** (max|ref| 63.7). deps: E1.6, E4.3, E7.2 · Verify: `test_inpaint` dit_velocity parity.
- ✅ **E7.4** **P1** continue/inpaint orchestration + CLI. `aria_gen_params` gains `init_audio` + `inpaint_from_s/to_s` + `inpaint_continue`; `sa3_generate` reads+prepares the clip (channel-major, pad/crop, 44.1 kHz), encodes it (E7.1), builds the keep/regenerate mask + local cond (E7.2), attaches it to the request (E7.3), and runs the normal pingpong+decode (pure-noise start). CLI: `--continue <wav>` (regenerate the tail) / `--inpaint <wav> --from <s> --to <s>` (regenerate a region). Inpaint runs on the CPU **and device** DiT (per-block local-additive cond uploads to the GPU, 48078cb). Verify: `test_inpaint_e2e` end-to-end parity vs the injected-noise PyTorch inpaint — **latent 6.6e-4, audio 2.4e-4**; CLI smoke (base→continue→inpaint) preserves kept regions (continue kept-L1 0.008; inpaint outside-mask 0.004 vs inside-mask 0.014). deps: E7.1–7.3, E6.4. **← M3**

## E8 — CUDA backend (parallelizable track)

- ✅ **E8.1** **P1** `aria_gpu.h` (device alloc/copy/sync + C-callable surface) + Makefile `cuda`/`test_cuda` wiring. nvcc 11.2 via `-ccbin gcc-9` (host gcc≤10), cudart at `/usr/lib/cuda/lib64`, `CUDA_ARCH ?= sm_61` (RTX 3070: `make cuda CUDA_ARCH=sm_86`). `make cuda` builds `libaria.a`+CLI; device detected (GT 1030, sm_61, 2 GB).
- ✅ **E8.2** **P1** CUDA gemm/linear: shared-memory tiled `aria_gemm_nt` (`y = x @ W^T + b`, device-pointer kernel + host wrapper). `make test_cuda` parity vs CPU on the DiT/encoder shapes + edge cases (odd dims, single row): maxdiff ≤ 1.8e-5. *Note:* GT 1030 (2 GB) is **validation-only** — it can't hold the 2.27 GB model and its fp32 won't beat the AVX2 CPU; real GPU speed is the RTX 3070 (sm_86, 8 GB). fp16 (sm_86) compute lands with E8.3/E9.
- ✅ **E8.3** **P1** CUDA hot ops: matmul, rmsnorm, gemma_rmsnorm, dynamic_tanh, softcap, silu, gelu_tanh, silu_gate, softmax (masked), rope, multi-head attention (gemm_nt+scale+softmax+matmul). Correctness-first kernels (norms use fp64 accum to match CPU). `make test_cuda` per-op parity on GT 1030: norms exact (0.0), rest ≤ 1.3e-4.
- ✅ **E8.4** **P1** CUDA conv1d (stride-1, pre-folded weights). Parity ≤ 1.4e-6.
- ✅ **E8.5** **P1** Device-resident DiT (`--device cuda`). The DiT weights upload once as **fp16** (fp32 compute: converted in the GEMM), activations live in a device bump arena, and the denoise loop runs on the GPU — only the per-step latent/velocity cross the bus. Encoder/decoder stay on CPU; per-request caches (cross K/V, RoPE) + per-step `gcond` computed on host and uploaded. Auto-fallback to CPU if no device / insufficient VRAM. Verified on GT 1030: weights ~0.84 GB fit the free VRAM, and GPU audio matches CPU at **0.57% rel-RMS (fp16-weight level, no NaNs)** — proves the orchestration. *Note:* GT 1030 is slower than the AVX2 CPU (weak Pascal + correctness-first kernels); the speedup target is the RTX 3070 (sm_86, 8 GB → F32 weights fit for exact parity). **← M4 (CUDA generation ≈ CPU)**
- ✅ **E8.5b** **P1** GPU perf pass on the RTX 3070 (sm_86), warm per-generation (10 s/8-step): **4.73 s → 1.28 s (3.7×)**, closing the gap to stable-audio-tools (0.36 s warm) from 13× to ~3.6×. Levers: persistent device weights (upload once, reuse — `setup` 6.7 s → 0.09 s warm) + `--bench`/`ARIA_PROFILE`; DiT GEMMs via **cuBLAS GemmEx** (fp16 tensor cores); **batched multi-head attention** (`cublasSgemmStridedBatched`, dit 1.77 s → 0.42 s); **parallel taae decoder** (per-thread arenas, decode 2.34 s → 0.76 s). Audio unchanged (0.6 % rel-RMS, no NaNs). Remaining warm cost: decode 0.76 s (CPU) > dit 0.42 s.
- ✅ **E8.5c** **P1** Device-resident GPU decoder. The taae decoder runs on the GPU, **batching all S=34 chunks of a pass** so the per-token linears become big cuBLAS GEMMs and the per-chunk differential attention runs B·H batched. RTX 3070 warm **decode 0.76 s → 0.02 s**; whole-pipeline warm generation **1.28 s → 0.61 s** — within **~1.7×** of stable-audio-tools (0.36 s), from 13× originally. Audio vs CPU 0.64 % rel-RMS, no NaNs (also verified on the GT 1030). Remaining micro-opts (DiT glue-kernel fusion, fewer fp16 conversions, trimming the 0.16 s warm setup) are now small vs the 0.42 s DiT. deps: E8.5b.
- ✅ **E8.5d** **P1** Profile-guided GPU kernels. `nsys` showed the naive **thread-per-row** norms dominated — `k_rmsnorm` alone was **51 %** of GPU time (and fp64-bound at 1/64 rate on Ampere), `k_softmax` 16 %. Rewrote both as **one-block-per-row fp32 shared-memory reductions** (`rsqrtf`). RTX 3070 warm **DiT 0.42 s → 0.16 s**, whole-pipeline generation **0.61 s → 0.29 s — now on par with / ahead of stable-audio-tools (0.36 s warm)**; aria's DiT (0.16 s / 8 steps) is ~2× its diffusion. fp32 norms within the fp16 tolerance (test_cuda 2.4e-7; audio vs CPU 0.66 % rel-RMS, no NaNs). **The 13× PyTorch gap is closed.**
- 🟡 **E8.5e** **P2** GPU micro-opts. **Done:** **on-device cross-K/V projection** (`ca_to_kv` device-resident; medium setup 0.33 → **0.02 s**) · **fp16 tensor-core attention** (the QK/AV ran as fp32 `sgemm`, 21 % of GPU time → `cublasGemmStridedBatchedEx`; small-music DiT attention 0.29 → **0.19 s**) · **fused sliding-window decoder attention** (warp-per-query O(N·35) band kernel; medium decode 0.23 → **0.13 s**). Net: small-music **0.29 → 0.19 s** (1.8× SAT), medium **0.98 → 0.54 s** (≈ SAT). **Flash-style DiT attention: assessed and declined** — at S≈172 the score buffer is L2-resident, so a hand-fused kernel gives no memory win and rarely beats cuBLAS+softmax. **Remaining (~1 %, low-ROI):** fuse the DiT glue kernels (adaLN/gate/add) + a cublasLt bias epilogue. deps: E8.5d · Verify: warm small-music 0.19 s / medium 0.54 s, parity unchanged ✓.
- ⬜ **E8.6** **P2** SSD weight streaming for VRAM-exceeding components (GT 1030). deps: E8.5 · Verify: medium runs on 2 GB.

## E9 — Precision & Quantization

- 🟡 **E9.0** **P1** Unified precision/dtype selection. `aria_dtype {fp32,fp16,bf16,q8,q4}` + the **dispatch seam** landed: `aria_qweight` (a Linear weight that is f32-borrowed or Q8/Q4-owned) + `aria_linear_qw`, plus `--precision` (CLI) / `aria_gen_params.precision`. The DiT block GEMMs (sa_qkv/out, ca_q/kv/out, ff) dispatch across precisions via an **additive overlay** (`aria_sa3_dit_quantize`) — the f32 path and the CUDA view are untouched (f32 stays the zero-copy mmap path, bit-identical; verified by `test_dit_full` 8.3e-4 + `test_quant_dit` fp32 round-trip = 0). q8/q4 force the CPU DiT. **Remaining:** fp16/bf16 as distinct CPU storage (currently fall back to fp32) + GPU precision selection (E9.4). deps: E2.9, E8.5 · Verify: `test_quant_dit` (q8 3.1 % velocity err, q4 36.8 %); `--precision` end-to-end runs. **Note:** CPU quant is a *footprint* win, not speed — the scalar dequant GEMM is ~6–8× slower than the AVX2 f32 microkernel; the speed win is the GPU (E9.4).
- 🟡 **E9.1** `∥` **P1** Q8 (int8 per-row) pack/unpack + CPU dequant-on-use gemm. **Primitive done** (`aria_quant.{c,h}`): `aria_q8_quant` (per-row symmetric, scale=max\|·\|/127) + `aria_linear_q8` (dequant-on-use, scale factors out per row). Hermetic `test_quant`: round-trip within half-step, GEMM relerr 0.6–1.0 % on the SA3 shapes (4× smaller than f32). **Wired** into the DiT (E9.0 overlay): `--precision q8` quantizes the block GEMMs on first use; `test_quant_dit` gates the model-level effect at **3.1 % velocity rel-RMS** (4.0× smaller block weights, 1.6 GB → 401 MB), and the f32 round-trip is bit-identical. deps: E9.0.
- ✅ **E9.2** **P1** `aria-quantize` offline tool (`make quantize` → `./aria-quantize <model_dir> <out.aria> [q8|q4]`). Reads `model.safetensors`, quantizes the DiT per-step block GEMMs (q4 = asym int4 FFN + Q8 attention) and writes a packed `.aria` overlay (`aria_qweight_write`/`_read`, magic `ARIAQNT1`). `aria --load-quant <file.aria>` loads it (CPU), skipping on-the-fly quantization. Verify: serialization round-trip **exact** (hermetic `test_quant`), and `--load-quant` generation is **bit-identical** to `--precision q4` (md5 match on the real model). deps: E9.1.
- ✅ **E9.3** **P1** Q4 (block) pack + dequant-on-use, **fidelity-tuned**. A parallel workflow swept 6 Q4 strategies vs `test_quant_dit`; the winner is **asymmetric (zero-point) int4** — per-block `[min, scale]`, `scale=(max−min)/15`, nibble in `[0,15]`, `W≈min+nib·scale` — **plus mixed precision**: the four attention projections (which feed qk-rmsnorm + softmax, where error is amplified) stay **Q8**, only the FFN goes Q4. Together this takes DiT velocity rel-RMS from **29.8 % → 9.32 %** (3.2× better, near q8's 2.45 %) at 300 MB (4.79× vs fp32, still < q8's 361 MB). Findings that lost: uniform asym 23.4 %, mixed-attn-only 11.5 %, block16 24 %. `test_quant` (GEMM relerr) + `test_quant_dit` (9.32 %, gate 13 %) green; fp32 round-trip bit-identical; GPU q4 matches CPU q4 (1.06 % rel-RMS, fp16-dequant level). deps: E9.1.
- ✅ **E9.4** **P1** CUDA dequant-on-use kernels (Q8/Q4). The device DiT stores the block GEMM weights **packed in VRAM** (`dqw`: q8/q4 + scales) and dequantizes each into a reused fp16 scratch (`k_dequant_q8/q4`) before the tensor-core cuBLAS GEMM — so resident VRAM shrinks (RTX 3070 small-music: fp16 **1464 MB → q8 1142 MB → q4 1080 MB**; q4 is the smallest at near-q8 fidelity), which is what lets medium fit. `--precision q8|q4` now runs on the GPU (host-quantizes the f32 view, uploads packed; host cross-K/V uses the CPU overlay, so it stays consistent). deps: E8.2, E9.3 · Verified on the GT 1030: compiles (nvcc 11.2), output **finite and numerically identical to the CPU quant path** (q8/q4 RMS match to 5 dp), `test_cuda` op-parity unchanged. **Note:** the GT 1030 (no tensor cores) is slightly *slower* at q8/q4 (dequant overhead, fp16 falls back to fp32); the speed win + the medium-fits payoff want the **RTX 3070** (sm_86) — bench there. **← M5 (mechanism; the medium demo lands with E10)**

## E10 — medium model

- ✅ **E10.1** **P1** Load medium config/weights (embed 1536, depth 24, heads 24) + SAME-L decoder. deps: E4.4, E5.4 · Verify: tensors/shapes. **Partial:** aria already parses the medium config and loads/runs it end-to-end on the GPU (the loader is dim-driven; q8/q4 quantize + upload + decode all handle medium's scale — verified on the RTX 3070: fp16 2.84 GB → **q4 1.91 GB**, fits comfortably). The **medium DiT is now correct** (E4.5 differential attention + the `differential` bool-parse fix): full 24-block medium denoiser parity vs PyTorch **0.108** (threshold 0.23) — medium produces correct **latents**. **Remaining for medium audio:** the medium **decoder** differs substantially from small-music's — taae_v2 with transformer_dim **1536** (vs 768), depth **12** (vs 6), inner 4608, **`sinusoidal_blocks: 8`** (sinusoidal FF in the later blocks), and **`sliding_window: [1,1]`** with `chunk_size None` (a different chunk/attention path vs small-music's `chunk_size 32` midpoint-shift). Norm is still DyT (alpha/gamma/beta). **Done** (E10.2): new `aria_sa3_dec_medium` + `taae_med_block_forward` (runtime dim, sliding-window band mask, SiLU/Sin GLU FF). Block parity 2.4e-7/5.4e-7, full decode 3.4e-6, e2e (DiT+sampler+decode) latent 7.3e-3 / audio 1.7e-2. The orchestrator detects medium (decoder.layers.1.weight `[1536,256]`) and runs it on the CPU **or GPU** — the device-resident `aria_cuda_dec_medium` (fused warp-per-query band attention, sin/SiLU FF) landed in E8.5e, so medium runs **fully on the GPU at 0.54 s** (≈ stable-audio-tools), GPU-vs-CPU audio ~1.2 %.
- ✅ **E10.2** **P1** Medium end-to-end (differential DiT attn + medium decoder). `aria -m <medium>` runs the full pipeline on the CPU: medium differential DiT → 8-step pingpong → medium taae_v2 decoder (sliding-window banded attention + sinusoidal FF) → stereo. Verify: `test_e2e_medium` (injected-noise, vs PyTorch) latent **7.3e-3** / audio **1.7e-2**; CLI uncond produces finite, in-range audio (CPU ~11.5 s / 3 s clip: DiT 9.1 s + decode 2.0 s). deps: E10.1, E4.5 · **← M6**. **Full GPU medium done**: the differential DiT (9.1 s → **0.36 s**, 26×) *and* the sliding-window/sinusoidal decoder (2.0 s → **0.05 s**, 40×) are device-resident — RTX 3070 medium 11.4 s → **2.2 s** (5.2×), GPU-vs-CPU audio 0.76 %, robust to 10 s clips. **medium q4/q8** quantize on the GPU (fp16 3.95 GB → **q4 2.84 GB**). The GPU decoder uses a banded `k_softmax_band` (window ±17) + `k_ff_singate` sinusoidal gate.

## E11 — Release polish (v1.0.0)

- ✅ **E11.1** **P1** Library API finalize + `make install` + public headers. `make install` lays down `aria` + `libaria.a` + the public headers (`aria.h`/`aria_wav.h`/`aria_quant.h`) under `PREFIX`; `examples/generate.c` + `make example` is an external program that links `libaria.a` through the public headers only and runs end-to-end (verified). Verify: external program links `libaria.a` ✓.
- ✅ **E11.2** **P1** CLI UX: per-step **progress** (`aria_pingpong_cb` + `gen_params.progress`), drawn in-place on stderr only when it's a TTY (pipes/logs stay clean); `--device`, `--seed`, `--rng` flags. Skipped `--cfg`: CFG is unimplemented and cfg>1 is OOD for the post-trained 8-step checkpoints, so a flag would be a misleading no-op (the schedule is model-fixed). Verify: TTY/non-TTY run matrix ✓.
- ✅ **E11.3** **P2** Torch-matched RNG option + determinism. `--rng torch` ports PyTorch's CPU `at::mt19937` + the float `normal_fill` (block-of-16 Box-Muller), so `aria_rng_randn(TORCH)` reproduces `torch.manual_seed(s); torch.randn(n)`. Measured over 22k draws (seeds 0/9/42/7/123, n 64/256/4096): **max abs diff ≤2e-6, ~60% bit-identical, the rest ≤1 ULP** — the residual is glibc-vs-torch libm rounding of `cosf/sinf/logf`, not the algorithm. The hermetic `test_rng` locks it against checked-in `torch.randn` goldens (≤2e-5). deps: E6.2 · Verify: bit-match vs torch.randn ✓.
- ✅ **E11.4** **P2** CPU perf pass. **SIMD RMSNorm/GemmaRMSNorm** (`__m256d` sum-of-squares + vectorized multiply; DiT 7.36 → 6.58 s) and — the big one — a **packed outer-product GEMM**: the dot-product microkernel was load-bound at its ~235 GFLOP/s ceiling, so `aria_linear_packed` packs A into k-major MR-panels once + each NR-panel of B on the fly and runs a 6×16 outer-product microkernel (no hsum, contiguous reads). Big DiT GEMMs **207 → ~600 GFLOP/s** (near MKL's ~600–700). With **thread-local persistent pack buffers** (no per-call malloc), **K-blocking** for large K (the `NR×K` B-panel overflows L2 at K≥4096; `ff_out` 304 → 474), and **head-parallel attention** (was serial-over-heads with each tiny K=64 head GEMM spawning its own thread pool): **CPU small-music 6.7 → 3.5 s, medium 17 → 8.1 s** (20-core), closing the gap to PyTorch/MKL from 2.4× to **1.15–1.25×**. Full parity green (36/36; sequential-k accumulation within tolerance). deps: E2.9, E2.9b · Verify: `make bench` + `--bench` warm run + parity green ✓. _Residual gap to MKL is its assembly-tuned GEMM edge + memory-bound non-GEMM kernels — diminishing returns._
- ✅ **E11.4d** **P2** SIMD elementwise/norm — **RMSNorm/GemmaRMSNorm vectorized** (`__m256d` double-precision sum-of-squares + AVX2 multiply; see E11.4). DynamicTanh/SiLU/GELU/masked-softmax were measured at **~1 %** of DiT (GEMM-dominated): a verified 1e-7 AVX2 `expf` for SiLU/softmax moved nothing measurable and was reverted, so they stay scalar.
- 🟡 **E11.4c / E11.4e / E11.4f** **P2** Partial. **Done:** thread-local pack buffers (no per-call malloc), decoder arena scoping (attention scratch freed before the FF — also a ~100 MB memory win), and head-parallel attention (was serial-over-heads spawning a thread pool per tiny K=64 head GEMM). **Not done, now low-ROI:** pingpong in-place `denoised`, a `head_dim=64` SDPA specialization, the SAME-decoder chunk pad/copy cleanup.
- ⬜ **E11.4a / E11.4b / E11.4g** **P3** Deferred — **measured GEMM-bound, ~1 % each** (demoted from P1): token-major latent layout (`CT↔TC` transposes are ~6 ms/gen), attention layout-churn / `extract/merge_heads` reduction, and the cold-path tensor-lookup index. The packed GEMM put the matmuls near MKL, so the remaining non-GEMM data movement is small. Revisit only if a profile shifts.
- ✅ **E11.5** **P1** Docs: README usage current (M1–M6, all flags), AGENTS.md repo layout + CONTRIBUTING.md parity list refreshed (real filenames, no stale `[planned]`, full `PARITY_TESTS` + `test_cuda`). Model-prep is documented in the README (download_model.sh → export_tokenizer.py → `--info`). Verify: a fresh user can build + generate ✓.
- ✅ **E11.6** **P2** CI (`make test` on push). `.github/workflows/test.yml` runs `make` + `make test` + `make example` on push/PR (ubuntu-latest, CPU-only, hermetic — no model/GPU/network). Verify: hermetic suite green locally; CI green on first run.
- 🔶 **E11.7** **P1** **v1.0.0 checklist** — validated, pending the release tag + (optional) E12 steering. Confirmed green this pass: **small-music + medium** both generate · **text→audio + continue + inpaint** (parity: `test_e2e`, `test_inpaint_e2e`) · **CPU + CUDA** (small-music CPU↔GPU 0.47 %, medium GPU 0.76 %, both finite) · **Q8/Q4** (q8 2.45 % / q4 9.32 % velocity, GPU VRAM cuts) · **all parity green** (36/36 small-music vs PyTorch after the E11 changes) · **docs** current. Outstanding: steering hooks (**E12**, deferred — research driver, recommended not required) and the actual `v1.0.0` git tag (release owner's call). **← v1.0.0**

## E12 — Steering (TasteSteer: training-free activation/representation steering)

Why: this runtime is the efficient inference vehicle for taste steering of SA3
(SAME-latent vs DiT-residual, steering-vs-LoRA, training-free core). The C runtime
exposes every intermediate tensor, so steering is an elementwise `x += α·d` at a
known op boundary plus a spec struct. **Can start as soon as M1 lands** and runs
in parallel with E7/E8/E9. Directions are `.atns` vectors; the parity harness
doubles as the activation-extraction path.

- ✅ **E12.1** **P1** Steering spec + dispatch: `aria_steer` struct (`site`, `layer`, `dir`, `scale`, `step_lo/hi`) + a steer set threaded through `aria_generate`; orchestrator applies steers at registered hooks. deps: E6.4 · Verify: empty/scale-0 steer set ⇒ bitwise-identical output vs unsteered; non-zero scale changes output deterministically. **Done** — `aria_steer.h` (kind-agnostic); step counted in `sa3_denoise`; `tests/steer_verify.sh` GATE1 (scale-0 byte-identical) + GATE2 (deterministic effect) pass.
- ⬜ **E12.2** **P1** Diffusion-latent steering (256-D SAME arm) hook in the sampler loop, with step-window gating. deps: E12.1, E6.3 · Verify: unit test — injecting known `d` adds exactly `α·d` to `x` only within `[step_lo,step_hi]`; α=0 ≡ unsteered.
- ✅ **E12.3** **P1** DiT residual-stream steering (per-layer 1024-D arm) hook after the chosen layer's residual add. deps: E12.1, E4.4 · Verify: steering layer L perturbs activations from L onward only; per-layer add matches expected; α=0 ≡ unsteered. **Done** — hook at the `dit_block_core` block output (broadcast over tokens, layer+step-window gated, matches sf-api `AdditiveInjector`); `steer_verify.sh` GATE3 `steer(2α,d)≡steer(α,2d)` byte-identical proves the op is exactly `scale·dir`. **CPU + GPU** — device hook in `dit_block_dev`. **Graph-resident since bf00c9e:** the steer kernels are recorded INTO the CUDA graph and read a device-side effective scale (0 out-of-window → bit-exact no-op) refreshed per step like dx/gcond; steered graph-vs-inline output byte-identical (full + windowed). A **projection/ablation op** (`ARIA_STEER_PROJECT`, Arditi/ds4-style, c6e5f95) exists alongside additive; 5 gates + graph/inline A/B pass. Steering adds **no measurable GPU overhead** (steered == unsteered 0.16 s warm).
- ⬜ **E12.4** **P1** Conditioning-space steering (global adaLN 1024-D + cross-attn cond tokens) hook. deps: E12.1, E1.4, E1.5 · Verify: direction on `global_cond` shifts modulation params deterministically; α=0 ≡ unsteered.
- ⬜ **E12.5** `∥` **P2** Pre-decode latent steering hook (output-space nudge). deps: E12.1, E5.4 · Verify: direction added to latent pre-decode; α=0 ≡ unsteered.
- ⬜ **E12.6** **P1** `.atns` activation extraction at each site (mean-pooled taps) so contrastive directions (e.g. sweet − neutral) can be built from C runs. deps: E12.2–E12.4 · Verify: tapped activation matches the site's parity dump; contrastive direction is reproducible across runs.
- ✅ **E12.7** **P1** CLI `--steer site:layer:dir.atns:scale:lo-hi` (repeatable) + direction loader. deps: E12.1 · Verify: parses multiple steers; produces measurably different, logged output. **← M7** **Done** — table-driven `--steer` (repeatable, `.atns` via `aria_parity_load`); `tests/steer_compare.sh` + `sa3-sf-api/experiments/aria_steer_compare.py` stand up the aria↔sf-api comparison (same `norm·unit` direction → both steer; aria ~1.2× faster CPU @ 4 s/8 steps).
- ✅ **E12.8** `∥` **P2** Steering validation sweep — done via the **sf-api campaign** (aria reproduces the SA3 dense window: 5 axes × 7 α × 12 prompts × 3 seeds, scored wav2taste+CLAP+FAD; dose-response parity vs PyTorch **r=0.95 small / 0.92 medium**, per-clip r=0.79/0.64; Wilcoxon p<1e-8 at peak α). Drivers live in `sa3-sf-api/experiments/` (aria_window_sweep, overlay_parity, efficiency_compare).
- ⬜ **E12.9** **P1** (paper-blocking) LoRA runtime-adapter arm, sd.cpp `WeightAdapter` pattern: `aria_linear_lora(x,W,down,up,scale)` = base GEMM + `scale·(x·downᵀ)·upᵀ` at runtime (keeps W quantized/mmap'd; merge mode optional later), safetensors adapter loader by name-prefix, device-resident `scale` so the CUDA graph stays captured (same trick as steering). Standard LoRA only (skip LoHa/LoKr/DoRA). deps: E2.x, E4.4 · Verify: adapter matches a Python LoRA forward within tol; zero-rank ≡ base; unblocks the paper's steering-vs-LoRA table (`tab:cost`).

## E13 — Batch / server (throughput)

Foundation laid in E2.9b: the immutable model (`aria_sa3_dit`, read-only weights)
is split from per-request scratch (`aria_sa3_dit_req`, owns its arena + caches),
so distinct requests share one loaded model with no shared mutable state.

**Shipped (serialized amortization — NOT E13.1/E13.2):**
- ✅ **E13.0a** `--batch <jobs.tsv>` (6d6d76e): many jobs against one resident ctx
  (amortizes model open, GPU upload/quantize, per-prompt T5 encode). ~2× sweep
  throughput vs one-shot CLI (0.60 vs 1.23 s/job); byte-identical to one-shot.
- ✅ **E13.0b** `aria-server` (c238392, ds4-server pattern): resident HTTP binary,
  thread-per-connection → job queue → ONE worker owning the ctx. `GET /health`,
  `GET /info`, `POST /generate` (JSON in, WAV out, optional steer). Verified on the
  3070: warm steered gen **0.185 s over HTTP**, server==CLI byte-identical,
  concurrent requests serialized cleanly. (The commit subject says "E13.2" — a
  mislabel; the true batch dim below is still open.)

Two complementary strategies remain:

- ⬜ **E13.1** **P1** *Request-parallel (multi-stream)* — a worker pool runs N
  independent `aria_sa3_dit_req` generations against one shared model. Correct
  **today** for the DiT (model is immutable, reqs are independent); deliverable =
  a server/CLI harness + a per-request thread budget (each step uses OpenMP, so
  choose: few requests × all cores for latency, or many requests × `OMP=1` for
  throughput). deps: E2.9b, E13.3 · Verify: K concurrent generations match K
  sequential ones bit-for-bit; aggregate throughput scales with cores.
- ⬜ **E13.2** **P2** *True batched forward (batch dim B)* — one req processes B
  same-duration latents together so each weight streams from memory once for all
  B (these GEMMs are memory-bound ⇒ the big throughput win). Plumb a batch dim
  through attention (block-diagonal per sample) and conditioning (per-sample
  `cross_ed` / cross-KV / `global`, shared RoPE): `aria_sa3_dit_req_begin_batch(m,
  T, cross[B], n_cond, global[B], B)` → `aria_sa3_dit_step_batch`. Ragged
  durations pad to max-T or fall back to E13.1. deps: E13.1, E4.4 · Verify: a
  B-batch matches B singletons within parity tol; throughput/sample beats E13.1
  at the same core count.
- ✅ **E13.3** **P1** Extend the arena to the encoder/decoder forwards. taae decoder
  block (`taae_block_forward`, the ~650-call/gen hot path) now draws all scratch
  from a save/restore arena instead of 12 `malloc`/`free`s per call; the T5Gemma
  encoder uses one allocate-once arena for its whole forward. Whole pipeline is now
  per-request-scratch with no hot-path heap churn. Bit-identical output; parity
  green (test_dec / test_t5enc / test_e2e). ~12% faster end-to-end (interleaved A/B).

## E14 — Streaming / interactive (shipped, previously untracked)

The `--stream` stack shipped without a roadmap entry: continuous sliding-window
generation (`--stream/--chunk/--context/--chunks/--hold`), post-hoc continuation
(context 6 s + skip + emit ~2 s + tail per window), phase-aligned crossfade, HPSS
hold mode, raw-f32 stdout piping with a writer thread; GPU continuation; RTF
0.6–4× on the 3070. Documented in BENCHMARKS.md/PAPER.md. **Open improvements**
(2026-07 review):

- ⬜ **E14.1** **P1** Partial (emit+halo) decode: decode only the emitted region +
  halo instead of the full ~12.5 s window each chunk (~6× less decode work/chunk;
  parity-safe with the banded decoder). Largely subsumed by E16.1 if that lands
  first. Verify: chunk output byte-parity vs full-window decode.
- ⬜ **E14.2** **P2** Keep-region-only re-encode: the taae encoder re-runs over the
  full padded window every chunk (CPU even in GPU mode); encode only the region
  the continuation actually needs. Verify: identical latents on the kept region.
- ⬜ **E14.3** **P2** Latent-domain continuation: carry the context as latents
  instead of decode→re-encode round-trips (removes the encoder from the loop,
  enables DiT/decode overlap). Design task — changes continuation quality
  characteristics; A/B against the current path.
- ⬜ **E14.4** **P2** Stream hygiene: sample-rate/channel handshake on the raw-f32
  stdout stream + underrun detection/pre-roll for RTF<1 CPU streaming.
- ⬜ **E14.5** **P3** Instant preview tier (sd.cpp latent-preview analog): a fitted
  `[256→k]` linear latent→envelope/mel projection for a monitor signal while the
  real decode runs a window behind; optional distilled tiny decoder later.

## E15 — Edge / Raspberry Pi 5 (the paper's edge claim)

The intro claims SA3 fits a Pi 5-class device; nothing has been measured on ARM.
**Blocker first:** aria_cpu.c does not even compile without AVX2.

- ⬜ **E15.1** **P0** Non-AVX2 build fix: `aria_rmsnorm`/`aria_gemma_rmsnorm` (and
  friends) use `__m256`/AVX2-only helpers with no scalar guard, and the Makefile
  hardcodes `-mavx2 -mfma`. Add scalar fallbacks + a `CPU_ARCH` Makefile knob.
  Verify: `make` succeeds on ARM (or x86 with `-mno-avx2`) and `make test` passes.
- ⬜ **E15.2** **P1** NEON kernels: port the 3 hot paths behind `__ARM_NEON` —
  (1) the 6×16 packed outer-product GEMM microkernel → `float32x4_t` tiles (the
  packing layer is ISA-agnostic, reuse as-is), (2) `aria_linear_q8/q4` dequant-GEMM
  (NEON int8 widening + `vfmaq`), (3) attention inner loops + rmsnorm. Scalar path
  stays as fallback. Verify: parity vs scalar; GEMM GFLOP/s benchmark on Pi 5.
- ⬜ **E15.3** **P2** Params-on-disk residency (`--params-disk`): per-DiT-block
  fault-in from mmap + `MADV_DONTNEED` after use → peak RAM = one block + arena
  (medium on an 8 GB Pi). aria already has both primitives. Verify: medium
  generates under a hard RSS cap; unchanged output.
- ⬜ **E15.4** **P2** Arena measure mode: dry-run the op sequence counting
  allocations (the `peak` field already exists), then allocate exactly peak —
  replaces static worst-case sizing; per-(seq_len, model) cache. Verify: RSS drop;
  no mid-run growth.
- ⬜ **E15.5** **P1** Pi 5 measurement for the paper: small (+ medium q4) 10 s gen
  wall-clock + RSS on a Raspberry Pi 5 8 GB; replaces the paper's
  `\tbd{measured Pi 5 latency}`. deps: E15.1 (+E15.2 for a usable number).

## E16 — Long-form decode (the 60 s gap; sd.cpp import)

aria trails stable-audio-tools at 60 s GPU (small 1.46×, medium 1.38× after
42962e0) and the decoder is the bound. The highest-leverage import from
stable-diffusion.cpp is its **rolling per-layer feature-cache decode**
(`wan_vae.hpp`: carry the last K frames of each layer's activations across
chunk boundaries → seam-free chunked decode with O(window) memory).

- ✅ **E16.1** **P1** Bounded-memory windowed decode (v1 = stateless halo-recompute,
  CPU): each window decodes `[start−H, end+H)` and keeps `[start, end)`. Derived
  halos: small **H=2** latent frames (two midpoint-shifted chunk passes compose to
  ±2), medium **H=12** (12 blocks × ±17-token band = ±204 tokens = ±12 frames; rope
  regenerated at each window's absolute offset). Auto above one window on the CPU
  path; `ARIA_DEC_WINDOW=<seconds>` overrides (default 8 s, 0 = off). **Byte-identical**
  to monolithic (both models, e2e cmp); 60 s medium decode peak RSS **−1.06 GB (−36 %)**.
  Windowed-vs-monolithic gates in test_dec / test_dec_medium. Follow-ups: **E16.1b**
  GPU windowed decode; **E16.1c** stateful feature-cache (sd.cpp wan_vae, saves the
  2H recompute) + streaming/continue paths honoring `ARIA_DEC_WINDOW`.
- ⬜ **E16.2** **P1** Medium GPU decoder glue fusion: `med_block_dev` still runs
  5×extract + 4×dyt + 4×rope as separate launches per block (the DiT got this
  fusion pass in E8.5d/e; the decoder didn't). Fuse into the extract/norm kernels.
  Verify: byte-identical decode; 60 s medium GPU timing.
- ⬜ **E16.3** **P3** Smootherstep crossfade + auto window sizing (sd.cpp
  `sd_tensor_merge_2d` / `get_tile_sizes` analogs) for stream joins and E16.1
  window policy.

## E17 — GPU perf backlog (consolidated, post-42962e0)

- ⬜ **E17.1** **P1** Producer kernels emit fp16: `k_rmsnorm_adaln`, `k_ff_silugate`,
  `k_merge_heads`, `k_extract_normrope` write fp32 that the next GEMM/attention
  re-converts (`k_f32_to_f16`, ~6 % of GPU time). Emit `__half` directly where the
  consumer is fp16. Verify: byte-identical (RTNE preserved) or documented tol.
- ⬜ **E17.2** **P2** On-device pingpong sampler: the per-step latent D2H→host
  pingpong→H2D round-trip + host RNG can move device-side (host keeps the
  schedule); removes 2 transfers/step and the host sync. Verify: parity vs host
  sampler (xoshiro sequence preserved).
- ⬜ **E17.3** **P2** Q8/Q4 resident dequant cache: quantized weights re-dequantize
  into the shared scratch on EVERY GEMM every step; cache the fp16 dequant per
  weight when VRAM allows (or per-block ring). Verify: q4 timing ≈ fp16 timing.
- ⬜ **E17.4** **P3** Pinned host staging for per-step latent/gcond + decoder output
  (pageable async currently degrades to sync copies).
- ⬜ **E17.5** **P3** cublasLt bias epilogue (drop the separate `k_add_bias` pass).

## E18 — Quantization v2 (sd.cpp-informed)

- ⬜ **E18.1** **P1** Generalized per-tensor recipe table: replace the hardcoded
  FFN-q4/attn-q8 split with a name-pattern→{q4,q8,f16} table + the auto rule
  "biases/norms/embeddings/first+last blocks stay high-precision"
  (sd.cpp `tensor_should_be_converted` / `--tensor-type-rules` analog). Pure
  metadata, no new kernels. Verify: `test_quant_dit` error ≤ current 9.3 %.
- ⬜ **E18.2** **P2** Q6_K-style superblock: group 8 of aria's 32-wide q4 blocks
  under a shared 16-bit super-scale (quantize the scales) → q4 footprint at
  materially lower error. CPU + CUDA dequant. Verify: velocity rel-RMS vs q8/q4.
- ⬜ **E18.3** **P3** imatrix-style calibrated quantization in `aria-quantize`
  (activation-importance pass over a few prompts → importance-weighted scales /
  per-row q8 promotion).

## E19 — Tests & docs debt (2026-07 audit)

- ⬜ **E19.1** **P1** Hermetic tests for the new surface: `aria_wav_to_mem` ==
  `aria_wav_write` bytes; `--batch` TSV parsing (incl. bad-spec line isolation);
  `read_request`/`parse_steer_json` unit harness; a GPU PROJECT dim-mismatch gate
  (`st->dim > dim`) in steer_verify.
- ⬜ **E19.2** **P1** STATUS.md refresh: it still marks E11/E12 ⬜ wholesale and
  lists medium/steering/aria-quantize as "not implemented" — all shipped. Bring it
  in line with this roadmap (or shrink it to a pointer at ROADMAP.md).
- ⬜ **E19.3** **P2** README/PAPER.md: document `--batch`, `aria-server`, the
  graph-resident steering, and the corrected efficiency methodology (warm =
  in-process `--bench`, both sides).

## Post-1.0 (north star)

- **AceStep 1.5 module** — add `aria_dit_acestep.c` (+ its AE) behind the
  `aria_model_module` seam; success = **zero** changes to kernels, sampler,
  quantization, or I/O. This is the proof that the modular design holds.

## Risk register (watch these)

- **E5.3 taae resampling** — chunked-halo attention + variable stride + learnable
  new_tokens + DynamicTanh + differential attn + softnorm. Highest risk; port
  decode first with aggressive per-op parity.
- **E2.6 differential attention** — get the exact base/diff/λ formula from
  `transformer.py` before relying on it (taae + medium).
- **E3.1 tokenizer** — BPE vs SentencePiece for Gemma; validate token IDs against
  HF before trusting downstream.
- **End-to-end RNG parity** — pingpong injects fresh noise per step; exact repro
  needs Philox (E11.3). Use injected noise for math parity meanwhile.
