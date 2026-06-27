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
| **M5** 🟡 | Quantized (Q4) — medium fits low VRAM | E9.4 |  ← mechanism done (CPU+GPU Q8/Q4); q4 fidelity tuning + medium (E10) remain |
| **M6** | medium model end-to-end | E10.2 |
| **M7** | Steering (TasteSteer): latent + DiT-residual + cond-space injection, training-free | E12.7 |
| **v1.0.0** | Release checklist green | E11.7 |

## Status snapshot

Milestones **M1, M2, M3, M4** reached. Done: **E0–E7** (full `text→audio` **+
continue/inpaint** pipeline: conditioning, DiT, taae decoder **+ encoder**, T5Gemma
encoder + tokenizer, sampler, inpaint local-cond, end-to-end — all parity-verified),
**E2.9/E2.9b** (CPU AVX2 GEMM + arena/KV-cache), **E8.1–E8.5d** (CUDA backend +
device-resident DiT/decoder + profile-guided kernels: 3070 warm 0.29 s), **E13.3**
(concurrency-clean arenas). Build + hermetic + parity all green. Full snapshot in
[STATUS.md](STATUS.md). Open: precision/quant (E9), medium (E10), release polish
(E11), steering (E12), batch APIs (E13.1/2), GPU inpaint local-cond (E8.5e-adjacent).

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
- ✅ **E2.9** **P1** Fast GEMM. Done: blocked+omp-simd (6.6× over naive) → **register-blocked AVX2 microkernel** (MR×NR ymm tiles + ILP, cache-blocked, K%8 + edge tails; `make bench` shows 175–280 GFLOP/s vs the loop's ~30–40), dependency-free with a pure-C fallback for non-AVX2. End-to-end generation 17 s → 10.5 s (~1.6×); compute now ~competitive with PyTorch CPU once the one-time weight fault-in is excluded. Optional `-DARIA_BLAS` backend kept (dimension-gated; pthread-OpenBLAS oversubscribes vs OpenMP, so prefer the OpenMP build or MKL/Accelerate). Verified: `test_ops` + e2e/t5enc parity green.
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
- ⬜ **E4.5** **P1** Differential attention path for medium (config-gated). deps: E2.6, E4.4 · Verify: medium block parity.

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
- ✅ **E7.4** **P1** continue/inpaint orchestration + CLI. `aria_gen_params` gains `init_audio` + `inpaint_from_s/to_s` + `inpaint_continue`; `sa3_generate` reads+prepares the clip (channel-major, pad/crop, 44.1 kHz), encodes it (E7.1), builds the keep/regenerate mask + local cond (E7.2), attaches it to the request (E7.3), and runs the normal pingpong+decode (pure-noise start). CLI: `--continue <wav>` (regenerate the tail) / `--inpaint <wav> --from <s> --to <s>` (regenerate a region). Inpaint runs on the CPU DiT (the device DiT has no local-cond path yet — follow-up). Verify: `test_inpaint_e2e` end-to-end parity vs the injected-noise PyTorch inpaint — **latent 6.6e-4, audio 2.4e-4**; CLI smoke (base→continue→inpaint) preserves kept regions (continue kept-L1 0.008; inpaint outside-mask 0.004 vs inside-mask 0.014). deps: E7.1–7.3, E6.4. **← M3**

## E8 — CUDA backend (parallelizable track)

- ✅ **E8.1** **P1** `aria_gpu.h` (device alloc/copy/sync + C-callable surface) + Makefile `cuda`/`test_cuda` wiring. nvcc 11.2 via `-ccbin gcc-9` (host gcc≤10), cudart at `/usr/lib/cuda/lib64`, `CUDA_ARCH ?= sm_61` (RTX 3070: `make cuda CUDA_ARCH=sm_86`). `make cuda` builds `libaria.a`+CLI; device detected (GT 1030, sm_61, 2 GB).
- ✅ **E8.2** **P1** CUDA gemm/linear: shared-memory tiled `aria_gemm_nt` (`y = x @ W^T + b`, device-pointer kernel + host wrapper). `make test_cuda` parity vs CPU on the DiT/encoder shapes + edge cases (odd dims, single row): maxdiff ≤ 1.8e-5. *Note:* GT 1030 (2 GB) is **validation-only** — it can't hold the 2.27 GB model and its fp32 won't beat the AVX2 CPU; real GPU speed is the RTX 3070 (sm_86, 8 GB). fp16 (sm_86) compute lands with E8.3/E9.
- ✅ **E8.3** **P1** CUDA hot ops: matmul, rmsnorm, gemma_rmsnorm, dynamic_tanh, softcap, silu, gelu_tanh, silu_gate, softmax (masked), rope, multi-head attention (gemm_nt+scale+softmax+matmul). Correctness-first kernels (norms use fp64 accum to match CPU). `make test_cuda` per-op parity on GT 1030: norms exact (0.0), rest ≤ 1.3e-4.
- ✅ **E8.4** **P1** CUDA conv1d (stride-1, pre-folded weights). Parity ≤ 1.4e-6.
- ✅ **E8.5** **P1** Device-resident DiT (`--device cuda`). The DiT weights upload once as **fp16** (fp32 compute: converted in the GEMM), activations live in a device bump arena, and the denoise loop runs on the GPU — only the per-step latent/velocity cross the bus. Encoder/decoder stay on CPU; per-request caches (cross K/V, RoPE) + per-step `gcond` computed on host and uploaded. Auto-fallback to CPU if no device / insufficient VRAM. Verified on GT 1030: weights ~0.84 GB fit the free VRAM, and GPU audio matches CPU at **0.57% rel-RMS (fp16-weight level, no NaNs)** — proves the orchestration. *Note:* GT 1030 is slower than the AVX2 CPU (weak Pascal + correctness-first kernels); the speedup target is the RTX 3070 (sm_86, 8 GB → F32 weights fit for exact parity). **← M4 (CUDA generation ≈ CPU)**
- ✅ **E8.5b** **P1** GPU perf pass on the RTX 3070 (sm_86), warm per-generation (10 s/8-step): **4.73 s → 1.28 s (3.7×)**, closing the gap to stable-audio-tools (0.36 s warm) from 13× to ~3.6×. Levers: persistent device weights (upload once, reuse — `setup` 6.7 s → 0.09 s warm) + `--bench`/`ARIA_PROFILE`; DiT GEMMs via **cuBLAS GemmEx** (fp16 tensor cores); **batched multi-head attention** (`cublasSgemmStridedBatched`, dit 1.77 s → 0.42 s); **parallel taae decoder** (per-thread arenas, decode 2.34 s → 0.76 s). Audio unchanged (0.6 % rel-RMS, no NaNs). Remaining warm cost: decode 0.76 s (CPU) > dit 0.42 s.
- ✅ **E8.5c** **P1** Device-resident GPU decoder. The taae decoder runs on the GPU, **batching all S=34 chunks of a pass** so the per-token linears become big cuBLAS GEMMs and the per-chunk differential attention runs B·H batched. RTX 3070 warm **decode 0.76 s → 0.02 s**; whole-pipeline warm generation **1.28 s → 0.61 s** — within **~1.7×** of stable-audio-tools (0.36 s), from 13× originally. Audio vs CPU 0.64 % rel-RMS, no NaNs (also verified on the GT 1030). Remaining micro-opts (DiT glue-kernel fusion, fewer fp16 conversions, trimming the 0.16 s warm setup) are now small vs the 0.42 s DiT. deps: E8.5b.
- ✅ **E8.5d** **P1** Profile-guided GPU kernels. `nsys` showed the naive **thread-per-row** norms dominated — `k_rmsnorm` alone was **51 %** of GPU time (and fp64-bound at 1/64 rate on Ampere), `k_softmax` 16 %. Rewrote both as **one-block-per-row fp32 shared-memory reductions** (`rsqrtf`). RTX 3070 warm **DiT 0.42 s → 0.16 s**, whole-pipeline generation **0.61 s → 0.29 s — now on par with / ahead of stable-audio-tools (0.36 s warm)**; aria's DiT (0.16 s / 8 steps) is ~2× its diffusion. fp32 norms within the fp16 tolerance (test_cuda 2.4e-7; audio vs CPU 0.66 % rel-RMS, no NaNs). **The 13× PyTorch gap is closed.**
- ⬜ **E8.5e** **P3** Last micro-opts (diminishing): compute the per-request cross-K/V on the GPU (shrinks the ~0.1 s setup), fuse the DiT glue kernels (adaLN/gate/add), Flash-style attention. deps: E8.5d · Verify: warm generation < 0.25 s; parity unchanged.
- ⬜ **E8.6** **P2** SSD weight streaming for VRAM-exceeding components (GT 1030). deps: E8.5 · Verify: medium runs on 2 GB.

## E9 — Precision & Quantization

- 🟡 **E9.0** **P1** Unified precision/dtype selection. `aria_dtype {fp32,fp16,bf16,q8,q4}` + the **dispatch seam** landed: `aria_qweight` (a Linear weight that is f32-borrowed or Q8/Q4-owned) + `aria_linear_qw`, plus `--precision` (CLI) / `aria_gen_params.precision`. The DiT block GEMMs (sa_qkv/out, ca_q/kv/out, ff) dispatch across precisions via an **additive overlay** (`aria_sa3_dit_quantize`) — the f32 path and the CUDA view are untouched (f32 stays the zero-copy mmap path, bit-identical; verified by `test_dit_full` 8.3e-4 + `test_quant_dit` fp32 round-trip = 0). q8/q4 force the CPU DiT. **Remaining:** fp16/bf16 as distinct CPU storage (currently fall back to fp32) + GPU precision selection (E9.4). deps: E2.9, E8.5 · Verify: `test_quant_dit` (q8 3.1 % velocity err, q4 36.8 %); `--precision` end-to-end runs. **Note:** CPU quant is a *footprint* win, not speed — the scalar dequant GEMM is ~6–8× slower than the AVX2 f32 microkernel; the speed win is the GPU (E9.4).
- 🟡 **E9.1** `∥` **P1** Q8 (int8 per-row) pack/unpack + CPU dequant-on-use gemm. **Primitive done** (`aria_quant.{c,h}`): `aria_q8_quant` (per-row symmetric, scale=max\|·\|/127) + `aria_linear_q8` (dequant-on-use, scale factors out per row). Hermetic `test_quant`: round-trip within half-step, GEMM relerr 0.6–1.0 % on the SA3 shapes (4× smaller than f32). **Wired** into the DiT (E9.0 overlay): `--precision q8` quantizes the block GEMMs on first use; `test_quant_dit` gates the model-level effect at **3.1 % velocity rel-RMS** (4.0× smaller block weights, 1.6 GB → 401 MB), and the f32 round-trip is bit-identical. deps: E9.0.
- ⬜ **E9.2** **P1** `aria-quantize` offline tool (safetensors → packed `.aria`: fp16/Q8/Q4). deps: E9.1 · Verify: round-trip; aria loads packed model + generates.
- ✅ **E9.3** **P1** Q4 (block) pack + dequant-on-use, **fidelity-tuned**. A parallel workflow swept 6 Q4 strategies vs `test_quant_dit`; the winner is **asymmetric (zero-point) int4** — per-block `[min, scale]`, `scale=(max−min)/15`, nibble in `[0,15]`, `W≈min+nib·scale` — **plus mixed precision**: the four attention projections (which feed qk-rmsnorm + softmax, where error is amplified) stay **Q8**, only the FFN goes Q4. Together this takes DiT velocity rel-RMS from **29.8 % → 9.32 %** (3.2× better, near q8's 2.45 %) at 300 MB (4.79× vs fp32, still < q8's 361 MB). Findings that lost: uniform asym 23.4 %, mixed-attn-only 11.5 %, block16 24 %. `test_quant` (GEMM relerr) + `test_quant_dit` (9.32 %, gate 13 %) green; fp32 round-trip bit-identical; GPU q4 matches CPU q4 (1.06 % rel-RMS, fp16-dequant level). deps: E9.1.
- ✅ **E9.4** **P1** CUDA dequant-on-use kernels (Q8/Q4). The device DiT stores the block GEMM weights **packed in VRAM** (`dqw`: q8/q4 + scales) and dequantizes each into a reused fp16 scratch (`k_dequant_q8/q4`) before the tensor-core cuBLAS GEMM — so resident VRAM shrinks (small-music 3070-class: fp16 ~1.73 GB → **q8 1.40 GB → q4 1.25 GB**), which is what lets medium fit. `--precision q8|q4` now runs on the GPU (host-quantizes the f32 view, uploads packed; host cross-K/V uses the CPU overlay, so it stays consistent). deps: E8.2, E9.3 · Verified on the GT 1030: compiles (nvcc 11.2), output **finite and numerically identical to the CPU quant path** (q8/q4 RMS match to 5 dp), `test_cuda` op-parity unchanged. **Note:** the GT 1030 (no tensor cores) is slightly *slower* at q8/q4 (dequant overhead, fp16 falls back to fp32); the speed win + the medium-fits payoff want the **RTX 3070** (sm_86) — bench there. **← M5 (mechanism; the medium demo lands with E10)**

## E10 — medium model

- ⬜ **E10.1** **P1** Load medium config/weights (embed 1536, depth 24, heads 24) + SAME-L decoder. deps: E4.4, E5.4 · Verify: tensors/shapes.
- ⬜ **E10.2** **P1** Medium end-to-end (differential DiT attn + larger decoder, quantized). deps: E10.1, E4.5, E9.3 · Verify: medium WAV parity. **← M6**

## E11 — Release polish (v1.0.0)

- ⬜ **E11.1** **P1** Library API finalize + `make install` + public headers. Verify: external program links `libaria.a`.
- ⬜ **E11.2** **P1** CLI UX: progress, seed, schedule/cfg flags, `--device`, error messages. Verify: manual run matrix.
- ⬜ **E11.3** **P2** Torch-matched RNG (Philox) option + determinism. deps: E6.2 · Verify: seed reproducibility + closer end-to-end parity.
- ⬜ **E11.4** **P2** CPU perf pass (profile-guided, beyond GEMM). Current built-in profile on `small-music` (`-d 10 -s 8 --device cpu`) is **setup 0.13 s / dit 7.36 s / decode 1.57 s**; `make bench` still shows the AVX2 GEMM at ~**175–280 GFLOP/s**, so the remaining CPU headroom is now mostly **data movement + non-GEMM kernels**, not more register-tile work. deps: E2.9, E2.9b, E13.3 · Verify: `make profile`/`--bench` warm run improves while parity stays green.
- ⬜ **E11.4a** **P1** Canonical token-major latent layout. Keep the latent/request path in one layout (prefer `[T,C]` / token-major) across sampler → DiT → decoder to delete the repeated `CT↔TC` full-buffer transposes and cut memory traffic at every denoise step + decode entry. deps: E11.4 · Verify: transpose helpers disappear from the hot path; warm CPU generation drops measurably; parity unchanged.
- ⬜ **E11.4b** **P1** Attention layout churn reduction. Project directly into the layout the attention kernel consumes (prefer head-major Q/K/V), keep cached cross-K/V packed, and avoid the per-block `extract_heads`/`merge_heads` memcpy traffic; only materialize score buffers when needed. deps: E11.4 · Verify: attention microbench + end-to-end DiT time improve; parity on `test_attn`/`test_dit(_full)` unchanged.
- ⬜ **E11.4c** **P2** Finish hot-loop scratch reuse. Move the remaining timestep/global-cond helper MLP scratch to `aria_arena`, keep decoder helper scratch stack/arena-resident, and fold pingpong's extra `denoised` buffer into an in-place update so the sampler only carries `x` + `v`. deps: E11.4 · Verify: no `malloc`/`free` in the steady-state denoise loop; peak RSS stable or lower; parity unchanged.
- ⬜ **E11.4d** **P2** SIMD elementwise/norm kernels. Add AVX2/FMA-specialized row reductions and vector math for RMSNorm / GemmaRMSNorm / DynamicTanh / SiLU / GELU-tanh / softmax (masked) so the post-GEMM scalar/libm kernels stop dominating the non-GEMM slice. deps: E11.4 · Verify: op microbenches + `make profile` show lower DiT/decoder wall time; tests green.
- ⬜ **E11.4e** **P2** Small-head CPU attention specialization. Add a CPU SDPA path specialized for `head_dim=64` (and decoder chunk size `S=34`) to reduce score-buffer traffic and improve locality relative to the generic `linear + softmax + matmul` decomposition. deps: E11.4b, E11.4d · Verify: `test_attn`/decoder parity; DiT/decoder CPU time drops.
- ⬜ **E11.4f** **P2** Decoder CPU cleanup. Restructure SAME decoder chunk passes to reduce pad/copy overhead, preserve per-thread arena reuse, and keep the chunked attention/conv path in cache-friendly batches; decoder is the second stage after DiT on CPU (~1.6 s / 10 s clip). deps: E11.4, E13.3 · Verify: decode stage shrinks on built-in profile; audio parity unchanged.
- ⬜ **E11.4g** **P3** Cold-path lookup/index cleanup. For exact-name tensor lookup and tokenizer tables, prefer a static exact-match index (sorted array + `bsearch`, robin-hood hash, or MPH) over radix/prefix trees; this is a load/tokenize cleanup, not a denoise-path optimization. deps: E11.4 · Verify: model load / tokenization improve modestly; no RSS regression from duplicating mmap-backed weights.
- ⬜ **E11.5** **P1** Docs: README usage, model-prep guide, AGENTS current. Verify: a fresh user can build + generate.
- ⬜ **E11.6** **P2** CI (`make test` on push) + parity smoke. Verify: CI green.
- ⬜ **E11.7** **P1** **v1.0.0 checklist**: small-music + medium · text→audio + continue + inpaint · CPU + CUDA · Q8/Q4 · steering hooks (E12, recommended — research driver) · all parity green · docs. **← v1.0.0**

## E12 — Steering (TasteSteer: training-free activation/representation steering)

Why: this runtime is the efficient inference vehicle for taste steering of SA3
(SAME-latent vs DiT-residual, steering-vs-LoRA, training-free core). The C runtime
exposes every intermediate tensor, so steering is an elementwise `x += α·d` at a
known op boundary plus a spec struct. **Can start as soon as M1 lands** and runs
in parallel with E7/E8/E9. Directions are `.atns` vectors; the parity harness
doubles as the activation-extraction path.

- ⬜ **E12.1** **P1** Steering spec + dispatch: `aria_steer` struct (`site`, `layer`, `dir`, `scale`, `step_lo/hi`) + a steer set threaded through `aria_generate`; orchestrator applies steers at registered hooks. deps: E6.4 · Verify: empty/scale-0 steer set ⇒ bitwise-identical output vs unsteered; non-zero scale changes output deterministically.
- ⬜ **E12.2** **P1** Diffusion-latent steering (256-D SAME arm) hook in the sampler loop, with step-window gating. deps: E12.1, E6.3 · Verify: unit test — injecting known `d` adds exactly `α·d` to `x` only within `[step_lo,step_hi]`; α=0 ≡ unsteered.
- ⬜ **E12.3** **P1** DiT residual-stream steering (per-layer 1024-D arm) hook after the chosen layer's residual add. deps: E12.1, E4.4 · Verify: steering layer L perturbs activations from L onward only; per-layer add matches expected; α=0 ≡ unsteered.
- ⬜ **E12.4** **P1** Conditioning-space steering (global adaLN 1024-D + cross-attn cond tokens) hook. deps: E12.1, E1.4, E1.5 · Verify: direction on `global_cond` shifts modulation params deterministically; α=0 ≡ unsteered.
- ⬜ **E12.5** `∥` **P2** Pre-decode latent steering hook (output-space nudge). deps: E12.1, E5.4 · Verify: direction added to latent pre-decode; α=0 ≡ unsteered.
- ⬜ **E12.6** **P1** `.atns` activation extraction at each site (mean-pooled taps) so contrastive directions (e.g. sweet − neutral) can be built from C runs. deps: E12.2–E12.4 · Verify: tapped activation matches the site's parity dump; contrastive direction is reproducible across runs.
- ⬜ **E12.7** **P1** CLI `--steer site:layer:dir.atns:scale:lo-hi` (repeatable) + direction loader. deps: E12.1 · Verify: parses multiple steers; produces measurably different, logged output. **← M7**
- ⬜ **E12.8** `∥` **P2** Steering validation sweep (scale/layer/site) scored with the taste regressor (wav2taste / sonic-taste-regressor). deps: E12.7 · Verify: taste metric responds monotonically-ish to scale for a known direction; reproducible.
- ⬜ **E12.9** **P2** (stretch) LoRA-style steering arm: `aria_linear_lora` op + adapter loader on chosen projections (the steering-vs-LoRA comparison). deps: E2.x, E4.4 · Verify: loaded adapter matches a Python LoRA forward within tol; zero-rank ≡ base.

## E13 — Batch / server (throughput)

Foundation laid in E2.9b: the immutable model (`aria_sa3_dit`, read-only weights)
is split from per-request scratch (`aria_sa3_dit_req`, owns its arena + caches),
so distinct requests share one loaded model with no shared mutable state. Two
complementary strategies:

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
