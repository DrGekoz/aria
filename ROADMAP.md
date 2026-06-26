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
| **M1** | First audio: `text→audio` (precomputed embeddings + injected noise) matches Python | E6.4 |
| **M2** | Self-contained text: T5Gemma encoder in C | E3.4 + E3.5 |
| **M3** | continue / inpaint working | E7.4 |
| **M4** | CUDA end-to-end generation | E8.5 |
| **M5** | Quantized (Q4) — medium fits low VRAM | E9.4 |
| **M6** | medium model end-to-end | E10.2 |
| **M7** | Steering (TasteSteer): latent + DiT-residual + cond-space injection, training-free | E12.7 |
| **v1.0.0** | Release checklist green | E11.7 |

## Status snapshot

Done: **E0** (scaffold/infra), **E1.1** (NumberConditioner), **E0.8** (parity
harness). T5Gemma weights downloaded locally. Build + hermetic tests +
number-cond parity all green.

## Start-now parallel front (no dependencies)

These can be implemented concurrently, in any order, right now:
**E1.2** (config parser), **E2.1** (RoPE), **E2.2** (attention), **E2.7**
(conv1d), **E3.1** (tokenizer), **E6.2** (RNG). Knocking these out unblocks most
of the critical path.

## Critical path to M1 (first audio)

`E1.2 → {E1.3, E1.4, E1.5}` · `E2.1,E2.2,E2.3,E2.4,E2.8` · `E4.1→E4.2→E4.3→E4.4`
· `E5.1→E5.2→E5.3→E5.4` · `E6.1,E6.3 → E6.4`.
T5Gemma (E3) is **off** the critical path — use E3.5 precomputed embeddings until
M2.

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
- ⬜ **E1.2** `∥` **P0** Nested JSON config parser (object navigation: `model.diffusion.config.*`, `model.pretransform.*`, `conditioning.configs[]`). deps: — · Verify: unit test on cached `model_config.json` returns embed_dim=1024, depth=20, num_heads=16, latent 256, downsample 4096.
- ⬜ **E1.3** **P0** Timestep features (`expo`) + `to_timestep_embed` MLP. deps: E1.1, E1.2 · Verify: parity of timestep embedding for fixed t (dump from dit.py).
- ⬜ **E1.4** **P0** Global adaLN: `to_global_embed` MLP on (timestep+seconds) → `global_cond[1024]`; per-layer `to_scale_shift_gate` add → 6 chunks. deps: E1.2, E1.3 · Verify: parity of global_cond + block-0 modulation params.
- ⬜ **E1.5** **P0** Cross-attn cond pack: `to_cond_embed` MLP on concat`[prompt 256×768 | seconds 1×768]` → `[257×1024]` + mask. deps: E1.2 · Verify: parity of cross_attn_cond.
- ⬜ **E1.6** **P1** Local-additive inpaint cond build (`[mask | masked_input]` 257-d) + zero-init `to_local_embed` MLP (NULL-safe for text→audio). deps: E1.2 · Verify: parity (full path in E7.3).

## E2 — Core ops (shared by DiT / T5Gemma / taae)

- ⬜ **E2.1** `∥` **P0** RoPE op (θ-param, per-head q/k rotation). deps: — · Verify: parity vs reference rope on random q + known positions.
- ⬜ **E2.2** `∥` **P0** Scaled-dot-product attention (multi-head, non-causal, optional additive mask). deps: — · Verify: parity vs torch SDPA on random q,k,v.
- ⬜ **E2.3** **P0** QK-RMSNorm in attention (per-head rmsnorm of q,k with `.gamma[64]`). deps: E2.2 · Verify: parity on self-attn with qk_norm.
- ⬜ **E2.4** **P0** Cross-attention variant (kv from context, split `to_q`/`to_kv`). deps: E2.2 · Verify: parity.
- ⬜ **E2.5** `∥` **P1** Attn logit softcapping (`tanh(s/50)*50`, T5Gemma). deps: E2.2 · Verify: parity.
- ⬜ **E2.6** **P1** Differential attention (q,k base/diff split; `attn(base) − λ·attn(diff)`). deps: E2.2 · Verify: parity (taae + medium DiT). *High-risk — confirm formula from transformer.py.*
- ⬜ **E2.7** `∥` **P0** conv1d (weight-normalized; kernel/stride/pad). deps: — · Verify: parity vs torch Conv1d.
- ⬜ **E2.8** `∥` **P0** GLU/SiLU-gated FFN helper (`ff.0.proj[2·inner]` → silu-gate → `ff.2`). deps: — · Verify: parity vs a DiT FFN.
- ✅ **E2.9** **P1** Fast GEMM. Done: blocked+omp-simd (6.6× over naive) → **register-blocked AVX2 microkernel** (MR×NR ymm tiles + ILP, cache-blocked, K%8 + edge tails; `make bench` shows 175–280 GFLOP/s vs the loop's ~30–40), dependency-free with a pure-C fallback for non-AVX2. End-to-end generation 17 s → 10.5 s (~1.6×); compute now ~competitive with PyTorch CPU once the one-time weight fault-in is excluded. Optional `-DARIA_BLAS` backend kept (dimension-gated; pthread-OpenBLAS oversubscribes vs OpenMP, so prefer the OpenMP build or MKL/Accelerate). Verified: `test_ops` + e2e/t5enc parity green.
- ✅ **E2.9b** **P1** Workspace reuse + step-invariant caching. New `aria_arena` (bump allocator, save/restore) replaces the per-block/per-step `malloc`/`free` churn in the DiT hot path (no re-faulting large scratch 160×/gen); `aria_ff_glu`/`aria_attention` take optional caller scratch. Split immutable model (`aria_sa3_dit`, weights) from a per-request context (`aria_sa3_dit_req`: arena + cached `cross_ed`, RoPE tables, `to_global_embed(seconds)`, and **per-block cross-attention K/V** projected once instead of every step). `req_begin`/`step`/`end`; one-shot `aria_sa3_dit_forward` wraps them. End-to-end 11.1 s → 9.1 s (~19%, min-of-N) with much tighter variance; bit-identical output, all parity green. The req/model split is the foundation for the batch API (E11).
- ⬜ **E2.9c** **P2** Further GEMM: weight packing (contiguous panels), tune MR/NR per cache, batch the per-head attention so it's BLAS/library-friendly; extend the arena to the encoder/decoder forwards. deps: E2.9, E2.9b · Verify: `make bench` improvement, parity unchanged.
- ⬜ **E2.10** **P2** Hardware-adaptive backend selection: compile-time (`make` / `blas` / `cuda` / `mps`) + runtime device detection behind the one `aria_ops.h` surface; a fat binary can bundle CPU+CUDA and pick at runtime. deps: E2.9, E8 · Verify: each backend produces matching audio; auto-picks the best available.

## E3 — T5Gemma text encoder + tokenizer (parallel track, off critical path)

- ⬜ **E3.1** `∥` **P1** Tokenizer: load BPE (tokenizer.json) or SentencePiece (tokenizer.model); encode + Gemma template. deps: — · Verify: token-id parity vs HF tokenizer on prompt set.
- ⬜ **E3.2** `∥` **P1** T5Gemma weight load: filter `encoder.*`, name-map 12 layers. deps: E1.2 · Verify: shapes/counts match config.
- ⬜ **E3.3** **P1** Encoder forward: embed `·√d_model`, RMSNorm, self-attn (bidirectional + RoPE + softcap + query_pre_attn_scalar 64), GeGLU `gelu_tanh`. deps: E2.1, E2.2, E2.5, E3.2 · Verify: per-layer + final hidden parity vs `T5GemmaEncoderModel`.
- ⬜ **E3.4** **P1** Conditioner wrapper → `[1,256,768]` + mask + learned padding embedding. deps: E3.1, E3.3 · Verify: parity vs `T5GemmaConditioner.forward`. **← M2 (with E3.5)**
- ⬜ **E3.5** **P0** Precomputed-embedding path (CLI `--prompt-embed <file>`). deps: E1.5 · Verify: generate from dumped embedding == Python (unblocks M1 before E3.3).

## E4 — SA3 DiT forward

- ⬜ **E4.1** **P0** DiT weight load + name map (pre/postprocess conv, to_cond/global/timestep embed, 20 layers, memory tokens). deps: E1.2 · Verify: all tensors found, shapes logged.
- ⬜ **E4.2** **P0** Input path: `preprocess_conv` + project to embed_dim + prepend 64 memory tokens + RoPE positions. deps: E4.1, E2.7, E2.1 · Verify: pre-block hidden parity.
- ⬜ **E4.3** **P0** DiT block forward (one block): adaLN(6) → self-attn(qk-rms,rope)+`σ(1−gate)`+res → cross-attn+res → local-add(NULL-safe) → FFN GLU+gate+res. deps: E2.1–2.4, E2.8, E1.4, E1.5 · Verify: **block-0 parity** vs Python (keystone op test).
- ⬜ **E4.4** **P0** Full DiT: 20 blocks + final norm + `postprocess` → velocity `[256,T]`. deps: E4.2, E4.3 · Verify: full `denoiser_forward(x,t,cond)` parity.
- ⬜ **E4.5** **P1** Differential attention path for medium (config-gated). deps: E2.6, E4.4 · Verify: medium block parity.

## E5 — taae_v2 autoencoder decoder

- ⬜ **E5.1** **P0** Decoder weight load + name map (bottleneck, layers, resampling, conv mapping). deps: E1.2 · Verify: tensors found/shaped.
- ⬜ **E5.2** **P0** Softnorm bottleneck inverse (running_std/scaling_factor + bias). deps: E5.1 · Verify: parity.
- ⬜ **E5.3** **P0** TransformerResamplingBlock (decode): chunked attention (chunk 32, midpoint shift), learnable `new_tokens`, DynamicTanh, differential attn, depth 6. deps: E2.6, E2.7 · Verify: one-block then stack parity. *Highest-risk task.*
- ⬜ **E5.4** **P0** Residual upsampler (stride 16) + conv mapping + unpatch (256-sample) → stereo audio. deps: E2.7, E5.3 · Verify: full decoder latent→audio parity vs `pretransform.decode`.

## E6 — Sampler + end-to-end (KEYSTONE)

- ⬜ **E6.1** **P0** LogSNR schedule (linspace + LogSNRShift anchor −6.2/end 2.0, t[0]=σ_max). deps: E1.2 · Verify: schedule parity vs `build_schedule`.
- ⬜ **E6.2** `∥` **P0** RNG: xoshiro256** + Gaussian. deps: — · Verify: distribution stats; Philox parity later (E11.3).
- ⬜ **E6.3** **P0** Pingpong loop `x ← (1−t_next)(x − t·v) + t_next·noise`. deps: E4.4, E6.1 · Verify: with **injected Python noise**, latent parity vs `sample_flow_pingpong`.
- ⬜ **E6.4** **P0** End-to-end `text→audio` (precomputed emb + injected noise): cond → init noise → sampler → decoder → WAV. deps: E1.4, E1.5, E3.5, E4.4, E5.4, E6.3 · Verify: C WAV vs Python WAV (waveform MSE under tol). **← M1**
- ⬜ **E6.5** **P2** CFG + `--cfg` CLI flag: cond/uncond two-pass guidance `v = v_uncond + cfg·(v_cond − v_uncond)` for `cfg_scale > 1` (≈2× cost/step; runs the denoiser twice with a null/empty prompt). **Only worthwhile on BASE checkpoints** — the post-trained/distilled small-music is tuned for `cfg = 1.0` (CFG off), so `cfg > 1` there is out-of-distribution and not recommended. deps: E6.3 · Verify: parity vs the `stable_audio_tools` CFG path on a base checkpoint. (`--seed` is already implemented.)

## E7 — continue / inpaint

- ⬜ **E7.1** **P1** taae_v2 encoder (audio→latent): patch embed, downsampler, resampling, softnorm fwd. deps: E5.3, E2.7 · Verify: encode parity vs `pretransform.encode`.
- ⬜ **E7.2** **P1** Inpaint mask build (audio mask → nearest-interp to latent → masked_input). deps: E7.1 · Verify: mask/masked_input parity.
- ⬜ **E7.3** **P1** Local-additive cond live in DiT (E1.6 with real mask). deps: E1.6, E4.3, E7.2 · Verify: inpaint-cond block parity.
- ⬜ **E7.4** **P1** `aria_continue` + `aria_inpaint` API + CLI (`--continue` / `--inpaint --from --to`) + init-noise blend. deps: E7.1–7.3, E6.4 · Verify: kept regions preserved, masked changes; parity vs `generate_diffusion_cond_inpaint`. **← M3**

## E8 — CUDA backend (parallelizable track)

- ✅ **E8.1** **P1** `aria_gpu.h` (device alloc/copy/sync + C-callable surface) + Makefile `cuda`/`test_cuda` wiring. nvcc 11.2 via `-ccbin gcc-9` (host gcc≤10), cudart at `/usr/lib/cuda/lib64`, `CUDA_ARCH ?= sm_61` (RTX 3070: `make cuda CUDA_ARCH=sm_86`). `make cuda` builds `libaria.a`+CLI; device detected (GT 1030, sm_61, 2 GB).
- ✅ **E8.2** **P1** CUDA gemm/linear: shared-memory tiled `aria_gemm_nt` (`y = x @ W^T + b`, device-pointer kernel + host wrapper). `make test_cuda` parity vs CPU on the DiT/encoder shapes + edge cases (odd dims, single row): maxdiff ≤ 1.8e-5. *Note:* GT 1030 (2 GB) is **validation-only** — it can't hold the 2.27 GB model and its fp32 won't beat the AVX2 CPU; real GPU speed is the RTX 3070 (sm_86, 8 GB). fp16 (sm_86) compute lands with E8.3/E9.
- ✅ **E8.3** **P1** CUDA hot ops: matmul, rmsnorm, gemma_rmsnorm, dynamic_tanh, softcap, silu, gelu_tanh, silu_gate, softmax (masked), rope, multi-head attention (gemm_nt+scale+softmax+matmul). Correctness-first kernels (norms use fp64 accum to match CPU). `make test_cuda` per-op parity on GT 1030: norms exact (0.0), rest ≤ 1.3e-4.
- ✅ **E8.4** **P1** CUDA conv1d (stride-1, pre-folded weights). Parity ≤ 1.4e-6.
- ⬜ **E8.5** **P1** Per-component device placement + load/free orchestration (DiT on GPU; enc/dec CPU-or-GPU; fit budget). deps: E8.2–8.4 · Verify: CUDA generation == CPU. **← M4**
- ⬜ **E8.6** **P2** SSD weight streaming for VRAM-exceeding components (GT 1030). deps: E8.5 · Verify: medium runs on 2 GB.

## E9 — Quantization

- ⬜ **E9.1** `∥` **P1** Q8 (int8 per-row) pack/unpack + CPU dequant-on-use gemm. deps: E2.9 · Verify: quantized linear parity (looser tol); size halved.
- ⬜ **E9.2** **P1** `aria-quantize` offline tool (safetensors → packed `.aria`: fp16/Q8/Q4). deps: E9.1 · Verify: round-trip; aria loads packed model + generates.
- ⬜ **E9.3** **P1** Q4 (block) pack + dequant-on-use. deps: E9.1 · Verify: parity (tol); medium fits low VRAM.
- ⬜ **E9.4** **P1** CUDA dequant-on-use kernels (Q8/Q4). deps: E8.2, E9.3 · Verify: cuda quant parity. **← M5**

## E10 — medium model

- ⬜ **E10.1** **P1** Load medium config/weights (embed 1536, depth 24, heads 24) + SAME-L decoder. deps: E4.4, E5.4 · Verify: tensors/shapes.
- ⬜ **E10.2** **P1** Medium end-to-end (differential DiT attn + larger decoder, quantized). deps: E10.1, E4.5, E9.3 · Verify: medium WAV parity. **← M6**

## E11 — Release polish (v1.0.0)

- ⬜ **E11.1** **P1** Library API finalize + `make install` + public headers. Verify: external program links `libaria.a`.
- ⬜ **E11.2** **P1** CLI UX: progress, seed, schedule/cfg flags, `--device`, error messages. Verify: manual run matrix.
- ⬜ **E11.3** **P2** Torch-matched RNG (Philox) option + determinism. deps: E6.2 · Verify: seed reproducibility + closer end-to-end parity.
- ⬜ **E11.4** **P2** CPU perf pass (SIMD hot paths, threading). deps: E2.9 · Verify: benchmark targets, parity unchanged.
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
