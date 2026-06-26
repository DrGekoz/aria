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
- ⬜ **E2.9** **P2** Blocked + threaded gemm (replace naive loop; keep parity). deps: E0.4 · Verify: `test_ops` green + benchmark speedup.

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
- ⬜ **E6.5** **P1** CFG (cond/uncond batch + guidance) for `cfg_scale>1` / base checkpoints. deps: E6.3 · Verify: parity with cfg>1.

## E7 — continue / inpaint

- ⬜ **E7.1** **P1** taae_v2 encoder (audio→latent): patch embed, downsampler, resampling, softnorm fwd. deps: E5.3, E2.7 · Verify: encode parity vs `pretransform.encode`.
- ⬜ **E7.2** **P1** Inpaint mask build (audio mask → nearest-interp to latent → masked_input). deps: E7.1 · Verify: mask/masked_input parity.
- ⬜ **E7.3** **P1** Local-additive cond live in DiT (E1.6 with real mask). deps: E1.6, E4.3, E7.2 · Verify: inpaint-cond block parity.
- ⬜ **E7.4** **P1** `aria_continue` + `aria_inpaint` API + CLI (`--continue` / `--inpaint --from --to`) + init-noise blend. deps: E7.1–7.3, E6.4 · Verify: kept regions preserved, masked changes; parity vs `generate_diffusion_cond_inpaint`. **← M3**

## E8 — CUDA backend (parallelizable track)

- ⬜ **E8.1** **P1** `aria_gpu.h` + device alloc/copy + Makefile cuda wiring (sm_86/sm_61). deps: op surface stable (E2.*) · Verify: `make cuda` builds, trivial kernel runs.
- ⬜ **E8.2** `∥` **P1** CUDA gemm/linear (fp32 sm_61; fp16 sm_86). deps: E8.1 · Verify: cuda-vs-cpu parity.
- ⬜ **E8.3** `∥` **P1** CUDA attention + rope + rmsnorm + softmax + activations. deps: E8.1 · Verify: per-op cuda-vs-cpu parity.
- ⬜ **E8.4** `∥` **P1** CUDA conv1d. deps: E8.1 · Verify: parity.
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
- ⬜ **E11.7** **P1** **v1.0.0 checklist**: small-music + medium · text→audio + continue + inpaint · CPU + CUDA · Q8/Q4 · all parity green · docs. **← v1.0.0**

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
