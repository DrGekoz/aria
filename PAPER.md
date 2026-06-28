# aria — paper notes: novelty, findings, and research directions

Working notes for a possible paper on `aria` (the dependency-free C runtime for
Stable Audio 3). Goal: separate **good engineering** (not publishable on its own)
from **defensible novelty**, record the empirical findings with their evidence, and
lay out two forward directions (megakernel inference; streaming/interactive
generation). Steering (TasteSteer) is deliberately kept for a later paper.

---

## 1. Honest triage — what is and isn't novel

Most of aria's techniques are **known, well-executed engineering**, not contributions:
packed AVX2 outer-product GEMM (BLIS-style), kernel fusion, dequant-on-use Q4/Q8,
mmap zero-copy weights + `madvise(MADV_DONTNEED)` reclaim, a bump-allocator arena,
banded/sliding-window attention, torch-matched RNG. A reviewer has seen all of these
in llama.cpp / ggml / FlashAttention / BLIS. **Do not lead with the kernels.**

Two things *are* contributions: the **artifact** and the **measurement story**.

---

## 2. Claim A — the artifact (systems/artifact contribution)

To our knowledge the **first dependency-free C runtime for audio *latent diffusion***:
a DiT denoiser + a transformer-resampling autoencoder (taae_v2) + a T5-Gemma text
encoder, **CPU and CUDA**, ~6.9 k LOC, no BLAS/framework. Runs SOTA Stable Audio 3
(small-music + medium) on **budget hardware** (8 GB / 2 GB GPUs, commodity CPUs).
llama.cpp-class artifacts exist for LLMs and a little for image diffusion; audio
diffusion has essentially none. Edge deployment of generative *audio* is the headline.

---

## 3. Claim B — re-deriving the efficiency priorities for audio diffusion

The interesting, citable part. Three findings, each counter-intuitive and each backed
by measurement on the same machine (RTX 3070 + i9-10900KF), aria vs `stable-audio-tools`:

### B1. The LLM efficiency playbook mis-transfers to audio diffusion
Audio latent diffusion compresses **4096×** in the autoencoder, so even a **60 s** clip
is only **S ≈ 710 tokens**. At that length the LLM staples are wrong:
- **FlashAttention is neutral-to-harmful.** We implemented it *twice* — an fp32 flash
  (DiT 4× *slower*) and a **WMMA tensor-core** flash (numerically correct, small-music
  GPU-vs-CPU 1.18 %, but still ~30 % slower; medium DiT 1.45→1.9 s). At S²=0.5 M the
  score matrix is L2-resident, so cuBLAS GemmEx + a parallel softmax reduction beats a
  hand-rolled flash. nsys: the softmax FA-2 would remove is only **~9 %** of GPU time.
- **KV-cache is irrelevant** (diffusion is non-autoregressive; every step is a full
  bidirectional pass).

  → *Negative result with a mechanism:* flash/KV-cache are tuned for the O(S²)-memory-
  bound, long-context LLM regime; audio diffusion lives below the crossover.

### B2. The bottleneck is the FFN GEMMs + a "long tail" of glue + the AE decoder
nsys breakdown of the medium GPU DiT: **GEMMs ~41 %**, attention softmax ~9 %, and a
**~32 % long tail** of small memory-bound elementwise/reshape kernels (fp16-convert
~9 %, rmsnorm ~7 %, RoPE ~5 %, extract/merge-heads, adaLN/gate). We fused the biggest —
extract-head + per-head qk-rmsnorm + RoPE into one kernel (`k_extract_normrope`), plus
rmsnorm+adaLN and gate+residual-add — for **medium GPU DiT 1.45 → 1.28 s (−12 %)**,
parity unchanged. The long tail, not attention, is the GPU lever.

### B3. A from-scratch C runtime *beats* the optimized framework for long audio
The reference implements the medium decoder's ±17 sliding window as a **dense O(N²)**
masked attention (the transformer sequence is `N = T×17`, 10,982 tokens at 60 s); aria
computes only the band, O(N·35). Result: **aria CPU is 1.94× faster than PyTorch+MKL at
60 s** (55.9 vs 108.4 s) — SAT's medium decode grows 9.6× from 10→60 s, aria's 5.0×.
"Hand-rolled C beats MKL" is a headline precisely because it shouldn't happen; the cause
(a latent O(N²) in the reference + diffusion's short sequences) is the point.

**Suggested paper:** *"Efficient audio-diffusion inference on edge hardware: why the LLM
playbook doesn't transfer."* Contribution = artifact + the B1–B3 characterization. The
novelty is the *re-derivation of priorities*, not any single kernel.

### Benchmark data (medium, 60 s reference, uncond 8-step, RTX 3070 + i9-10900KF)
| backend | aria 60 s | SAT 60 s | verdict |
|---|---|---|---|
| CPU (20 thr) | 55.9 s | 108.4 s | aria 1.94× faster |
| GPU (fp16) | 2.07 s (DiT 1.28 + dec 0.78) | 1.32 s | SAT 1.57× faster |

GPU DiT kernel mix (nsys): GEMM ~41 %, long-tail glue ~32 %, softmax ~9 %, decoder band
+ conv ~14 %. ~14 k kernel launches/gen (launch-heavy).

---

## 4. Direction 1 — megakernel / persistent-kernel diffusion (research novelty)

**Idea.** Fuse the denoise step (or the DiT block) into one persistent kernel, keeping
the small audio latent + activations on-chip across the launch-heavy glue, eliminating
HBM round-trips of activations and per-kernel launch overhead. Megakernels are emerging
for **LLM batch-1 decode**; nobody has applied them to **diffusion**, and audio is
uniquely suited (batch-1, fixed 8-step loop, tiny working set).

**Feasibility (working-set budget, RTX 3070: 100 KB shared/SM, 4 MB L2):**
- **Latent** 256×T: a 6 s streaming window is T≈65 → 256×65 = 65 KB. Fits L2 easily;
  keeping it resident across the 8 steps is feasible.
- **Block activations** S×dim and the FFN intermediate S×(4·dim): at S≈130 that's
  ~0.2–3 MB — too big for shared (100 KB) but L2-resident (4 MB).
- **Weights** (~10 MB fp16/layer): do **not** fit on-chip; the GEMMs must stream them
  from HBM regardless. So the GEMMs stay HBM/weight-bound — a hand-rolled megakernel
  GEMM gains nothing there and risks losing to cuBLAS (cf. the flash result, §B1).

**Conclusion → two tiers:**
- *Megakernel-lite (recommended, low-risk):* **CUDA Graphs** — capture the per-step
  kernel sequence once and replay it 8×, killing the ~14 k-launch overhead while keeping
  the cuBLAS GEMMs. Plus the glue fusions already landed (−12 %). This captures the
  launch/activation-traffic win without the cuBLAS-vs-hand-rolled risk.
- *True megakernel (research):* hand-fuse a whole block incl. tensor-core GEMMs (raw
  `mma.sync`, `cp.async` pipelining, on-chip activation residency). High effort, real
  risk it doesn't beat cuBLAS at these sizes. Only worthwhile if framed as a study of
  *when* megakernels win for short-sequence diffusion (the crossover analysis is the
  contribution). The −12 % from fusing *three* glue kernels is the motivating evidence.

**Implemented — CUDA Graphs (megakernel-lite), validated on the RTX 3070.** `nvcc
--default-stream per-thread` makes every `<<<>>>` use the capturable per-thread stream; all
three cuBLAS handles (DiT + 2 decoders) are pinned to it, with a fixed 4 MB workspace so
cuBLAS can't lazily `cudaMalloc` mid-capture. The per-step DiT compute (`dit_step_compute`)
is captured on step 0 and replayed via `cudaGraphLaunch` for the rest, with a robust fallback
(run inline if capture/instantiate fails → correctness preserved). `free_request` drops the
graph per request (the baked device pointers + (S,T) shape change); `ARIA_NO_GRAPH=1` toggles
the path for A/B.
- **Correct:** graph output is **byte-identical** to the inline path (rel_rmse 0.00000,
  corr 1.00000); both differ from CPU only by the usual fp16 (2.1 % rel_rmse text→audio).
- **Modest win at these sizes:** ~**4 %** warm-min on the **medium** DiT (s=24: 0.96→0.92 s);
  within noise on small-music (s=8: 0.12 vs 0.13 s). The small audio latent keeps the DiT
  GPU-compute-bound, so most of the ~14 k launches already hide behind compute — the graph
  only trims the non-overlapped tail. The clearer secondary benefit is **lower per-step
  latency variance** (fewer launch-jitter spikes → steadier streaming chunk times). The win
  grows with launch density (more steps; the medium differential path's extra kernels).
- **Takeaway (confirms §B1/§B3):** at short audio-diffusion sequences the bottleneck is
  weight/compute, not launches. The graph is the right *low-risk* tier; the bigger lever is
  the True-Megakernel GEMM fusion, which must beat cuBLAS to pay off (the crossover study).

---

## 5. Direction 2 — streaming / interactive generation (the one to build now)

**Goal (user's):** generate music continuously and **change the prompt as it plays** —
the Lyria RealTime / Magenta RealTime experience, but with latent diffusion on edge HW.

**Prior art — Magenta RT / Lyria RT ("Live Music Models", arXiv 2508.04651).** A
**codec language model**: SpectroStream RVQ codec (48 kHz, frame rate 25 Hz, 16 of 64
RVQ levels = 4 kbps = 400 tok/s) + an encoder-decoder T5 LM with a depthformer decoder
(temporal × depth modules). The streaming mechanism is **chunk-based autoregression**:
generate chunk `C = 2 s` conditioned, under a Markov assumption, on a **coarse** history
(first 4 RVQ levels) of `H = 5` previous chunks (**10 s context**) plus a live
**MusicCoCa** style embedding `c_i` (a joint audio-text 768-d embedding quantized to 12
tokens — text *or* audio prompt). Crucially it is **stateless**: each chunk is predicted
afresh from the 10 s context (no KV cache), so the conditioning (style/prompt) can be
swapped *between every chunk* → live control. RTF 1.8 on an H100 (T5-Large). Two ideas
do the heavy lifting: chunk-AR for infinite streaming without train/inference length
mismatch, and the coarse-context trick for throughput.

**aria's angle (and the novelty).** Keep the **same streaming structure** — chunked,
context-conditioned, stateless, live-prompted — but replace the AR codec LM with
**latent diffusion**. SA3 already gives the primitive: `--continue`/inpaint encodes
context audio → continuous latent, masks the tail as "regenerate," and an 8-step
diffusion fills the tail conditioned on the kept head (`sa3_build_inpaint_local`). So
"chunk `i`" = run diffusion over a `[context | new chunk]` window, emit the new chunk,
slide. This is **chunk-based streaming with a diffusion core instead of an AR codec LM**
— not, to our knowledge, demonstrated for music. Mapping to Magenta RT: their `C=2 s`
chunk / `H=10 s` context → our emit-chunk / context-window; their per-chunk MusicCoCa
style swap → our per-chunk T5-Gemma prompt re-encode (and later a steering vector); their
coarse-RVQ history → our latent-encoded context (already compact, 256-d @ 10.7 Hz).
A streaming loop is:
1. generate an initial window;
2. each step: take the last `ctx` s as context, build a window `[context | regenerate
   chunk]`, run 8-step diffusion, **emit the new `chunk` s**;
3. slide the context forward; **re-read the prompt each step** → live prompting;
4. (later) inject a steering vector each step → live *taste* control.

This is **"streaming music generation via sliding-window latent-diffusion continuation
with live prompting"** — distinct from the AR-codec-LM approach of Magenta/Lyria, and
not (to our knowledge) demonstrated for audio. It rides on aria's existing inpaint path.

**Latency budget (why it's real-time):** aria GPU does ~0.034 s of compute per second of
audio (60 s in 2.07 s). A 2 s emit on a ~12 s window (matching Magenta's 10 s context)
costs ~0.4 s GPU → **~5× real-time (RTF ≈ 5)**, vs Magenta RT's RTF 1.8 on an H100 — the
diffusion core is *cheaper per chunk* on far smaller hardware (a few denoise steps over a
256-d latent), trading the open coherence question below. Smaller windows lower latency
further. small-music CPU (~0.27 s/s on the i9) is also real-time-capable; medium CPU is
borderline. So the **GPU is already fast enough**; megakernel/CUDA-graph work only widens
the margin (lower inter-chunk latency = snappier live control).

**Open questions (the empirical study):**
- *Coherence:* SA3 is trained for fixed clips, not infinite streaming — does
  sliding-window continuation stay musically coherent over many chunks, or drift? How
  much context (`ctx`) is needed? Crossfade/overlap at boundaries to avoid seams.
- *Prompt-switch dynamics:* how fast/smoothly does the output follow a mid-stream prompt
  change (the interactive feel)?
- *Window vs latency vs quality* trade-off (smaller chunks = lower latency, more
  overhead, less context).
- *vs a purpose-built streaming model* (Magenta RT) on coherence/controllability.

**Build plan:** add a resident `--stream` mode (model loaded once; in-memory rolling
context; per-chunk inpaint-continuation; emit chunks to a growing WAV or stdout; read
prompt changes from stdin). Verify continuous output + per-chunk latency first; live
stdin prompting is a small addition; steering hooks slot in later.

**Prototype results (implemented; `aria --stream`, small-music, i7 CPU).** The
mechanism works end-to-end: resident model, in-memory sliding-window continuation, live
prompt re-read per chunk. Two honest findings:
- *Latency / the diffusion-continuation tax.* Chunk 1 (plain text→audio, 6 s) ran at
  **RTF 1.4×**, but each continuation chunk emits 2 s while **regenerating the whole 6 s
  window** → **RTF 0.4×** on CPU. Unlike an AR codec LM (which only generates the new
  chunk's tokens, context as cheap attention), diffusion-continuation pays O(window) per
  O(chunk) emitted — the context/chunk ratio is a hard overhead multiplier. Real-time
  needs the **GPU** continuation path (≈10× projected; currently the local-additive
  inpaint cond is **CPU-only** — the GPU DiT must gain it) or a smaller context.
- *Quality — diagnosed and **fixed**.* The naive tail-inpaint faded: profiling a long
  continuation (8 s seed → regenerate to 18 s) gave the regenerated region's per-second RMS
  as `[0.0, 0.218, 0.212, 0.208, 0.196, 0.224, 0.206, 0.045, 0.006, 0.0]` — a **~1.5 s
  silent seam, then ~6 s of strong coherent body, then a ~3 s fade-out**. SA3 inpaint-of-
  the-**tail** generates a clip *ending* (it has only seen "end of clip" in that position),
  but the **body is real**. So the fix is **lookahead-emit + crossfade**: each step
  regenerates `context + skip + emit + tail` and emits only the **strong body** — skipping
  the seam, discarding the fade — crossfaded onto the output, with the next context taken
  from the just-emitted (strong) audio. **Result:** the fade is gone — per-second RMS now
  **flat at ~0.22, 0/25 s near-silent** (was 0.12, 6/24 s silent). Continuous, coherent
  streaming music with live prompt re-steering between chunks. This is the streaming
  algorithm a paper would present (and where live *taste* steering later slots in).

  *Side effect:* the prototype exposed and fixed a latent heap-overflow in the existing
  `--continue`/inpaint path (the SAME encoder's latent length is `ceil(T/2)·2`, i.e. `T+1`
  for odd `T`, vs the DiT's `T`; the `256·T` buffer overflowed — ASan-confirmed, now sized
  to the encoder output).

**Net:** streaming + live prompting **works** (`aria --stream`). The streaming *method*
is three dependency-free stages addressing three distinct failure modes:
1. **lookahead-emit** — emit the strong body of the regenerated region, skip the post-
   context seam, discard the trailing fade (SA3 inpaint makes tails into outros);
2. **phase-aligned crossfade** (`--stream`) — cross-correlate the seam over ±0.18 s and
   shift before crossfading, removing drum "flam" from beat-misaligned takes;
3. **HPSS background-hold** (`--stream --hold`) — separate each chunk (median-filter
   HPSS, own FFT), accumulate only the **harmonic** (evolving melody) while a single
   **percussive loop** (the drum groove) is held and tiled across the whole stream and
   reset on a prompt change — so the rhythm is identical every bar (no flam, no drift)
   and the foreground evolves on top. This is the user's foreground/background idea.

Remaining: (a) **real-time on the edge** — continuations regenerate a ~14 s window per
~4 s emit (RTF 0.5–0.7× CPU, **~10× on the 3070** once the inpaint local-cond is ported
to the device DiT, currently CPU-only); (b) the held loop is a fixed repeat with un-
seamed loop points (a beat-synchronous loop + crossfade would polish it); (c) the
harmonic can still degrade past ~5–6 chunks (the inpaint-drift ceiling — the principled
fix is fine-tuning SA3 for chunk-AR, cf. Magenta RT). Live *taste* steering then slots
in as a per-chunk vector add. The proper end state is a streaming-trained model; the
above is how far a careful post-hoc pipeline gets on a fixed-clip diffusion model.

**Follow-up (later) — beat-synchronous seamless drum loop.** The `--hold` loop is
currently a fixed-length repeat (= chunk-0 emit length) tiled with naive modulo, so it
sounds loopy and ticks at the loop point. Polish: (1) estimate tempo from the
**onset-envelope autocorrelation** of the held *percussive* stem (half-wave-rectified
spectral-flux → autocorrelate → first strong lag = beat period); (2) trim the held loop
to an **integer number of bars**; (3) **crossfade the loop seam** (blend the loop tail
into its head) so the modulo wrap is clickless; (4) optionally re-extract the loop every
few bars so a slowly-evolving groove is allowed while staying phase-locked. Same
machinery (the existing HPSS + a ~50-line autocorrelation tempo estimator), no new deps.

---

## 6. Standard-engineering disclosure (for honest framing)

Not claimed as novel, but part of the artifact: AVX2 packed GEMM, OpenMP head-parallel
attention, query-blocked attention, Q4/Q8 dequant-on-use, mmap + madvise weight
management, device-resident DiT/decoder with cuBLAS tensor cores, the arena allocator,
torch-matched RNG, the differential-attention / sliding-window / sinusoidal-FF specifics
of the medium model. Reproducible via `BENCHMARKS.md` (`./aria --uncond -d 60 --bench 3`
+ `scripts/bench_sat.py`).

---

## 7. One-line summary

The publishable core is **not** "we wrote fast kernels" but **(1)** a dependency-free C
runtime that puts SOTA audio diffusion on edge hardware, and **(2)** the empirical
finding that audio diffusion's heavy VAE compression inverts the LLM efficiency
playbook — attention is cheap, the glue and the AE decoder are the cost, flash hurts,
and a careful C runtime can beat the framework. The forward bets are **megakernel/CUDA-
graph inference** and **streaming latent-diffusion generation with live control**.
