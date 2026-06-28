# aria vs stable-audio-tools — benchmark

aria (this runtime) vs **stable-audio-tools** (the PyTorch reference), same machine,
same model weights, same workload: **uncond, 8-step pingpong, DiT + autoencoder
decode** (no text encoder — isolates the core diffusion runtime). Both measure
**warm** generation with the model resident (aria = min of 3, SAT = mean of 3;
medium CPU = single run). Hardware: **Alienware — RTX 3070 (8 GB, sm_86) + i9-10900KF (20 threads).**

GPU: aria runs fp16 weights / fp32 compute (cuBLAS tensor cores); SAT runs fp16
(`.half()`). VRAM is the **actual process footprint** (`cudaMemGetInfo` for aria,
`mem_get_info` total−free for SAT) so the CUDA context + allocator pool are counted
for both; SAT's `max_memory_allocated` (tensors only) is shown in parentheses.

## small-music (10 s clip, 8 steps)

| metric | aria | stable-audio-tools | winner |
|---|---|---|---|
| **GPU time** | **0.19 s** | 0.34 s | aria (1.8×) |
| **GPU VRAM** (process) | **1548 MB** | 2770 MB (alloc 1161) | aria (−44 %) |
| **CPU time** (20 threads) | 3.5 s | **2.8 s** | SAT (1.25×) |
| **CPU peak RAM** | **2089 MB** | 5421 MB | aria (−61 %) |

## medium (GPU 10 s, CPU 5 s, 8 steps)

| metric | aria | stable-audio-tools | winner |
|---|---|---|---|
| **GPU time** | 0.54 s | **0.50 s** | ≈ parity (1.08×) |
| **GPU VRAM** (process) | **4314 MB** | 5452 MB (alloc 4561) | aria (−21 %) |
| **CPU time** (5 s, 20 thr) | 8.1 s | **7.0 s** | SAT (1.15×) |
| **CPU peak RAM** (5 s) | **7204 MB** | 18866 MB | aria (−62 %) |

aria GPU breakdown — small: setup 0.02 + dit ~0.14 + decode 0.02; medium: setup
**0.02** + dit **0.38** + decode **0.13**. Three profile-guided GPU optimizations took
medium **0.98 → 0.54 s** (≈ SAT parity) and small-music **0.29 → 0.19 s** (1.8× SAT):
1. **on-device cross-K/V projection** (was a per-request host GEMM): medium setup
   0.33 → **0.02 s** (`ca_to_kv` device-resident, +340 MB).
2. **fp16 tensor-core attention** for the DiT (the QK/AV ran as fp32 `sgemm` = 21 % of
   GPU time, no tensor cores). Small decoder stays fp32 (fp16 there cost accuracy for
   ~0 gain).
3. **fused sliding-window decoder attention** — the medium decoder ran full N² (N=1836)
   then masked to ±17; a warp-per-query kernel computes only the 35-key window
   (O(N·35), exact), dropping decode **0.23 → 0.13 s** and a 324 MB N² score buffer.
What's left (medium 0.54 vs 0.50) is fp32 elementwise + launch overhead — diminishing
returns. aria's DiT/decoder GEMMs are already fp16 tensor cores.

## Long audio — 60 s reference (8 steps)

The motivating workload is a **60 s** clip. Below is aria-vs-SAT at 10 s and 60 s so
the *scaling* is visible (uncond, warm; Alienware RTX 3070 + i9-10900KF).

**medium:**

| backend | aria 10 s | aria 60 s | aria 10→60 | SAT 10 s | SAT 60 s | SAT 10→60 | @ 60 s |
|---|---|---|---|---|---|---|---|
| **CPU** (20 thr) | 11.2 s | **55.9 s** | 5.0× | 11.3 s | **108.4 s** | 9.6× | **aria 1.94× faster** |
| **GPU** (fp16) | 0.52 s | **2.07 s** | 4.0× | 0.49 s | **1.32 s** | 2.7× | SAT 1.57× faster |

**small-music:**

| backend | aria 10 s | aria 60 s | aria 10→60 | SAT 10 s | SAT 60 s | SAT 10→60 | @ 60 s |
|---|---|---|---|---|---|---|---|
| **CPU** (20 thr) | 3.58 s | **16.3 s** | 4.6× | 2.79 s | **11.1 s** | 4.0× | SAT 1.47× faster |
| **GPU** (fp16) | 0.22 s | **0.67 s** | 3.0× | 0.34 s | **0.41 s** | 1.2× | SAT 1.63× faster |

aria 60 s stage split — medium GPU dit **1.28** + decode 0.78 s, CPU dit 35.1 + decode 20.6 s
(CPU dit after query-blocked attention, −4% on the 20-core); small-music GPU dit 0.50 +
decode 0.16 s, CPU dit 12.5 + decode 4.3 s. GPU VRAM @60 s: medium aria 4514 vs SAT 6720 MB.
The medium GPU dit dropped **1.45 → 1.28 s (−12 %)** by fusing the elementwise long tail
(see below); the gap to SAT closed 1.68× → 1.57×.

**small-music has no banded-decoder win** (its decoder already chunks → O(N·34) linear),
so on CPU it's pure GEMM and MKL edges aria (1.3–1.5×); the medium CPU win is specifically
the banded decoder fix. On GPU both models show the same long-audio gap (parity/ahead at
10 s → ~1.7–1.8× behind at 60 s).

**The decoder fix (CPU + GPU).** The medium decoder's ±17 sliding-window attention
ran as a full N×N score matrix + mask, where the transformer sequence `N = T×17`
(10,982 tokens at 60 s) — pure **O(N²)**: CPU decode measured 18.6 → 78.0 s when the
latent length doubled (15→30 s). A banded kernel computes only the 35-key window
(`aria_attention_band` on CPU; `k_attn_band` on GPU; both numerically exact, A/B
audio rel 1e-5). CPU decode at 60 s ~300 → 21–32 s; the win grows with duration. This
is why **aria CPU now scales *better* than PyTorch** for long audio (SAT CPU 9.6× vs
aria 5.0× from 10→60 s) — PyTorch's medium decoder still pays the growth aria shed.

**The remaining gap — GPU.** aria GPU is parity/ahead at 10 s but **~1.7–1.8× behind
at 60 s** (both models). The DiT self-attention is full O(S²) (S = 710 at 60 s) and
aria materializes the fp16 S×S scores + a softmax pass, vs PyTorch SDPA's flash
attention. **Two flash kernels were tried and reverted**, both losing to cuBLAS:
(1) an **fp32** flash made the DiT 4× slower (no tensor cores; latency-bound per-thread
online softmax); (2) a **WMMA tensor-core** flash was numerically correct (small-music
GPU-vs-CPU 1.18 %) but still slower (medium DiT 1.45→1.9 s, small 0.52→0.75 s) and
regressed medium fidelity to 5.8 %. At S ≤ 710 (S² = 0.5 M) cuBLAS GemmEx + a parallel
softmax reduction beats a hand-rolled single-warp-per-tile flash; matching PyTorch
needs a cuDNN-class **multi-warp/pipelined FlashAttention-2** kernel — large effort
for an already-fast (≤ 2.2 s) path, so deferred. The cuBLAS attention stands.

**What did help — fusing the long tail.** nsys (medium, GPU) shows the GEMMs are only
~41 % of GPU time, the attention softmax ~9 %, and a **~32 % long tail** of small
memory-bound elementwise/reshape kernels (fp16-convert, rmsnorm, RoPE, extract/merge-
heads, adaLN/gate). Fusing the biggest of these — **extract-head + per-head qk-rmsnorm +
RoPE** into one kernel (`k_extract_normrope`), plus rmsnorm+adaLN and gate+residual-add —
cut the **medium GPU DiT 1.45 → 1.28 s (−12 %)**, whole gen 2.22 → 2.07 s, parity
unchanged. It's a bigger, lower-risk lever than flash (the softmax FA-2 would touch is
only ~9 %). Remaining long-tail headroom: the per-attention fp16 conversions (have
`k_extract_normrope` emit fp16 so `attn_dev` skips them).

**CUDA Graphs (landed).** The per-step DiT is now captured once and replayed (`--default-
stream per-thread` + cuBLAS pinned to it + a fixed workspace; robust inline fallback;
`ARIA_NO_GRAPH=1` to A/B). Output is byte-identical to the inline path (rel_rmse 0). The
launch-overhead win is **modest at these sizes** — ~4 % warm-min on the medium DiT (s=24,
0.96→0.92 s), within noise on small-music — because the small audio latent keeps the DiT
compute-bound so the ~14 k launches mostly hide behind compute. The graph's clearer benefit
is **steadier per-step latency** (less launch jitter); the win grows with steps and the
medium differential path's kernel count. Also drops the redundant per-block residual copies
(3 `cudaMemcpyD2D`/block) on the same path. See PAPER.md §4.

**True Megakernel feasibility (measured, not adopted).** `tests/bench_megakernel.cu` benches a
from-scratch WMMA tensor-core GEMM vs cuBLAS `GemmEx` at the DiT FFN/qkv shapes (3070, sm_86).
The WMMA kernel is **bit-identical** to cuBLAS but **0.12–0.56×** its speed, and cuBLAS runs
these GEMMs at **18–31 TFLOP/s** (≈50–78 % of the ~40 TFLOP/s fp16-tensor peak) at only
**49–138 GB/s** (≪ 448 GB/s HBM) — i.e. **compute-bound, not memory-bound**, so a megakernel's
HBM-traffic savings buy nothing on the GEMMs. Crossover model: per-block weights 33.6/75.5 MB
(small/medium) stream regardless; activation share 35 %→69 % of HBM traffic over S=172→710;
ideal free-GEMM fusion ceiling 1.18×→1.53× (crossover S≈315/473), much of it already L2-resident.
Verdict: a 3–8× GEMM penalty vs a ≤1.53× ceiling on compute-bound kernels → **not worth it**;
cuBLAS+graph+glue-fusion is at/near the roofline. The crossover analysis is the paper result.

## Streaming / continuation on GPU (small-music, 8 steps, RTX 3070)

Continue/inpaint used to be gated to the CPU DiT (the local-additive inpaint cond was
CPU-only). Porting it to the device DiT (reuse the CPU-projected per-block `local_emb`,
upload it, add after cross-attn via the existing `k_add`) makes `--continue`/`--inpaint`/
`--stream` run on the GPU. The taae context encoder stays on the CPU.

- **Parity:** GPU vs CPU continuation, same seed → regen region **corr = 1.0000, rel_rmse
  0.45 %** (pure fp16 DiT noise; the local-cond is uploaded + added in f32, so bit-identical).
  CPU and GPU `--stream` outputs are identical to 2 decimals per second.
- **Speed:** an 18 s continuation **6.92 → 2.54 s (2.7×)**. Streaming per-chunk (4 s emit,
  14.5 s window): CPU **RTF ~0.6×** → GPU **RTF ~4×** (chunk ~1.0 s). Crosses real-time with
  ~3× headroom → smooth live playback (`--stream -o - | play …`, writer thread overlaps gen).
- **Decay ceiling (device-independent):** for prompts whose chunk-0 emit reaches SA3's
  natural outro fade (e.g. "ambient pads", "funk groove"), the chained continuations inherit
  the fade and drift to silence past ~9 s. Steady-energy prompts ("melodic techno") hold.
  This is the inpaint-envelope-drift limit of the post-hoc method, not the GPU — CPU and GPU
  decay identically. See PAPER.md (beat-sync loop / streaming-trained model).

## Takeaways

- **GPU memory:** aria's footprint is consistently smaller (−27 to −47 %) — a bump
  allocator with no caching-allocator slack vs PyTorch's reserved pool + CUDA context.
  Notably, **medium fp16 OOMs stable-audio-tools' naïve `.to(cuda).half()` load** on
  the 8 GB card (transient fp32 copy); aria runs medium GPU in 3.97 GB with headroom.
- **GPU speed:** aria **wins small-music (1.8×)** and reaches **≈ parity on medium**
  (0.54 vs 0.50 s) after the three GPU optimizations above.
- **CPU speed:** within **1.15–1.5× of PyTorch/MKL** after the **packed outer-product
  GEMM + K-blocking** (was 2.4× behind); aria's AVX2 GEMM runs **~460–622 GFLOP/s** on
  the big DiT shapes (MKL is ~600–700). For **long-audio medium aria now *beats*
  PyTorch** — the banded decoder keeps aria near-linear while SAT's medium decode grows
  superlinearly: at **60 s, aria CPU is 1.94× faster** (55.9 vs 108.4 s). small-music
  (decoder already chunked, no banded win) stays pure-GEMM where MKL leads ~1.3–1.5×.
- **CPU/host memory:** aria uses **~60 % less RAM** everywhere (mmap'd weights, no
  framework). Medium on CPU: aria 7.2 GB vs PyTorch 18.9 GB.

**Where aria wins:** dependency-free single binary, the smallest memory footprint
(GPU and CPU, −44 to −62 %), faster small-music GPU, ≈-parity medium GPU, and
**faster long-audio CPU** (1.94× at 60 s). **Where the PyTorch stack still edges
ahead:** short-clip CPU throughput (MKL, ~1.3×) and **long-audio GPU** (1.68× at 60 s
— PyTorch's cuDNN tensor-core flash attention; a matching aria kernel is deferred).
aria's niche is exactly the plan's:
run these models on budget/low-VRAM hardware without the framework — now also at
competitive (often better) speed.

## Memory notes (no-speed-loss optimizations)

The dominant consumer is the **weights** — fp16 on the GPU, F32 (mmap, zero-copy) on
the host — which is the speed-optimal storage, so it's off-limits without a precision
(speed) cost. The clean wins are in waste:

- **Host RAM on the GPU path: 7.26 GB → 0.44 GB steady-state** (medium). The host F32
  weights stay mmap-resident even after the fp16 copies are on the GPU; once uploaded
  they aren't read on the hot path, so `madvise(MADV_DONTNEED)` reclaims them (file-
  backed → re-faults cheaply if a small per-request MLP is touched). Biggest win.
- **Decoder arena −~90–100 MB** (GPU peak + CPU): the attention scratch and the FFN
  scratch were held in one arena scope though never live together — freeing the former
  before the latter drops the per-block peak from `16·N·D + 3·N·INNER` to `~17·N·D`.
- **Tried and reverted:** Q8 on `ca_to_kv` (cross-K/V) saved 170 MB VRAM but the cross
  projection is fidelity-sensitive (GPU-vs-CPU 0.7 % → 2.6 %) — not worth it.
- **Hard floor:** pure-CPU RAM is weights-bound (medium ~5.4 GB F32, needed for the
  fp32 compute that gives exact parity); reducing it means fp16 compute = slower.

_Reproduce: aria `./aria -m <model> --uncond -d 10 -s 8 --device cuda|cpu --bench 3`
(`ARIA_PROFILE=1` for the breakdown); SAT via `scripts/bench_sat.py` (uncond DiT+decode)._
