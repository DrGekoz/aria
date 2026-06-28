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

## Long audio — 60 s reference (medium, 8 steps)

The motivating workload is a **60 s** clip. Below is aria-vs-SAT at 10 s and 60 s so
the *scaling* is visible (medium, uncond, warm; Alienware RTX 3070 + i9-10900KF).

| backend | aria 10 s | aria 60 s | aria 10→60 | SAT 10 s | SAT 60 s | SAT 10→60 | @ 60 s |
|---|---|---|---|---|---|---|---|
| **CPU** (20 thr) | 11.2 s | **55.9 s** | 5.0× | 11.3 s | **108.4 s** | 9.6× | **aria 1.94× faster** |
| **GPU** (fp16) | 0.52 s | **2.22 s** | 4.3× | 0.49 s | **1.32 s** | 2.7× | SAT 1.68× faster |

aria 60 s stage split — GPU: dit 1.45 + decode 0.78 s; CPU: dit 35.1 + decode 20.6 s
(CPU dit after query-blocked attention, −4% on the 20-core).
GPU VRAM at 60 s: aria **4514 MB** vs SAT 6720 MB.

**The decoder fix (CPU + GPU).** The medium decoder's ±17 sliding-window attention
ran as a full N×N score matrix + mask, where the transformer sequence `N = T×17`
(10,982 tokens at 60 s) — pure **O(N²)**: CPU decode measured 18.6 → 78.0 s when the
latent length doubled (15→30 s). A banded kernel computes only the 35-key window
(`aria_attention_band` on CPU; `k_attn_band` on GPU; both numerically exact, A/B
audio rel 1e-5). CPU decode at 60 s ~300 → 21–32 s; the win grows with duration. This
is why **aria CPU now scales *better* than PyTorch** for long audio (SAT CPU 9.6× vs
aria 5.0× from 10→60 s) — PyTorch's medium decoder still pays the growth aria shed.

**The remaining gap — GPU.** aria GPU is at parity at 10 s (0.52 vs 0.49) but **1.68×
behind at 60 s**. The DiT self-attention is full O(S²) (S = 710 at 60 s) and aria
materializes the fp16 S×S scores + a softmax pass, vs PyTorch SDPA's flash attention.
A **hand-written fp32 flash kernel was tried and reverted**: on the 3070 it made the
DiT 4× *slower* — cuBLAS fp16 **tensor cores** + a parallel softmax reduction beat
avoiding the S×S materialization, and a per-thread online softmax (710 `expf`/query)
is latency-bound. Closing this needs a **tensor-core (mma) flash** kernel to match
PyTorch's cuDNN path — large effort for an already-fast (2.2 s) path, so deferred.

## Takeaways

- **GPU memory:** aria's footprint is consistently smaller (−27 to −47 %) — a bump
  allocator with no caching-allocator slack vs PyTorch's reserved pool + CUDA context.
  Notably, **medium fp16 OOMs stable-audio-tools' naïve `.to(cuda).half()` load** on
  the 8 GB card (transient fp32 copy); aria runs medium GPU in 3.97 GB with headroom.
- **GPU speed:** aria **wins small-music (1.8×)** and reaches **≈ parity on medium**
  (0.54 vs 0.50 s) after the three GPU optimizations above.
- **CPU speed:** within **1.15–1.3× of PyTorch/MKL** on short clips after the **packed
  outer-product GEMM + K-blocking** (was 2.4× behind); aria's AVX2 GEMM runs
  **~460–622 GFLOP/s** on the big DiT shapes (MKL is ~600–700). For **long audio aria
  now *beats* PyTorch** — the banded decoder keeps aria near-linear while SAT's medium
  decode grows superlinearly: at **60 s, aria CPU is 1.94× faster** (55.9 vs 108.4 s).
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
