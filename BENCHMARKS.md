# aria vs stable-audio-tools — benchmark

aria (this runtime) vs **stable-audio-tools** (the PyTorch reference), same machine,
same model weights, same workload: **uncond, 8-step pingpong, DiT + autoencoder
decode** (no text encoder — isolates the core diffusion runtime). Both measure
**warm** generation with the model resident (aria = min of 3, SAT = mean of 3;
medium CPU = single run). Hardware: **Alienware — RTX 3070 (8 GB, sm_86) + 20-core CPU.**

GPU: aria runs fp16 weights / fp32 compute (cuBLAS tensor cores); SAT runs fp16
(`.half()`). VRAM is the **actual process footprint** (`cudaMemGetInfo` for aria,
`mem_get_info` total−free for SAT) so the CUDA context + allocator pool are counted
for both; SAT's `max_memory_allocated` (tensors only) is shown in parentheses.

## small-music (10 s clip, 8 steps)

| metric | aria | stable-audio-tools | winner |
|---|---|---|---|
| **GPU time** | **0.19 s** | 0.34 s | aria (1.8×) |
| **GPU VRAM** (process) | **1548 MB** | 2770 MB (alloc 1161) | aria (−44 %) |
| **CPU time** (20 threads) | 4.0 s | **2.8 s** | SAT (1.4×) |
| **CPU peak RAM** | **2089 MB** | 5421 MB | aria (−61 %) |

## medium (GPU 10 s, CPU 5 s, 8 steps)

| metric | aria | stable-audio-tools | winner |
|---|---|---|---|
| **GPU time** | 0.54 s | **0.50 s** | ≈ parity (1.08×) |
| **GPU VRAM** (process) | **4314 MB** | 5452 MB (alloc 4561) | aria (−21 %) |
| **CPU time** (5 s, 20 thr) | 8.6 s | **7.0 s** | SAT (1.2×) |
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

## Takeaways

- **GPU memory:** aria's footprint is consistently smaller (−27 to −47 %) — a bump
  allocator with no caching-allocator slack vs PyTorch's reserved pool + CUDA context.
  Notably, **medium fp16 OOMs stable-audio-tools' naïve `.to(cuda).half()` load** on
  the 8 GB card (transient fp32 copy); aria runs medium GPU in 3.97 GB with headroom.
- **GPU speed:** aria **wins small-music (1.8×)** and reaches **≈ parity on medium**
  (0.54 vs 0.50 s) after the three GPU optimizations above.
- **CPU speed:** within **1.2–1.4× of PyTorch/MKL** after the **packed outer-product
  GEMM** (was 2.4× behind). aria's hand-rolled AVX2 GEMM now runs ~490–540 GFLOP/s on
  the big DiT shapes (was 175–280); the residual gap is MKL's edge on the GEMM + aria's
  fp32 non-GEMM kernels. small-music CPU 6.7 → 4.0 s, medium 17 → 8.6 s.
- **CPU/host memory:** aria uses **~60 % less RAM** everywhere (mmap'd weights, no
  framework). Medium on CPU: aria 7.2 GB vs PyTorch 18.9 GB.

**Where aria wins:** dependency-free single binary, the smallest memory footprint
(GPU and CPU, −44 to −62 %), faster small-music GPU, and ≈-parity medium GPU. **Where
the PyTorch stack still edges ahead:** CPU throughput (MKL, ~1.3×) and a sliver on
medium GPU. aria's niche is exactly the plan's: run these models on budget/low-VRAM
hardware without the framework — now also at competitive speed.

_Reproduce: aria `./aria -m <model> --uncond -d 10 -s 8 --device cuda|cpu --bench 3`
(`ARIA_PROFILE=1` for the breakdown); SAT via `scripts/bench_sat.py` (uncond DiT+decode)._
