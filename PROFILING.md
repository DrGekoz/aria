# Profiling aria

Two layers, from "where's the time + memory" to "which kernel to optimize".

## 1. Built-in — stages + memory (no external tools)

Set `ARIA_PROFILE=1` on any run; aria prints, to stderr, the per-stage wall time
plus peak host RSS and (CUDA build) resident GPU VRAM:

```
$ ARIA_PROFILE=1 ./aria -m "$MODEL" --uncond -d 10 -s 8 --device cuda -o out.wav
[aria] profile: setup=0.10s dit=0.16s decode=0.03s (T=108 steps=8) | peak RSS 2303 MB | GPU 1464/7832 MB
```

- **setup** = per-request conditioning + (GPU) weight upload · **dit** = the
  denoise loop · **decode** = taae decoder.
- **peak RSS** = `getrusage` high-water mark (the mmap'd weights fault in here).
- **GPU x/y MB** = `cudaMemGetInfo` used/total at the end of the run.

Pair it with `--bench N` to time the **warm** steady state (model resident),
which is the number to compare against PyTorch.

## 2. Deep — per-kernel GPU time (`make profile`)

```
make profile ARGS="-m $MODEL --uncond -d 10 -s 8 --device cuda -o /tmp/p.wav"
```

`scripts/profile.sh` runs the built-in profile, then — on a CUDA build with
**Nsight Systems (`nsys`)** installed — the per-kernel GPU-time and host↔device
transfer breakdown. On a CPU build it falls back to `perf stat`. Example (3070):
the cuBLAS GEMMs (`ampere_*gemm_*`) dominate (~45 %), then
`k_softmax` 9 %, `k_f32_to_f16` 6 %, `k_rope` 4 %, `k_conv1d` 3 % — the next
optimization targets. (The full `.nsys-rep` opens in the Nsight Systems GUI for a
timeline.)

For the CPU GEMM specifically, `make bench` micro-benchmarks the dominant shapes
and `make bench-sweep` tunes the AVX2 register tile.

## Optimization workflow

1. `--bench N` + `ARIA_PROFILE=1` → which **stage** and how much **memory**.
2. `make profile` → which **kernel** dominates that stage.
3. Optimize the top kernel; re-profile. (This is how we found a fp64
   thread-per-row `k_rmsnorm` eating 51 % of GPU time — invisible at the stage level.)
