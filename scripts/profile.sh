#!/usr/bin/env bash
# aria profiling helper — find compute bottlenecks + memory usage.
#
#   scripts/profile.sh <aria-binary> -m <model> [aria args...]
#   e.g.  scripts/profile.sh ./aria -m "$MODEL" --uncond -d 10 -s 8 --device cuda
#
# Layers:
#   1. aria's built-in per-stage timing + peak RSS / GPU VRAM (ARIA_PROFILE=1) —
#      always printed, no external tools needed.
#   2. nsys (CUDA build, if installed): per-kernel GPU time + host<->device
#      transfer time — shows which kernels to optimize.
#   3. perf (CPU build, if installed): cycles / cache / IPC summary.
set -u
BIN="${1:?usage: scripts/profile.sh <aria-binary> -m <model> [aria args...]}"; shift
ARGS=("$@")
OUT="${ARIA_PROFILE_OUT:-/tmp/aria_prof}"

is_cuda() { ldd "$BIN" 2>/dev/null | grep -qi 'libcudart'; }

echo "== built-in profile (stages + memory) =="
ARIA_PROFILE=1 "$BIN" "${ARGS[@]}" 2>&1 | grep -E 'profile:|DiT:|decoder:|wrote' || true

if is_cuda && command -v nsys >/dev/null 2>&1; then
    echo; echo "== nsys: GPU kernel + transfer breakdown =="
    nsys profile -o "$OUT" -f true "$BIN" "${ARGS[@]}" >/dev/null 2>&1
    echo "-- top GPU kernels by time --"
    nsys stats --force-export=true --report cuda_gpu_kern_sum "$OUT.nsys-rep" 2>/dev/null \
        | sed -n '/Time (%)/,/^[[:space:]]*$/p' | head -18
    echo "-- host<->device transfers --"
    nsys stats --report cuda_gpu_mem_time_sum "$OUT.nsys-rep" 2>/dev/null \
        | sed -n '/Time (%)/,/^[[:space:]]*$/p' | head -10
    echo "(full report: $OUT.nsys-rep — open in Nsight Systems for the timeline)"
elif is_cuda; then
    echo; echo "(CUDA build but 'nsys' not found — install Nsight Systems for kernel profiling)"
elif command -v perf >/dev/null 2>&1; then
    echo; echo "== perf stat (CPU) =="
    perf stat -d "$BIN" "${ARGS[@]}" 2>&1 | tail -24
else
    echo; echo "(install 'perf' for CPU hardware-counter profiling)"
fi
