#!/usr/bin/env bash
# aria <-> sf-api steering comparison harness: run the SAME residual steer through both the
# aria C runtime and the sf-api PyTorch reference, on identical math (delta = alpha*norm*unit;
# aria bakes dir=norm*unit and uses scale=alpha). Reports per-side timing + that each steers.
#
# Usage: tests/steer_compare.sh <aria_model_dir> [sf_api_dir] [sf_model_name]
set -e
ARIA_MODEL="$1"; [ -z "$ARIA_MODEL" ] && { echo "usage: $0 <aria_model_dir> [sf_api_dir] [sf_model_name]"; exit 1; }
SF_DIR="$2"; [ -z "$SF_DIR" ] && { echo "usage: $0 <aria_model_dir> <sf_api_dir> [sf_model_name]"; exit 1; }
SF_MODEL="${3:-stabilityai/stable-audio-3-small-music}"
ARIA="${ARIA:-./aria}"
PY="$SF_DIR/.venv/bin/python"
PROMPT=techno; DUR=4; SEED=7; LAYER=16; ALPHA=0.2; NORM=20
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
el() { awk "BEGIN{printf \"%.2f\", $2-$1}"; }   # elapsed seconds

# shared direction: a fixed unit [1024]; aria .atns bakes norm*unit, python loads unit + norm
"$PY" - "$T" "$NORM" <<'PY'
import sys, struct, numpy as np
T, norm = sys.argv[1], float(sys.argv[2])
v = np.random.default_rng(0).standard_normal(1024).astype(np.float32); unit = v/np.linalg.norm(v)
np.save(T+'/unit.npy', unit)
d = (norm*unit).astype(np.float32)
open(T+'/d.atns','wb').write(b'ATNS'+struct.pack('<III',1,1,0)+struct.pack('<q',1024)+d.tobytes())
PY

echo "== aria (C runtime, CPU) =="
t0=$(date +%s.%N); env OMP_NUM_THREADS=8 $ARIA -m "$ARIA_MODEL" -p $PROMPT -d $DUR -s 8 --seed $SEED --device cpu \
    -o "$T/aria_base.wav" >/dev/null 2>&1; t1=$(date +%s.%N)
t2=$(date +%s.%N); env OMP_NUM_THREADS=8 $ARIA -m "$ARIA_MODEL" -p $PROMPT -d $DUR -s 8 --seed $SEED --device cpu \
    --steer residual:$LAYER:"$T/d.atns":$ALPHA:0-7 -o "$T/aria_steer.wav" >/dev/null 2>&1; t3=$(date +%s.%N)
echo "  base=$(el $t0 $t1)s steered=$(el $t2 $t3)s"
cmp -s "$T/aria_base.wav" "$T/aria_steer.wav" && echo "  aria: steered==base (NO EFFECT)" || echo "  aria: steered!=base (effect OK)"

echo "== sf-api (PyTorch reference) =="
( cd "$SF_DIR" && "$PY" experiments/aria_steer_compare.py --model "$SF_MODEL" --prompt $PROMPT \
    --duration $DUR --seed $SEED --layer $LAYER --alpha $ALPHA --unit-npy "$T/unit.npy" --norm $NORM \
    --out-base "$T/sf_base.wav" --out-steer "$T/sf_steer.wav" )
cmp -s "$T/sf_base.wav" "$T/sf_steer.wav" && echo "  sf-api: steered==base (NO EFFECT)" || echo "  sf-api: steered!=base (effect OK)"
echo "Both apply delta = alpha*norm*unit ($ALPHA*$NORM*unit) at layer $LAYER -- matched steering math."
