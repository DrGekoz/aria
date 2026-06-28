#!/usr/bin/env bash
# E12 steering-mechanism verification (3 gates). Usage: tests/steer_verify.sh <model_dir>
#   GATE1  scale-0 == unsteered (byte-identical)      -> the hook is a bit-exact no-op
#   GATE2  steered != unsteered                       -> the steer has a deterministic effect
#   GATE3  steer(2a,d) == steer(a,2d) (byte-identical) -> the op is exactly scale*dir (linear)
# Runs single-threaded so byte-identical comparisons aren't flaky under OMP reduction-order
# variance. Builds a synthetic [1024] direction (no model/dataset needed).
set -e
M="$1"; [ -z "$M" ] && { echo "usage: $0 <model_dir>"; exit 1; }
ARIA="${ARIA:-./aria}"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
python3 - "$T" <<'PY'
import struct, math, sys
d=[0.3*math.sin(0.1*i)+0.2*math.cos(0.03*i) for i in range(1024)]
n=math.sqrt(sum(x*x for x in d)); d=[x/n*20.0 for x in d]          # unit dir * norm 20
def w(p,v): open(p,'wb').write(b'ATNS'+struct.pack('<III',1,1,0)+struct.pack('<q',len(v))+struct.pack('<%df'%len(v),*v))
w(sys.argv[1]+'/d.atns', d); w(sys.argv[1]+'/2d.atns', [2*x for x in d])
PY
A="env OMP_NUM_THREADS=1 $ARIA -m $M -p techno -d 4 -s 8 --seed 7 --device cpu"
$A -o "$T/uns.wav"  >/dev/null 2>&1
$A --steer residual:16:"$T/d.atns":0.0:0-7  -o "$T/s0.wav"   >/dev/null 2>&1
$A --steer residual:16:"$T/d.atns":0.2:0-7  -o "$T/sa.wav"   >/dev/null 2>&1
$A --steer residual:16:"$T/d.atns":0.4:0-7  -o "$T/s2a.wav"  >/dev/null 2>&1
$A --steer residual:16:"$T/2d.atns":0.2:0-7 -o "$T/sa2d.wav" >/dev/null 2>&1
fail=0
cmp -s "$T/uns.wav" "$T/s0.wav"  && echo "GATE1 scale-0==unsteered: PASS"               || { echo "GATE1: FAIL"; fail=1; }
cmp -s "$T/uns.wav" "$T/sa.wav"  && { echo "GATE2 effect: FAIL"; fail=1; }              || echo "GATE2 steered!=unsteered: PASS"
cmp -s "$T/s2a.wav" "$T/sa2d.wav" && echo "GATE3 linearity (exactly scale*dir): PASS"   || { echo "GATE3: FAIL"; fail=1; }
[ $fail -eq 0 ] && echo "steer-verify: ALL PASS" || echo "steer-verify: FAILED"
exit $fail
