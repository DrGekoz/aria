#!/usr/bin/env bash
# E12 steering-mechanism verification. Usage: tests/steer_verify.sh <model_dir>
# RESIDUAL site (1024-D, E12.3):
#   GATE1  add scale-0 == unsteered (byte-identical)      -> the hook is a bit-exact no-op
#   GATE2  add steered != unsteered                       -> the steer has a deterministic effect
#   GATE3  add(2a,d) == add(a,2d) (byte-identical)        -> ADD op is exactly scale*dir (linear)
#   GATE4  project scale-0 == unsteered (byte-identical)  -> PROJECT is also a bit-exact no-op
#   GATE5  project(a,d) == project(a,2d) (byte-identical) -> PROJECT uses the unit (||dir||-invariant)
# LATENT site (256-D, E12.2 -- host-side x[256,T] between pingpong steps, CPU+GPU):
#   GATE6  add scale-0 == unsteered (byte-identical)      -> the between-steps hook is a no-op
#   GATE7  add steered != unsteered                       -> deterministic latent effect
#   GATE8  add(2a,d) == add(a,2d) (byte-identical)        -> latent ADD is exactly scale*dir
# COND/text site (768-D, E12.4 -- prompt embedding rows before to_cond_embed, no step window):
#   GATE9  add scale-0 == unsteered (byte-identical)      -> the cond hook is a no-op
#   GATE10 add steered != unsteered                       -> deterministic cond effect
# (No add-linearity gate for cond: the steer is linear in the embedding but to_cond_embed +
#  cross-attn are nonlinear, so (2a,d) and (a,2d) need not match at the WAV. Latent keeps it
#  because both inject the identical delta into x before the same downstream.)
# Runs single-threaded so byte-identical comparisons aren't flaky under OMP reduction-order
# variance. Builds synthetic directions of each site's dim (no model/dataset needed).
set -e
M="$1"; [ -z "$M" ] && { echo "usage: $0 <model_dir>"; exit 1; }
ARIA="${ARIA:-./aria}"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
python3 - "$T" <<'PY'
import struct, math, sys
def mkdir_vec(dim):
    d=[0.3*math.sin(0.1*i)+0.2*math.cos(0.03*i) for i in range(dim)]
    n=math.sqrt(sum(x*x for x in d)); return [x/n*20.0 for x in d]   # unit dir * norm 20
def w(p,v): open(p,'wb').write(b'ATNS'+struct.pack('<III',1,1,0)+struct.pack('<q',len(v))+struct.pack('<%df'%len(v),*v))
for dim,tag in ((1024,''),(256,'256'),(768,'768')):
    d=mkdir_vec(dim)
    w('%s/d%s.atns'%(sys.argv[1],tag), d); w('%s/2d%s.atns'%(sys.argv[1],tag), [2*x for x in d])
PY
# DEVICE=cpu (default) | cuda. ARIA_NO_GRAPH keeps the unsteered baseline on the same inline
# path the steered runs use (steering auto-disables the graph), so GATE1 is byte-clean on GPU.
A="env OMP_NUM_THREADS=1 ARIA_NO_GRAPH=1 $ARIA -m $M -p techno -d 4 -s 8 --seed 7 --device ${DEVICE:-cpu}"
$A -o "$T/uns.wav"  >/dev/null 2>&1
$A --steer residual:16:"$T/d.atns":0.0:0-7  -o "$T/s0.wav"   >/dev/null 2>&1
$A --steer residual:16:"$T/d.atns":0.2:0-7  -o "$T/sa.wav"   >/dev/null 2>&1
$A --steer residual:16:"$T/d.atns":0.4:0-7  -o "$T/s2a.wav"  >/dev/null 2>&1
$A --steer residual:16:"$T/2d.atns":0.2:0-7 -o "$T/sa2d.wav" >/dev/null 2>&1
$A --steer residual:16:"$T/d.atns":0.0:0-7:project  -o "$T/pp0.wav"  >/dev/null 2>&1
$A --steer residual:16:"$T/d.atns":0.2:0-7:project  -o "$T/ppd.wav"  >/dev/null 2>&1
$A --steer residual:16:"$T/2d.atns":0.2:0-7:project -o "$T/pp2d.wav" >/dev/null 2>&1
# latent (256-D). layer field is ignored for this site (pass 0).
$A --steer latent:0:"$T/d256.atns":0.0:0-7  -o "$T/l0.wav"   >/dev/null 2>&1
$A --steer latent:0:"$T/d256.atns":0.2:0-7  -o "$T/la.wav"   >/dev/null 2>&1
$A --steer latent:0:"$T/d256.atns":0.4:0-7  -o "$T/l2a.wav"  >/dev/null 2>&1
$A --steer latent:0:"$T/2d256.atns":0.2:0-7 -o "$T/la2d.wav" >/dev/null 2>&1
# cond (768-D). layer + step-window are ignored for this site (pass 0 / 0-7).
$A --steer cond:0:"$T/d768.atns":0.0:0-7 -o "$T/c0.wav"  >/dev/null 2>&1
$A --steer cond:0:"$T/d768.atns":0.2:0-7 -o "$T/ca.wav"  >/dev/null 2>&1
fail=0
cmp -s "$T/uns.wav" "$T/s0.wav"  && echo "GATE1 residual add scale-0==unsteered: PASS"     || { echo "GATE1: FAIL"; fail=1; }
cmp -s "$T/uns.wav" "$T/sa.wav"  && { echo "GATE2 effect: FAIL"; fail=1; }                 || echo "GATE2 residual add steered!=unsteered: PASS"
cmp -s "$T/s2a.wav" "$T/sa2d.wav" && echo "GATE3 residual add linearity (exactly scale*dir): PASS" || { echo "GATE3: FAIL"; fail=1; }
cmp -s "$T/uns.wav" "$T/pp0.wav" && echo "GATE4 residual project scale-0==unsteered: PASS" || { echo "GATE4: FAIL"; fail=1; }
cmp -s "$T/ppd.wav" "$T/pp2d.wav" && echo "GATE5 residual project ||dir||-invariant: PASS" || { echo "GATE5: FAIL"; fail=1; }
cmp -s "$T/uns.wav" "$T/l0.wav"  && echo "GATE6 latent add scale-0==unsteered: PASS"       || { echo "GATE6: FAIL"; fail=1; }
cmp -s "$T/uns.wav" "$T/la.wav"  && { echo "GATE7 latent effect: FAIL"; fail=1; }          || echo "GATE7 latent add steered!=unsteered: PASS"
cmp -s "$T/l2a.wav" "$T/la2d.wav" && echo "GATE8 latent add linearity (exactly scale*dir): PASS" || { echo "GATE8: FAIL"; fail=1; }
cmp -s "$T/uns.wav" "$T/c0.wav"  && echo "GATE9 cond add scale-0==unsteered: PASS"         || { echo "GATE9: FAIL"; fail=1; }
cmp -s "$T/uns.wav" "$T/ca.wav"  && { echo "GATE10 cond effect: FAIL"; fail=1; }           || echo "GATE10 cond add steered!=unsteered: PASS"
[ $fail -eq 0 ] && echo "steer-verify: ALL PASS" || echo "steer-verify: FAILED"
exit $fail
