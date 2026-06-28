/*
 * aria_steer.h - kind-agnostic activation-steering control (E12).
 *
 * Each steer adds a fixed calibrated direction to one activation site during generation.
 * The direction is just a vector, so the same mechanism steers ANYTHING (taste is one use
 * case). The op matches the reference injector (sa3-sf-api/steering/injector.py) exactly:
 * at the site, `x += scale * dir`, where `dir` is the alpha-less calibrated direction
 * (= mean_residual_norm * unit_direction, baked at export). Gated to the inclusive denoise-
 * step window [step_lo,step_hi] and, for the residual site, the target DiT layer. scale==0
 * is a bit-exact no-op. `dir` is caller-owned (not freed by the runtime).
 *
 * Shared between the public API (aria.h, where a set hangs off aria_gen_params) and the
 * DiT layer (aria_sa3_dit.c, which applies the residual-site hook).
 */
#ifndef ARIA_STEER_H
#define ARIA_STEER_H

typedef enum { ARIA_STEER_RESIDUAL = 0, ARIA_STEER_LATENT, ARIA_STEER_COND } aria_steer_site;

typedef struct {
    aria_steer_site site;   /* where to inject (residual = DiT block output, per layer) */
    int   layer;            /* DiT block index (site=residual); ignored otherwise */
    const float *dir;       /* [dim] calibrated direction; caller-owned */
    int   dim;              /* direction length (1024 for small-music residual) */
    float scale;            /* alpha multiplier; 0 = bit-exact no-op */
    int   step_lo, step_hi; /* inclusive denoise-step window the steer is active in */
} aria_steer;

typedef struct { const aria_steer *items; int n; } aria_steer_set;

#endif /* ARIA_STEER_H */
