/* test_config.c - hermetic: parse a trimmed SA3 model_config and check scoping.
 * The pretransform section has io_channels:2 and the diffusion section has
 * io_channels:256 -- the parser must return 256 for the DiT. */
#include "../src/aria_sa3.h"
#include <stdio.h>
#include <math.h>

static const char *CONFIG =
"{"
"  \"model_type\": \"diffusion_cond_inpaint\","
"  \"sample_rate\": 44100,"
"  \"model\": {"
"    \"pretransform\": {"
"      \"type\": \"autoencoder\","
"      \"config\": {"
"        \"pretransform\": { \"type\": \"patched\", \"config\": { \"patch_size\": 256, \"channels\": 2 } },"
"        \"encoder\": { \"type\": \"taae_v2\", \"config\": { \"latent_dim\": 256, \"dim_heads\": 64 } },"
"        \"latent_dim\": 256,"
"        \"downsampling_ratio\": 4096,"
"        \"io_channels\": 2"
"      }"
"    },"
"    \"conditioning\": { \"cond_dim\": 768 },"
"    \"diffusion\": {"
"      \"type\": \"dit\","
"      \"config\": {"
"        \"io_channels\": 256,"
"        \"embed_dim\": 1024,"
"        \"depth\": 20,"
"        \"num_heads\": 16,"
"        \"cond_token_dim\": 768,"
"        \"global_cond_dim\": 768,"
"        \"local_add_cond_dim\": 257,"
"        \"attn_kwargs\": { \"qk_norm\": \"rms\", \"differential\": false },"
"        \"ff_kwargs\": { \"mult\": 4.0 },"
"        \"num_memory_tokens\": 64"
"      }"
"    },"
"    \"io_channels\": 256"
"  }"
"}";

static int fails = 0;
static void eq_i(const char *name, int got, int want) {
    if (got != want) { printf("FAIL %s: got %d want %d\n", name, got, want); fails++; }
}

int main(void) {
    aria_sa3_config c;
    if (aria_sa3_parse_config(CONFIG, &c) != 0) { printf("FAIL: parse returned error\n"); return 1; }

    eq_i("io_channels", c.io_channels, 256);   /* must NOT pick the AE's 2 */
    eq_i("embed_dim", c.embed_dim, 1024);
    eq_i("depth", c.depth, 20);
    eq_i("num_heads", c.num_heads, 16);
    eq_i("head_dim", c.head_dim, 64);
    eq_i("cond_token_dim", c.cond_token_dim, 768);
    eq_i("global_cond_dim", c.global_cond_dim, 768);
    eq_i("local_add_cond_dim", c.local_add_cond_dim, 257);
    eq_i("num_memory_tokens", c.num_memory_tokens, 64);
    eq_i("qk_norm_rms", c.qk_norm_rms, 1);
    eq_i("differential_attn", c.differential_attn, 0);
    eq_i("latent_dim", c.latent_dim, 256);
    eq_i("downsampling_ratio", c.downsampling_ratio, 4096);
    eq_i("patch_size", c.patch_size, 256);
    if (fabsf(c.ff_mult - 4.0f) > 1e-6f) { printf("FAIL ff_mult: got %f\n", c.ff_mult); fails++; }

    if (fails) { printf("test_config: %d failures\n", fails); return 1; }
    printf("test_config: OK\n");
    return 0;
}
