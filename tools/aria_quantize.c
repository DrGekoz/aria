/* aria-quantize - offline DiT weight quantizer (E9.2).
 *
 *   aria-quantize <model_dir> <out.aria> [q8|q4]
 *
 * Reads <model_dir>/model.safetensors, quantizes the DiT per-step block GEMMs to
 * the chosen precision (default q4 = asymmetric int4 FFN + Q8 attention), and
 * writes a packed .aria overlay. aria loads it with `--load-quant <out.aria>`,
 * skipping the on-the-fly quantization (the generation is bit-identical). */
#include "../src/aria_safetensors.h"
#include "../src/aria_sa3.h"
#include "../src/aria_sa3_dit.h"
#include "../src/aria_json.h"
#include "../src/aria_quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model_dir> <out.aria> [q8|q4]\n", argv[0]);
        return 1;
    }
    const char *model_dir = argv[1], *out = argv[2];
    aria_dtype dt = ARIA_Q4;
    if (argc > 3 && aria_dtype_parse(argv[3], &dt) != 0) { fprintf(stderr, "bad precision %s\n", argv[3]); return 1; }
    if (dt != ARIA_Q8 && dt != ARIA_Q4) { fprintf(stderr, "precision must be q8 or q4\n"); return 1; }

    char path[1024];
    snprintf(path, sizeof(path), "%s/model_config.json", model_dir);
    char *cfgjson = aria_read_file(path, NULL);
    aria_sa3_config cfg;
    if (!cfgjson || aria_sa3_parse_config(cfgjson, &cfg) != 0) { fprintf(stderr, "cannot read/parse %s\n", path); return 1; }
    free(cfgjson);

    snprintf(path, sizeof(path), "%s/model.safetensors", model_dir);
    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    aria_sa3_dit *dit = aria_sa3_dit_load(sf, &cfg);
    if (!dit) { fprintf(stderr, "DiT load failed\n"); return 1; }

    if (aria_sa3_dit_quant_save(dit, dt, out) != 0) { fprintf(stderr, "quantize/write failed\n"); return 1; }

    struct stat st; double mb = (stat(out, &st) == 0) ? st.st_size / 1048576.0 : 0;
    printf("wrote %s : DiT %s, %.0f MB (%.0f MB resident, %d blocks, ed=%d)\n",
           out, aria_dtype_name(dt), mb, aria_sa3_dit_weight_bytes(dit) / 1048576.0,
           cfg.depth, cfg.embed_dim);

    aria_sa3_dit_free(dit);
    safetensors_close(sf);
    return 0;
}
