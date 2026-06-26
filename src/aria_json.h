/*
 * aria_json.h - Tiny read-only JSON value extractor for model configs.
 *
 * Not a full parser: enough to pull scalar fields out of model_config.json /
 * config.json by key name. Phase 1 extends this for nested DiT/AE configs.
 */

#ifndef ARIA_JSON_H
#define ARIA_JSON_H

#include <stddef.h>

/* Read an entire file into a NUL-terminated heap buffer (caller frees). */
char *aria_read_file(const char *path, size_t *out_len);

/* Find the first "key": "string" and copy the value. Returns 0 on success. */
int aria_json_get_string(const char *json, const char *key, char *out, size_t outlen);

/* Find the first "key": <number> and store as double. Returns 0 on success. */
int aria_json_get_number(const char *json, const char *key, double *out);

#endif /* ARIA_JSON_H */
