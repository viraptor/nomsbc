#ifndef NOMSBC_WEIGHTS_H
#define NOMSBC_WEIGHTS_H

/*
 * Load model weights from the binary format produced by
 * training/export_weights.py.
 *
 * File format (little-endian):
 *   Magic:    'NMSB' (4 bytes)
 *   Version:  uint32 = 1
 *   Total:    uint32 (total float32 count)
 *   Layers:   uint32 (number of named layers)
 *     Per layer:
 *       name_len: uint32
 *       name:     UTF-8 string
 *       count:    uint32
 *   Weights:  float32[] (all concatenated in architecture order)
 */

#include <stdint.h>

#define NOMSBC_WEIGHTS_MAGIC  0x42534D4E  /* 'NMSB' little-endian */

typedef struct {
    char   *name;
    int     count;       /* number of float32 values */
    float  *data;        /* pointer into the contiguous weight buffer */
} NomsbcWeightLayer;

typedef struct {
    float             *data;         /* contiguous weight buffer (owned) */
    int                total_count;  /* total float32 values */
    NomsbcWeightLayer *layers;       /* per-layer metadata */
    int                num_layers;
} NomsbcWeights;

/*
 * Load weights from a binary file.  Returns NULL on failure.
 * The caller must free with nomsbc_weights_destroy().
 */
NomsbcWeights *nomsbc_weights_load(const char *path);

void nomsbc_weights_destroy(NomsbcWeights *w);

/*
 * Look up a layer by name.  Returns NULL if not found.
 */
const NomsbcWeightLayer *nomsbc_weights_find(const NomsbcWeights *w,
                                             const char *name);

#endif
