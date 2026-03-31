#include "darwin/weights.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_u32(FILE *f, uint32_t *out)
{
    unsigned char buf[4];
    if (fread(buf, 1, 4, f) != 4) return -1;
    *out = (uint32_t)buf[0]
         | ((uint32_t)buf[1] << 8)
         | ((uint32_t)buf[2] << 16)
         | ((uint32_t)buf[3] << 24);
    return 0;
}

NomsbcWeights *nomsbc_weights_load(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "nomsbc: cannot open weights file: %s\n", path);
        return NULL;
    }

    uint32_t magic, version, total, num_layers;
    if (read_u32(f, &magic) || magic != NOMSBC_WEIGHTS_MAGIC) {
        fprintf(stderr, "nomsbc: bad magic in %s\n", path);
        fclose(f);
        return NULL;
    }
    if (read_u32(f, &version) || version != 1) {
        fprintf(stderr, "nomsbc: unsupported weight version %u\n", version);
        fclose(f);
        return NULL;
    }
    if (read_u32(f, &total) || read_u32(f, &num_layers)) {
        fprintf(stderr, "nomsbc: truncated header in %s\n", path);
        fclose(f);
        return NULL;
    }

    NomsbcWeights *w = calloc(1, sizeof(*w));
    if (!w) { fclose(f); return NULL; }

    w->total_count = (int)total;
    w->num_layers  = (int)num_layers;
    w->layers = calloc(num_layers, sizeof(NomsbcWeightLayer));
    if (!w->layers) goto fail;

    /* Read layer table */
    for (uint32_t i = 0; i < num_layers; i++) {
        uint32_t name_len, count;
        if (read_u32(f, &name_len)) goto fail;

        char *name = malloc(name_len + 1);
        if (!name) goto fail;
        if (fread(name, 1, name_len, f) != name_len) { free(name); goto fail; }
        name[name_len] = '\0';

        if (read_u32(f, &count)) { free(name); goto fail; }

        w->layers[i].name  = name;
        w->layers[i].count = (int)count;
    }

    /* Read contiguous weight data */
    w->data = malloc((size_t)total * sizeof(float));
    if (!w->data) goto fail;
    if (fread(w->data, sizeof(float), total, f) != total) goto fail;

    fclose(f);

    /* Assign data pointers into contiguous buffer */
    float *ptr = w->data;
    for (int i = 0; i < w->num_layers; i++) {
        w->layers[i].data = ptr;
        ptr += w->layers[i].count;
    }

    return w;

fail:
    fclose(f);
    nomsbc_weights_destroy(w);
    return NULL;
}

void nomsbc_weights_destroy(NomsbcWeights *w)
{
    if (!w) return;
    if (w->layers) {
        for (int i = 0; i < w->num_layers; i++)
            free(w->layers[i].name);
        free(w->layers);
    }
    free(w->data);
    free(w);
}

const NomsbcWeightLayer *nomsbc_weights_find(const NomsbcWeights *w,
                                             const char *name)
{
    for (int i = 0; i < w->num_layers; i++) {
        if (strcmp(w->layers[i].name, name) == 0)
            return &w->layers[i];
    }
    return NULL;
}
