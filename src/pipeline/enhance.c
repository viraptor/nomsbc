#include "pipeline/enhance.h"
#include <stdlib.h>
#include <string.h>

struct NomsbcEnhancer {
    /* Feature extraction */
    NomsbcFeatures *features;

    /* Processing modules */
    NomsbcAdaComb  *comb1;
    NomsbcAdaComb  *comb2;
    NomsbcAdaConv  *conv;
    NomsbcAdaShape *shape1;
    NomsbcAdaShape *shape2;
    NomsbcBWE      *bwe;

    /* DNN inference */
    NomsbcDNNInferFn infer_fn;
    void            *user_data;
};

NomsbcEnhancer *nomsbc_enhancer_create(NomsbcDNNInferFn infer_fn,
                                        void *user_data)
{
    NomsbcEnhancer *e = calloc(1, sizeof(*e));
    if (!e) return NULL;

    e->features = nomsbc_features_create();
    e->comb1    = nomsbc_adacomb_create(NOMSBC_FRAME_SIZE);
    e->comb2    = nomsbc_adacomb_create(NOMSBC_FRAME_SIZE);
    e->conv     = nomsbc_adaconv_create(NOMSBC_FRAME_SIZE);
    e->shape1   = nomsbc_adashape_create(NOMSBC_FRAME_SIZE);
    e->shape2   = nomsbc_adashape_create(NOMSBC_FRAME_SIZE);
    e->bwe      = nomsbc_bwe_create(NOMSBC_FRAME_SIZE);

    e->infer_fn  = infer_fn;
    e->user_data = user_data;

    if (!e->features || !e->comb1 || !e->comb2 || !e->conv ||
        !e->shape1 || !e->shape2 || !e->bwe) {
        nomsbc_enhancer_destroy(e);
        return NULL;
    }

    return e;
}

void nomsbc_enhancer_destroy(NomsbcEnhancer *e)
{
    if (!e) return;
    nomsbc_features_destroy(e->features);
    nomsbc_adacomb_destroy(e->comb1);
    nomsbc_adacomb_destroy(e->comb2);
    nomsbc_adaconv_destroy(e->conv);
    nomsbc_adashape_destroy(e->shape1);
    nomsbc_adashape_destroy(e->shape2);
    nomsbc_bwe_destroy(e->bwe);
    free(e);
}

void nomsbc_enhancer_process_frame(NomsbcEnhancer *e,
                                    const float *input,
                                    float *output)
{
    /* 1. Extract blind features */
    float feat[NOMSBC_FEATURE_DIM];
    int pitch_lag;
    nomsbc_features_extract(e->features, input, feat, &pitch_lag);

    /* 2. DNN inference: features -> module parameters */
    EnhanceFrameParams params;
    e->infer_fn(feat, NOMSBC_FEATURE_DIM, &params, e->user_data);

    /* 3-4. Adaptive comb filtering (two passes) */
    float signal[NOMSBC_FRAME_SIZE];
    memcpy(signal, input, NOMSBC_FRAME_SIZE * sizeof(float));

    nomsbc_adacomb_process(e->comb1, signal, NOMSBC_FRAME_SIZE,
                           pitch_lag, &params.comb1);
    nomsbc_adacomb_process(e->comb2, signal, NOMSBC_FRAME_SIZE,
                           pitch_lag, &params.comb2);

    /* 5. Adaptive convolution (spectral reshaping) */
    nomsbc_adaconv_process(e->conv, signal, NOMSBC_FRAME_SIZE, &params.conv);

    /* 6-7. Adaptive temporal shaping (two iterations) */
    nomsbc_adashape_process(e->shape1, signal, NOMSBC_FRAME_SIZE,
                            pitch_lag, &params.shape1);
    nomsbc_adashape_process(e->shape2, signal, NOMSBC_FRAME_SIZE,
                            pitch_lag, &params.shape2);

    /* 8. Bandwidth extension */
    float highband[NOMSBC_FRAME_SIZE];
    nomsbc_bwe_synthesize(e->bwe, signal, highband,
                          NOMSBC_FRAME_SIZE, pitch_lag, &params.bwe);

    /* 9. Combine */
    nomsbc_bwe_combine(signal, highband, output, NOMSBC_FRAME_SIZE);
}
