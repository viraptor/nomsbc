#ifndef NOMSBC_ENHANCE_H
#define NOMSBC_ENHANCE_H

/*
 * Main enhancement pipeline: NoLACE-mSBC.
 *
 * Signal flow per 10 ms frame:
 *   1. Feature extraction (blind, from decoded mSBC)
 *   2. DNN inference -> module parameters
 *   3. AdaComb #1 (pitch cleanup, coarse)
 *   4. AdaComb #2 (pitch cleanup, fine)
 *   5. AdaConv (spectral reshaping)
 *   6. AdaShape iteration #1 (select-shape-mix)
 *   7. AdaShape iteration #2 (select-shape-mix)
 *   8. BWE (8-16 kHz synthesis)
 *   9. Combine lowband + highband -> fullband output
 */

#include "dsp/features.h"
#include "modules/adacomb.h"
#include "modules/adaconv.h"
#include "modules/adashape.h"
#include "modules/bwe.h"

/* All DNN-predicted parameters for one frame */
typedef struct {
    AdaCombParams  comb1;
    AdaCombParams  comb2;
    AdaConvParams  conv;
    AdaShapeParams shape1;
    AdaShapeParams shape2;
    BWEParams      bwe;
} EnhanceFrameParams;

/*
 * DNN inference callback.
 * The pipeline calls this once per frame to get module parameters.
 * Implementations can wrap PyTorch, ONNX Runtime, a custom C runtime, etc.
 */
typedef void (*NomsbcDNNInferFn)(const float *features, int feat_dim,
                                  EnhanceFrameParams *params,
                                  void *user_data);

typedef struct NomsbcEnhancer NomsbcEnhancer;

NomsbcEnhancer *nomsbc_enhancer_create(NomsbcDNNInferFn infer_fn,
                                        void *user_data);
void            nomsbc_enhancer_destroy(NomsbcEnhancer *e);

/*
 * Process one 10 ms frame.
 *   input:   NOMSBC_FRAME_SIZE samples of decoded mSBC (16 kHz)
 *   output:  NOMSBC_FRAME_SIZE samples of enhanced fullband (16 kHz)
 */
void nomsbc_enhancer_process_frame(NomsbcEnhancer *e,
                                    const float *input,
                                    float *output);

#endif
