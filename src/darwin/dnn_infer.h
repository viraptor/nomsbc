#ifndef NOMSBC_DNN_INFER_H
#define NOMSBC_DNN_INFER_H

/*
 * Minimal C inference runtime for the NoLACE-mSBC DNN.
 *
 * Architecture (must match training/model.py):
 *   FeatureEncoder:  Linear(28,128) + Tanh + Linear(128,128) + Tanh
 *   FrameGRU:        2-layer GRU, input=128, hidden=192
 *   ParameterHead:   Linear(192, PARAM_DIM)
 *
 * One call to nomsbc_dnn_infer() processes a single 10 ms frame,
 * advancing the GRU hidden state.
 */

#include "pipeline/enhance.h"
#include "darwin/weights.h"

/* Architecture constants (must match model.py) */
#define NOMSBC_ENC_HIDDEN    128
#define NOMSBC_GRU_HIDDEN    192
#define NOMSBC_GRU_LAYERS    2

typedef struct NomsbcDNN NomsbcDNN;

/*
 * Create inference engine from loaded weights.
 * Returns NULL on failure (missing layers, dimension mismatch).
 */
NomsbcDNN *nomsbc_dnn_create(const NomsbcWeights *weights);
void       nomsbc_dnn_destroy(NomsbcDNN *dnn);

/*
 * Run one frame of inference.
 * Matches the NomsbcDNNInferFn signature so it can be used directly
 * as the enhancer callback (pass the NomsbcDNN* as user_data).
 */
void nomsbc_dnn_infer(const float *features, int feat_dim,
                      EnhanceFrameParams *params, void *user_data);

/*
 * Reset GRU hidden state to zeros (e.g. on stream discontinuity).
 */
void nomsbc_dnn_reset(NomsbcDNN *dnn);

#endif
