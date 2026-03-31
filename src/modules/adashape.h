#ifndef NOMSBC_ADASHAPE_H
#define NOMSBC_ADASHAPE_H

/*
 * Adaptive Temporal Shaping (AdaShape) module.
 *
 * This is the key differentiator from LACE. AdaShape applies a non-linear,
 * sample-level shaping function that adds fine temporal detail that linear
 * filtering (AdaComb/AdaConv) cannot produce.
 *
 * The module implements a "select-shape-mix" procedure:
 *   1. SELECT: choose a basis signal from a set of candidates
 *   2. SHAPE:  apply a per-sample non-linear shaping function
 *   3. MIX:    blend the shaped signal with the input using predicted gains
 *
 * The DNN predicts the selection weights, shaping parameters, and mix gains
 * per frame. The shaping function is a learnable activation applied
 * sample-by-sample at high temporal resolution.
 */

#define NOMSBC_ADASHAPE_NUM_BASES  4  /* number of basis signals */
#define NOMSBC_ADASHAPE_SHAPE_DIM  8  /* shaping function parameters */

typedef struct {
    float select_weights[NOMSBC_ADASHAPE_NUM_BASES]; /* softmax'd selection */
    float shape_params[NOMSBC_ADASHAPE_SHAPE_DIM];   /* non-linear shaping */
    float mix_gain;    /* how much shaped signal to mix in (0..1) */
} AdaShapeParams;

typedef struct NomsbcAdaShape NomsbcAdaShape;

NomsbcAdaShape *nomsbc_adashape_create(int max_frame_size);
void            nomsbc_adashape_destroy(NomsbcAdaShape *as);

/*
 * Process one frame.
 *   signal:     input/output, frame_size samples (modified in place)
 *   pitch_lag:  current pitch lag (used to generate basis signals)
 *   params:     DNN-predicted shaping parameters
 */
void nomsbc_adashape_process(NomsbcAdaShape *as,
                             float *signal, int frame_size,
                             int pitch_lag,
                             const AdaShapeParams *params);

#endif
