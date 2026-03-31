#ifndef NOMSBC_ADACOMB_H
#define NOMSBC_ADACOMB_H

/*
 * Adaptive Comb Filter (AdaComb) module.
 *
 * Applies a pitch-periodic comb filter whose kernel weights are predicted
 * by the DNN. The filter cleans up harmonic structure using the estimated
 * pitch lag.
 *
 * y[n] = x[n] + alpha * sum_{k} w[k] * x[n - lag + k]
 *
 * where w[k] is the comb kernel (predicted per-frame) and alpha is the
 * global gain (also predicted). Kernel size is typically 3-5 taps centered
 * on the pitch lag.
 */

#define NOMSBC_ADACOMB_KERNEL_SIZE  5

typedef struct {
    /* Weights predicted by DNN each frame */
    float kernel[NOMSBC_ADACOMB_KERNEL_SIZE];
    float global_gain;   /* alpha: 0 = bypass, 1 = full comb */
} AdaCombParams;

typedef struct NomsbcAdaComb NomsbcAdaComb;

NomsbcAdaComb *nomsbc_adacomb_create(int max_frame_size);
void           nomsbc_adacomb_destroy(NomsbcAdaComb *ac);

/*
 * Process one frame in-place.
 *   signal:    frame_size samples (modified in place)
 *   pitch_lag: pitch period in samples for this frame
 *   params:    DNN-predicted kernel weights and gain
 */
void nomsbc_adacomb_process(NomsbcAdaComb *ac,
                            float *signal, int frame_size,
                            int pitch_lag,
                            const AdaCombParams *params);

#endif
