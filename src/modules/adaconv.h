#ifndef NOMSBC_ADACONV_H
#define NOMSBC_ADACONV_H

/*
 * Adaptive Convolution (AdaConv) module.
 *
 * Applies a short FIR filter whose coefficients are predicted per-frame
 * by the DNN. This handles spectral reshaping -- correcting the spectral
 * envelope distortions from mSBC encoding.
 *
 * The filter operates in the time domain with overlap-add to avoid
 * discontinuities at frame boundaries.
 */

#define NOMSBC_ADACONV_KERNEL_SIZE  16

typedef struct {
    float kernel[NOMSBC_ADACONV_KERNEL_SIZE];
    float gain;
} AdaConvParams;

typedef struct NomsbcAdaConv NomsbcAdaConv;

NomsbcAdaConv *nomsbc_adaconv_create(int max_frame_size);
void           nomsbc_adaconv_destroy(NomsbcAdaConv *ac);

/*
 * Process one frame in-place.
 *   signal:     frame_size samples (modified in place)
 *   params:     DNN-predicted kernel and gain
 */
void nomsbc_adaconv_process(NomsbcAdaConv *ac,
                            float *signal, int frame_size,
                            const AdaConvParams *params);

#endif
