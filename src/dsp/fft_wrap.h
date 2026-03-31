#ifndef NOMSBC_FFT_WRAP_H
#define NOMSBC_FFT_WRAP_H

/*
 * Thin wrapper around kiss_fft providing the FFT sizes used in the pipeline.
 * All transforms are real-to-complex (forward) / complex-to-real (inverse).
 */

#include <stddef.h>

typedef struct NomsbcFFT NomsbcFFT;

/* Create an FFT context for a given window length (must be power of 2). */
NomsbcFFT *nomsbc_fft_create(int nfft);
void       nomsbc_fft_destroy(NomsbcFFT *f);

/* Forward: nfft real samples -> nfft/2+1 complex bins (interleaved re,im). */
void nomsbc_fft_forward(NomsbcFFT *f, const float *in, float *out);

/* Inverse: nfft/2+1 complex bins -> nfft real samples. */
void nomsbc_fft_inverse(NomsbcFFT *f, const float *in, float *out);

/* Power spectrum: nfft real samples -> nfft/2+1 power values. */
void nomsbc_fft_power_spectrum(NomsbcFFT *f, const float *in, float *out);

#endif
