#ifndef NOMSBC_CEPSTRAL_H
#define NOMSBC_CEPSTRAL_H

/*
 * Bark-scale cepstral feature extraction.
 *
 * Computes 20 bark-scale cepstral coefficients from a 10 ms frame at 16 kHz.
 * Pipeline: windowing -> FFT -> power spectrum -> bark filterbank -> log -> DCT
 */

#include "dsp/fft_wrap.h"
#include <stddef.h>

#define NOMSBC_CEPSTRAL_ORDER  20
#define NOMSBC_BARK_BANDS      22  /* Number of bark-scale triangular filters */

typedef struct NomsbcCepstral NomsbcCepstral;

/* Create for a given FFT size (should be >= frame_len, power of 2). */
NomsbcCepstral *nomsbc_cepstral_create(int nfft);
void            nomsbc_cepstral_destroy(NomsbcCepstral *c);

/*
 * Compute bark-scale cepstral coefficients for one frame.
 *   frame:    frame_len samples (typically 160 for 10 ms @ 16 kHz)
 *   cepstra:  output array of NOMSBC_CEPSTRAL_ORDER coefficients
 *   bark_energies: (optional, may be NULL) output NOMSBC_BARK_BANDS log energies
 */
void nomsbc_cepstral_compute(NomsbcCepstral *c,
                             const float *frame, int frame_len,
                             float *cepstra, float *bark_energies);

#endif
