#ifndef NOMSBC_FEATURES_H
#define NOMSBC_FEATURES_H

/*
 * Blind feature extraction from decoded mSBC audio.
 *
 * Aggregates all features needed by the enhancement DNN into a single
 * feature vector per 10 ms frame:
 *
 *   [0..19]   20 bark-scale cepstral coefficients
 *   [20]      pitch lag (normalized to 0..1 range)
 *   [21]      pitch correlation
 *   [22]      log frame energy
 *   [23]      spectral tilt (ratio of low-band to high-band energy)
 *   [24..27]  4 sub-band pitch correlations (voicing features)
 *
 * Total: 28 features per frame.
 */

#include "dsp/pitch.h"
#include "dsp/cepstral.h"

#define NOMSBC_FEATURE_DIM     28
#define NOMSBC_FRAME_SIZE      160  /* 10 ms at 16 kHz */
#define NOMSBC_SAMPLE_RATE     16000

typedef struct NomsbcFeatures NomsbcFeatures;

NomsbcFeatures *nomsbc_features_create(void);
void            nomsbc_features_destroy(NomsbcFeatures *fe);

/*
 * Extract features for one 10 ms frame.
 *   frame:     NOMSBC_FRAME_SIZE samples at 16 kHz
 *   features:  output array of NOMSBC_FEATURE_DIM floats
 *   pitch_lag: (output, optional) raw pitch lag in samples
 */
void nomsbc_features_extract(NomsbcFeatures *fe,
                             const float *frame,
                             float *features,
                             int *pitch_lag);

#endif
