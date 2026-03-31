#ifndef NOMSBC_PITCH_H
#define NOMSBC_PITCH_H

/*
 * Autocorrelation-based pitch tracker.
 *
 * Operates on 10 ms frames at 16 kHz (160 samples).
 * Produces pitch lag (in samples) and pitch correlation (0..1).
 * Search range: 20–320 samples (50 Hz – 800 Hz).
 */

#include <stddef.h>

#define NOMSBC_PITCH_MIN_LAG   20   /* 800 Hz at 16 kHz */
#define NOMSBC_PITCH_MAX_LAG   320  /* 50 Hz at 16 kHz  */

typedef struct NomsbcPitch NomsbcPitch;

NomsbcPitch *nomsbc_pitch_create(void);
void         nomsbc_pitch_destroy(NomsbcPitch *p);

/*
 * Estimate pitch for one frame.
 *   frame:       160 samples (10 ms at 16 kHz)
 *   *lag:        estimated pitch period in samples
 *   *correlation: normalized correlation at the chosen lag (0..1)
 */
void nomsbc_pitch_estimate(NomsbcPitch *p,
                           const float *frame, int frame_len,
                           int *lag, float *correlation);

#endif
