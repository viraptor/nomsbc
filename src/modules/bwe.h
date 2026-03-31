#ifndef NOMSBC_BWE_H
#define NOMSBC_BWE_H

/*
 * Bandwidth Extension (BWE) module.
 *
 * mSBC hard-limits at 8 kHz. This module synthesizes the 8-16 kHz band
 * from the enhanced wideband (0-8 kHz) signal, producing fullband output
 * at 16 kHz sample rate.
 *
 * Architecture: framewise autoregressive subband synthesis, inspired by
 * FARGAN / UBGAN approaches. The DNN predicts spectral envelope and
 * excitation parameters for the highband from the lowband features.
 *
 * The module operates on 10 ms frames with up to 5 ms lookahead.
 */

#define NOMSBC_BWE_HIGHBAND_BINS   8   /* subband decomposition bins */
#define NOMSBC_BWE_ENVELOPE_DIM   16   /* spectral envelope parameters */

typedef struct {
    float envelope[NOMSBC_BWE_ENVELOPE_DIM]; /* predicted highband envelope */
    float excitation_gain;                    /* overall excitation level */
    float voicing_factor;                     /* 0=noise, 1=harmonic */
} BWEParams;

typedef struct NomsbcBWE NomsbcBWE;

NomsbcBWE *nomsbc_bwe_create(int frame_size);
void       nomsbc_bwe_destroy(NomsbcBWE *bwe);

/*
 * Synthesize highband for one frame.
 *   lowband:   enhanced 0-8 kHz signal, frame_size samples at 16 kHz
 *   highband:  output 8-16 kHz band, frame_size samples at 16 kHz
 *   pitch_lag: from pitch tracker
 *   params:    DNN-predicted BWE parameters
 */
void nomsbc_bwe_synthesize(NomsbcBWE *bwe,
                           const float *lowband, float *highband,
                           int frame_size, int pitch_lag,
                           const BWEParams *params);

/*
 * Combine lowband + highband into fullband output.
 *   lowband:   frame_size samples
 *   highband:  frame_size samples
 *   output:    frame_size samples of fullband signal
 */
void nomsbc_bwe_combine(const float *lowband, const float *highband,
                        float *output, int frame_size);

#endif
