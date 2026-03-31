#include "modules/bwe.h"
#include "dsp/fft_wrap.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

struct NomsbcBWE {
    NomsbcFFT *fft;
    int        frame_size;
    float      noise_state;  /* simple noise generator state */
    float      prev_highband[256]; /* overlap buffer */
};

NomsbcBWE *nomsbc_bwe_create(int frame_size)
{
    NomsbcBWE *bwe = calloc(1, sizeof(*bwe));
    if (!bwe) return NULL;
    bwe->frame_size = frame_size;
    bwe->fft = nomsbc_fft_create(256);
    bwe->noise_state = 1.0f;
    return bwe;
}

void nomsbc_bwe_destroy(NomsbcBWE *bwe)
{
    if (!bwe) return;
    nomsbc_fft_destroy(bwe->fft);
    free(bwe);
}

/* Simple white noise generator */
static float noise_sample(float *state)
{
    /* LCG PRNG mapped to [-1, 1] */
    unsigned int s = *(unsigned int *)state;
    s = s * 1664525u + 1013904223u;
    *(unsigned int *)state = s;
    return (float)(int)s / 2147483648.0f;
}

void nomsbc_bwe_synthesize(NomsbcBWE *bwe,
                           const float *lowband, float *highband,
                           int frame_size, int pitch_lag,
                           const BWEParams *params)
{
    /*
     * Highband synthesis strategy:
     *   1. Generate excitation: mix of pitch-harmonic copy and noise
     *   2. Shape with predicted spectral envelope
     *
     * The DNN predicts envelope + voicing; this code applies them.
     * Production model would use FARGAN-style autoregressive generation.
     */

    /* Generate excitation signal */
    float excitation[256];
    for (int n = 0; n < frame_size; n++) {
        float harmonic = 0.0f;
        /* Fold lowband harmonics into highband range */
        if (pitch_lag > 0 && n >= pitch_lag)
            harmonic = lowband[n - pitch_lag];

        float noise = noise_sample(&bwe->noise_state);

        excitation[n] = params->voicing_factor * harmonic +
                        (1.0f - params->voicing_factor) * noise;
        excitation[n] *= params->excitation_gain;
    }

    /* Apply spectral envelope shaping in frequency domain */
    float padded[256];
    memset(padded, 0, sizeof(padded));
    memcpy(padded, excitation, frame_size * sizeof(float));

    float spec[258]; /* 129 complex bins */
    nomsbc_fft_forward(bwe->fft, padded, spec);

    /* Shape upper half of spectrum using envelope parameters */
    int nbins = 129;
    int hb_start = nbins / 2; /* 4 kHz */
    for (int i = 0; i < nbins; i++) {
        float gain;
        if (i < hb_start) {
            /* Suppress lowband content */
            gain = 0.0f;
        } else {
            /* Map envelope parameters across highband bins */
            int env_idx = (i - hb_start) * NOMSBC_BWE_ENVELOPE_DIM / (nbins - hb_start);
            if (env_idx >= NOMSBC_BWE_ENVELOPE_DIM) env_idx = NOMSBC_BWE_ENVELOPE_DIM - 1;
            gain = expf(params->envelope[env_idx]);
        }
        spec[2*i]   *= gain;
        spec[2*i+1] *= gain;
    }

    float synth[256];
    nomsbc_fft_inverse(bwe->fft, spec, synth);
    memcpy(highband, synth, frame_size * sizeof(float));
}

void nomsbc_bwe_combine(const float *lowband, const float *highband,
                        float *output, int frame_size)
{
    for (int i = 0; i < frame_size; i++)
        output[i] = lowband[i] + highband[i];
}
