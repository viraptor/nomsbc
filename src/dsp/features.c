#include "dsp/features.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

struct NomsbcFeatures {
    NomsbcPitch    *pitch;
    NomsbcCepstral *cepstral;
    NomsbcFFT      *fft256; /* for sub-band analysis */
};

NomsbcFeatures *nomsbc_features_create(void)
{
    NomsbcFeatures *fe = calloc(1, sizeof(*fe));
    if (!fe) return NULL;

    fe->pitch    = nomsbc_pitch_create();
    fe->cepstral = nomsbc_cepstral_create(256);
    fe->fft256   = nomsbc_fft_create(256);

    if (!fe->pitch || !fe->cepstral || !fe->fft256) {
        nomsbc_features_destroy(fe);
        return NULL;
    }
    return fe;
}

void nomsbc_features_destroy(NomsbcFeatures *fe)
{
    if (!fe) return;
    nomsbc_pitch_destroy(fe->pitch);
    nomsbc_cepstral_destroy(fe->cepstral);
    nomsbc_fft_destroy(fe->fft256);
    free(fe);
}

static float compute_log_energy(const float *frame, int len)
{
    float e = 0.0f;
    for (int i = 0; i < len; i++)
        e += frame[i] * frame[i];
    return logf(fmaxf(e / len, 1e-10f));
}

static float compute_spectral_tilt(const float *power, int nbins)
{
    /* Ratio of energy below 2 kHz to energy above 2 kHz.
     * At 16 kHz SR with 256-pt FFT: bin spacing = 62.5 Hz, 2 kHz = bin 32 */
    float lo = 0.0f, hi = 0.0f;
    int split = nbins / 4; /* ~2 kHz */
    for (int i = 1; i < split; i++) lo += power[i];
    for (int i = split; i < nbins; i++) hi += power[i];
    if (hi < 1e-10f) return 10.0f;
    return logf(fmaxf(lo, 1e-10f)) - logf(fmaxf(hi, 1e-10f));
}

/* Sub-band pitch correlation: split spectrum into 4 bands, compute
 * autocorrelation at the estimated pitch lag for each band. */
static void compute_subband_correlations(NomsbcFFT *fft,
                                         const float *frame, int frame_len,
                                         int lag, float *corr4)
{
    /* Simplified: compute time-domain correlation in 4 frequency sub-bands
     * by band-pass filtering (just partition the frame into short segments
     * and compute per-segment correlations as a proxy). */
    int quarter = frame_len / 4;
    for (int b = 0; b < 4; b++) {
        float xy = 0, xx = 0, yy = 0;
        /* Use overlapping windows of the signal at different offsets
         * as a crude sub-band proxy. For production, use proper band-pass. */
        int start = b * quarter;
        int end = start + quarter;
        if (lag > start) { corr4[b] = 0.0f; continue; }
        for (int i = start; i < end; i++) {
            float x = frame[i];
            float y = (i - lag >= 0) ? frame[i - lag] : 0.0f;
            xy += x * y;
            xx += x * x;
            yy += y * y;
        }
        float denom = sqrtf(xx * yy);
        corr4[b] = (denom > 1e-12f) ? xy / denom : 0.0f;
    }
}

void nomsbc_features_extract(NomsbcFeatures *fe,
                             const float *frame,
                             float *features,
                             int *pitch_lag_out)
{
    int lag;
    float pitch_corr;
    nomsbc_pitch_estimate(fe->pitch, frame, NOMSBC_FRAME_SIZE, &lag, &pitch_corr);

    /* Cepstral coefficients */
    float cepstra[NOMSBC_CEPSTRAL_ORDER];
    nomsbc_cepstral_compute(fe->cepstral, frame, NOMSBC_FRAME_SIZE,
                            cepstra, NULL);

    /* Power spectrum for tilt */
    float padded[256];
    memset(padded, 0, sizeof(padded));
    memcpy(padded, frame, NOMSBC_FRAME_SIZE * sizeof(float));
    float power[129];
    nomsbc_fft_power_spectrum(fe->fft256, padded, power);

    /* Pack features */
    memcpy(features, cepstra, NOMSBC_CEPSTRAL_ORDER * sizeof(float));

    features[20] = (float)lag / NOMSBC_PITCH_MAX_LAG; /* normalized lag */
    features[21] = pitch_corr;
    features[22] = compute_log_energy(frame, NOMSBC_FRAME_SIZE);
    features[23] = compute_spectral_tilt(power, 129);

    float corr4[4];
    compute_subband_correlations(fe->fft256, frame, NOMSBC_FRAME_SIZE, lag, corr4);
    memcpy(features + 24, corr4, 4 * sizeof(float));

    if (pitch_lag_out) *pitch_lag_out = lag;
}
