#include "dsp/cepstral.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
 * Bark scale: frequency to bark using Traunmuller's formula.
 *   bark = 26.81 / (1 + 1960/f) - 0.53
 */
static float hz_to_bark(float hz)
{
    return 26.81f / (1.0f + 1960.0f / hz) - 0.53f;
}

static float bark_to_hz(float bark)
{
    return 1960.0f * (bark + 0.53f) / (26.28f - bark);
}

struct NomsbcCepstral {
    NomsbcFFT *fft;
    int        nfft;
    int        nbins;         /* nfft/2 + 1 */
    float      window[512];   /* Hann window (max supported nfft) */
    int        bank_start[NOMSBC_BARK_BANDS]; /* first bin of each filter */
    int        bank_end[NOMSBC_BARK_BANDS];   /* last bin (exclusive) */
    float     *bank_weights;  /* triangular filter weights, packed */
    int        bank_weight_count;
    /* DCT-II matrix for bark_bands -> cepstral_order */
    float      dct[NOMSBC_CEPSTRAL_ORDER * NOMSBC_BARK_BANDS];
};

NomsbcCepstral *nomsbc_cepstral_create(int nfft)
{
    NomsbcCepstral *c = calloc(1, sizeof(*c));
    if (!c) return NULL;

    c->nfft  = nfft;
    c->nbins = nfft / 2 + 1;
    c->fft   = nomsbc_fft_create(nfft);
    if (!c->fft) { free(c); return NULL; }

    /* Hann window */
    for (int i = 0; i < nfft; i++)
        c->window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / nfft));

    /* Build bark-scale triangular filterbank */
    float sr = 16000.0f;
    float max_bark = hz_to_bark(sr / 2.0f);
    float bark_step = max_bark / (NOMSBC_BARK_BANDS + 1);

    float center_hz[NOMSBC_BARK_BANDS + 2];
    for (int i = 0; i <= NOMSBC_BARK_BANDS + 1; i++)
        center_hz[i] = bark_to_hz(i * bark_step);

    /* Convert centers to FFT bin indices */
    int center_bin[NOMSBC_BARK_BANDS + 2];
    for (int i = 0; i <= NOMSBC_BARK_BANDS + 1; i++) {
        center_bin[i] = (int)roundf(center_hz[i] * nfft / sr);
        if (center_bin[i] >= c->nbins) center_bin[i] = c->nbins - 1;
    }

    /* Count total weights needed */
    c->bank_weight_count = 0;
    for (int b = 0; b < NOMSBC_BARK_BANDS; b++) {
        c->bank_start[b] = center_bin[b];
        c->bank_end[b]   = center_bin[b + 2];
        c->bank_weight_count += c->bank_end[b] - c->bank_start[b];
    }

    c->bank_weights = calloc(c->bank_weight_count, sizeof(float));
    int wi = 0;
    for (int b = 0; b < NOMSBC_BARK_BANDS; b++) {
        int lo = center_bin[b];
        int mid = center_bin[b + 1];
        int hi = center_bin[b + 2];
        for (int k = lo; k < hi; k++) {
            if (k < mid)
                c->bank_weights[wi] = (float)(k - lo) / fmaxf(1.0f, (float)(mid - lo));
            else
                c->bank_weights[wi] = (float)(hi - k) / fmaxf(1.0f, (float)(hi - mid));
            wi++;
        }
    }

    /* DCT-II matrix: dct[i][j] = cos(pi * i * (j + 0.5) / N) */
    for (int i = 0; i < NOMSBC_CEPSTRAL_ORDER; i++)
        for (int j = 0; j < NOMSBC_BARK_BANDS; j++)
            c->dct[i * NOMSBC_BARK_BANDS + j] =
                cosf((float)M_PI * i * (j + 0.5f) / NOMSBC_BARK_BANDS);

    return c;
}

void nomsbc_cepstral_destroy(NomsbcCepstral *c)
{
    if (!c) return;
    nomsbc_fft_destroy(c->fft);
    free(c->bank_weights);
    free(c);
}

void nomsbc_cepstral_compute(NomsbcCepstral *c,
                             const float *frame, int frame_len,
                             float *cepstra, float *bark_energies)
{
    float windowed[512];
    memset(windowed, 0, c->nfft * sizeof(float));
    for (int i = 0; i < frame_len && i < c->nfft; i++)
        windowed[i] = frame[i] * c->window[i];

    /* Power spectrum */
    float power[257]; /* max nbins for nfft=512 */
    nomsbc_fft_power_spectrum(c->fft, windowed, power);

    /* Apply bark filterbank */
    float log_energy[NOMSBC_BARK_BANDS];
    int wi = 0;
    for (int b = 0; b < NOMSBC_BARK_BANDS; b++) {
        float energy = 0.0f;
        for (int k = c->bank_start[b]; k < c->bank_end[b]; k++)
            energy += power[k] * c->bank_weights[wi++];
        log_energy[b] = logf(fmaxf(energy, 1e-10f));
    }

    if (bark_energies)
        memcpy(bark_energies, log_energy, NOMSBC_BARK_BANDS * sizeof(float));

    /* DCT to get cepstral coefficients */
    for (int i = 0; i < NOMSBC_CEPSTRAL_ORDER; i++) {
        float sum = 0.0f;
        for (int j = 0; j < NOMSBC_BARK_BANDS; j++)
            sum += c->dct[i * NOMSBC_BARK_BANDS + j] * log_energy[j];
        cepstra[i] = sum;
    }
}
