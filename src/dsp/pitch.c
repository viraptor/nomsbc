#include "dsp/pitch.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PITCH_HISTORY 480  /* Keep extra history for max-lag correlation */

struct NomsbcPitch {
    float history[PITCH_HISTORY];
    int   prev_lag;
};

NomsbcPitch *nomsbc_pitch_create(void)
{
    NomsbcPitch *p = calloc(1, sizeof(*p));
    if (p) p->prev_lag = 80; /* ~200 Hz default */
    return p;
}

void nomsbc_pitch_destroy(NomsbcPitch *p)
{
    free(p);
}

/*
 * Normalized autocorrelation at a given lag over a window of samples.
 * Uses the formula: r(lag) = sum(x[n]*x[n-lag]) / sqrt(sum(x[n]^2) * sum(x[n-lag]^2))
 */
static float normalized_autocorr(const float *buf, int len, int lag)
{
    float xy = 0.0f, xx = 0.0f, yy = 0.0f;
    for (int i = 0; i < len; i++) {
        float x = buf[i];
        float y = buf[i - lag]; /* caller ensures valid indices */
        xy += x * y;
        xx += x * x;
        yy += y * y;
    }
    float denom = sqrtf(xx * yy);
    if (denom < 1e-12f) return 0.0f;
    return xy / denom;
}

void nomsbc_pitch_estimate(NomsbcPitch *p,
                           const float *frame, int frame_len,
                           int *lag, float *correlation)
{
    /* Shift history and append new frame */
    int hist_shift = PITCH_HISTORY - frame_len;
    memmove(p->history, p->history + frame_len, hist_shift * sizeof(float));
    memcpy(p->history + hist_shift, frame, frame_len * sizeof(float));

    /* Pointer to the analysis window: last frame_len samples of history */
    const float *buf = p->history + NOMSBC_PITCH_MAX_LAG;
    int win = PITCH_HISTORY - NOMSBC_PITCH_MAX_LAG;

    /* Coarse search: step by 2 */
    float best_corr = -1.0f;
    int best_lag = p->prev_lag;
    for (int t = NOMSBC_PITCH_MIN_LAG; t <= NOMSBC_PITCH_MAX_LAG; t += 2) {
        float c = normalized_autocorr(buf, win, t);
        if (c > best_corr) {
            best_corr = c;
            best_lag = t;
        }
    }

    /* Fine search: +/- 1 around coarse best */
    for (int t = best_lag - 1; t <= best_lag + 1; t++) {
        if (t < NOMSBC_PITCH_MIN_LAG || t > NOMSBC_PITCH_MAX_LAG) continue;
        float c = normalized_autocorr(buf, win, t);
        if (c > best_corr) {
            best_corr = c;
            best_lag = t;
        }
    }

    /* Octave check: prefer half-lag if correlation is close (avoid octave errors) */
    if (best_lag >= NOMSBC_PITCH_MIN_LAG * 2) {
        int half = best_lag / 2;
        if (half >= NOMSBC_PITCH_MIN_LAG) {
            float c_half = normalized_autocorr(buf, win, half);
            if (c_half > 0.85f * best_corr) {
                best_corr = c_half;
                best_lag = half;
            }
        }
    }

    p->prev_lag = best_lag;
    *lag = best_lag;
    *correlation = fmaxf(0.0f, best_corr);
}
