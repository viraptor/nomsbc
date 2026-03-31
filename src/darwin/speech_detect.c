#include "darwin/speech_detect.h"
#include "dsp/fft_wrap.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * Detection runs every CHECK_INTERVAL frames to save CPU.
 * At 48 kHz / 480-sample frames = 10 ms per frame, so checking every
 * 4 frames means ~40 ms between spectral checks.
 */
#define CHECK_INTERVAL  4

/*
 * Hysteresis thresholds (in frames that passed detection check).
 * ENTER_COUNT consecutive "codec-limited" checks to switch to enhance.
 * EXIT_COUNT  consecutive "fullband" checks to switch back to passthrough.
 * The asymmetry makes exiting harder, preventing flapping.
 */
#define ENTER_COUNT     8    /* ~320 ms to start enhancing */
#define EXIT_COUNT     15    /* ~600 ms to stop enhancing  */

/*
 * Spectral energy ratio threshold.
 * If energy above 8 kHz is less than this fraction of total energy,
 * the audio is likely bandwidth-limited to mSBC's range.
 */
#define HIGHBAND_RATIO_THRESHOLD  0.02f

/*
 * Minimum total energy (squared magnitude sum) to consider a frame
 * as non-silent.  Prevents triggering on quiet/silent streams.
 */
#define SILENCE_THRESHOLD  1e-6f

/*
 * FFT size for spectral analysis.  Must be a power of 2 and large
 * enough to resolve the 8 kHz boundary at the system sample rate.
 * 1024 bins at 48 kHz gives ~47 Hz resolution.
 */
#define FFT_SIZE  1024

struct NomsbcSpeechDetect {
    int sample_rate;
    int frame_size;

    /* FFT */
    NomsbcFFT *fft;
    float      window[FFT_SIZE];
    float      buf[FFT_SIZE];   /* accumulation buffer */
    int        buf_pos;

    /* Detection state */
    int  frame_counter;       /* counts frames between checks */
    int  detect_count;        /* consecutive "codec" detections */
    int  fullband_count;      /* consecutive "fullband" detections */
    bool current_state;       /* true = mSBC speech detected */

    /* Frequency bin boundary */
    int  highband_start_bin;  /* first FFT bin above 8 kHz */
};

static void build_hann_window(float *w, int n)
{
    for (int i = 0; i < n; i++)
        w[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (n - 1)));
}

NomsbcSpeechDetect *nomsbc_speech_detect_create(int sample_rate,
                                                 int frame_size)
{
    NomsbcSpeechDetect *sd = calloc(1, sizeof(*sd));
    if (!sd) return NULL;

    sd->sample_rate = sample_rate;
    sd->frame_size  = frame_size;
    sd->fft = nomsbc_fft_create(FFT_SIZE);
    if (!sd->fft) { free(sd); return NULL; }

    build_hann_window(sd->window, FFT_SIZE);

    /* First FFT bin whose center frequency >= 8000 Hz */
    float bin_hz = (float)sample_rate / FFT_SIZE;
    sd->highband_start_bin = (int)ceilf(8000.0f / bin_hz);

    return sd;
}

void nomsbc_speech_detect_destroy(NomsbcSpeechDetect *sd)
{
    if (!sd) return;
    nomsbc_fft_destroy(sd->fft);
    free(sd);
}

void nomsbc_speech_detect_reset(NomsbcSpeechDetect *sd)
{
    sd->frame_counter  = 0;
    sd->detect_count   = 0;
    sd->fullband_count = 0;
    sd->current_state  = false;
    sd->buf_pos        = 0;
}

/*
 * Run spectral analysis on the accumulated buffer and return whether
 * the audio looks bandwidth-limited to ~8 kHz.
 */
static bool check_spectrum(NomsbcSpeechDetect *sd)
{
    /* Apply window */
    float windowed[FFT_SIZE];
    for (int i = 0; i < FFT_SIZE; i++)
        windowed[i] = sd->buf[i] * sd->window[i];

    /* Real-to-complex FFT */
    float spectrum[FFT_SIZE + 2];  /* complex: (FFT_SIZE/2+1) pairs */
    nomsbc_fft_forward(sd->fft, windowed, spectrum);

    /* Compute energy in lowband (0..8 kHz) and highband (8 kHz..Nyquist) */
    int half = FFT_SIZE / 2 + 1;
    float total_energy = 0.0f;
    float high_energy  = 0.0f;

    for (int k = 0; k < half; k++) {
        float re = spectrum[2 * k];
        float im = spectrum[2 * k + 1];
        float mag2 = re * re + im * im;
        total_energy += mag2;
        if (k >= sd->highband_start_bin)
            high_energy += mag2;
    }

    /* Silent frame: no detection */
    if (total_energy < SILENCE_THRESHOLD)
        return false;

    float ratio = high_energy / total_energy;
    return ratio < HIGHBAND_RATIO_THRESHOLD;
}

bool nomsbc_speech_detect_feed(NomsbcSpeechDetect *sd, const float *samples)
{
    /* Accumulate samples into the FFT buffer (ring-style, keep latest) */
    int n = sd->frame_size;
    if (n >= FFT_SIZE) {
        /* Frame is larger than FFT: just take the last FFT_SIZE samples */
        memcpy(sd->buf, samples + n - FFT_SIZE, FFT_SIZE * sizeof(float));
        sd->buf_pos = FFT_SIZE;
    } else {
        /* Shift old data left and append new */
        int keep = FFT_SIZE - n;
        if (keep > 0)
            memmove(sd->buf, sd->buf + n, keep * sizeof(float));
        memcpy(sd->buf + keep, samples, n * sizeof(float));
        sd->buf_pos = FFT_SIZE;
    }

    /* Only run spectral check every CHECK_INTERVAL frames */
    sd->frame_counter++;
    if (sd->frame_counter < CHECK_INTERVAL)
        return sd->current_state;

    sd->frame_counter = 0;

    bool looks_limited = check_spectrum(sd);

    /* Update hysteresis counters */
    if (looks_limited) {
        sd->detect_count++;
        sd->fullband_count = 0;
    } else {
        sd->fullband_count++;
        sd->detect_count = 0;
    }

    /* State transitions with stickiness */
    if (!sd->current_state && sd->detect_count >= ENTER_COUNT) {
        sd->current_state = true;
    } else if (sd->current_state && sd->fullband_count >= EXIT_COUNT) {
        sd->current_state = false;
    }

    return sd->current_state;
}
