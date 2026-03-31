/*
 * Data preparation tool for NoLACE-mSBC training.
 *
 * Reads clean speech audio files (WAV/RAW), passes them through the mSBC
 * encode/decode chain, and writes paired (degraded, clean) feature files
 * for training.
 *
 * Output format: binary files with interleaved frames of:
 *   - NOMSBC_FEATURE_DIM floats of input features (from degraded signal)
 *   - NOMSBC_FRAME_SIZE floats of clean target (for loss computation)
 *   - NOMSBC_FRAME_SIZE floats of degraded input signal
 *
 * Usage:
 *   prep_data <input_dir> <output_dir> [--raw-16k]
 */

#include "data/sbc_codec.h"
#include "dsp/features.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef HAVE_SNDFILE
#include <sndfile.h>
#endif

#define MAX_AUDIO_SAMPLES (16000 * 600)  /* 10 minutes max */

static int read_raw_16k(const char *path, int16_t *pcm, int max_samples)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int n = (int)fread(pcm, sizeof(int16_t), max_samples, f);
    fclose(f);
    return n;
}

#ifdef HAVE_SNDFILE
static int read_wav(const char *path, int16_t *pcm, int max_samples)
{
    SF_INFO info = {0};
    SNDFILE *sf = sf_open(path, SFM_READ, &info);
    if (!sf) return -1;
    if (info.samplerate != 16000 || info.channels != 1) {
        fprintf(stderr, "Warning: %s is not 16kHz mono, skipping\n", path);
        sf_close(sf);
        return -1;
    }
    int n = (int)sf_readf_short(sf, pcm, max_samples);
    sf_close(sf);
    return n;
}
#endif

static int degrade_through_msbc(NomsbcSBC *sbc,
                                const int16_t *clean, int16_t *degraded,
                                int num_samples)
{
    int pos = 0;
    while (pos + NOMSBC_SBC_FRAME_SAMPLES <= num_samples) {
        uint8_t encoded[NOMSBC_SBC_FRAME_BYTES + 16];
        int enc_len;
        if (nomsbc_sbc_encode(sbc, clean + pos, NOMSBC_SBC_FRAME_SAMPLES,
                              encoded, &enc_len) != 0)
            return -1;

        int dec_samples;
        if (nomsbc_sbc_decode(sbc, encoded, enc_len,
                              degraded + pos, &dec_samples) != 0)
            return -1;

        pos += NOMSBC_SBC_FRAME_SAMPLES;
    }
    return pos; /* number of samples actually processed */
}

static void int16_to_float(const int16_t *in, float *out, int n)
{
    for (int i = 0; i < n; i++)
        out[i] = in[i] / 32768.0f;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.raw|input.wav> <output.bin> [--raw-16k]\n",
                argv[0]);
        return 1;
    }

    const char *input_path = argv[1];
    const char *output_path = argv[2];
    int raw_mode = (argc > 3 && strcmp(argv[3], "--raw-16k") == 0);

    int16_t *clean = malloc(MAX_AUDIO_SAMPLES * sizeof(int16_t));
    int16_t *degraded = malloc(MAX_AUDIO_SAMPLES * sizeof(int16_t));
    if (!clean || !degraded) {
        fprintf(stderr, "Out of memory\n");
        return 1;
    }

    /* Read input audio */
    int num_samples;
    if (raw_mode) {
        num_samples = read_raw_16k(input_path, clean, MAX_AUDIO_SAMPLES);
    } else {
#ifdef HAVE_SNDFILE
        num_samples = read_wav(input_path, clean, MAX_AUDIO_SAMPLES);
#else
        fprintf(stderr, "WAV support requires libsndfile. Use --raw-16k for raw input.\n");
        return 1;
#endif
    }

    if (num_samples < 0) {
        fprintf(stderr, "Failed to read %s\n", input_path);
        return 1;
    }
    fprintf(stderr, "Read %d samples (%.1f seconds)\n",
            num_samples, num_samples / 16000.0f);

    /* Degrade through mSBC */
    NomsbcSBC *sbc = nomsbc_sbc_create();
    if (!sbc) {
        fprintf(stderr, "Failed to init SBC codec\n");
        return 1;
    }

    int processed = degrade_through_msbc(sbc, clean, degraded, num_samples);
    nomsbc_sbc_destroy(sbc);
    if (processed < 0) {
        fprintf(stderr, "mSBC encode/decode failed\n");
        return 1;
    }

    /* Extract features and write paired data */
    NomsbcFeatures *fe = nomsbc_features_create();
    FILE *out = fopen(output_path, "wb");
    if (!out) {
        fprintf(stderr, "Cannot open %s for writing\n", output_path);
        return 1;
    }

    float clean_f[NOMSBC_FRAME_SIZE];
    float degraded_f[NOMSBC_FRAME_SIZE];
    float features[NOMSBC_FEATURE_DIM];

    int num_frames = 0;
    for (int pos = 0; pos + NOMSBC_FRAME_SIZE <= processed; pos += NOMSBC_FRAME_SIZE) {
        int16_to_float(degraded + pos, degraded_f, NOMSBC_FRAME_SIZE);
        int16_to_float(clean + pos, clean_f, NOMSBC_FRAME_SIZE);

        nomsbc_features_extract(fe, degraded_f, features, NULL);

        fwrite(features, sizeof(float), NOMSBC_FEATURE_DIM, out);
        fwrite(clean_f, sizeof(float), NOMSBC_FRAME_SIZE, out);
        fwrite(degraded_f, sizeof(float), NOMSBC_FRAME_SIZE, out);
        num_frames++;
    }

    fclose(out);
    fprintf(stderr, "Wrote %d frames to %s\n", num_frames, output_path);
    fprintf(stderr, "Frame layout: %d features + %d clean + %d degraded = %d floats\n",
            NOMSBC_FEATURE_DIM, NOMSBC_FRAME_SIZE, NOMSBC_FRAME_SIZE,
            NOMSBC_FEATURE_DIM + 2 * NOMSBC_FRAME_SIZE);

    nomsbc_features_destroy(fe);
    free(clean);
    free(degraded);
    return 0;
}
