/*
 * Demo / test: runs the enhancement pipeline with a dummy DNN
 * (identity passthrough) to verify the DSP pipeline compiles and runs.
 */

#include "pipeline/enhance.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static void dummy_infer(const float *features, int feat_dim,
                        EnhanceFrameParams *params, void *user_data)
{
    (void)features; (void)feat_dim; (void)user_data;

    /* Bypass all modules: zero comb/shape gain, identity conv */
    memset(params, 0, sizeof(*params));

    /* AdaConv: identity (impulse at [0]) */
    params->conv.kernel[0] = 1.0f;
    params->conv.gain = 1.0f;

    /* BWE: silent highband */
    params->bwe.excitation_gain = 0.0f;
}

int main(int argc, char **argv)
{
    (void)argc; (void)argv;

    NomsbcEnhancer *e = nomsbc_enhancer_create(dummy_infer, NULL);
    if (!e) {
        fprintf(stderr, "Failed to create enhancer\n");
        return 1;
    }

    /* Generate a simple test signal: 200 Hz sine at 16 kHz */
    float input[NOMSBC_FRAME_SIZE];
    float output[NOMSBC_FRAME_SIZE];
    for (int i = 0; i < NOMSBC_FRAME_SIZE; i++)
        input[i] = 0.5f * sinf(2.0f * (float)M_PI * 200.0f * i / NOMSBC_SAMPLE_RATE);

    /* Process 100 frames (1 second) */
    float max_diff = 0.0f;
    for (int f = 0; f < 100; f++) {
        nomsbc_enhancer_process_frame(e, input, output);
        for (int i = 0; i < NOMSBC_FRAME_SIZE; i++) {
            float d = fabsf(output[i] - input[i]);
            if (d > max_diff) max_diff = d;
        }
    }

    printf("Pipeline OK. Max deviation from passthrough: %.6f\n", max_diff);

    nomsbc_enhancer_destroy(e);
    return 0;
}
