#include "modules/adaconv.h"
#include <stdlib.h>
#include <string.h>

struct NomsbcAdaConv {
    float overlap[NOMSBC_ADACONV_KERNEL_SIZE - 1]; /* overlap-add tail */
    float prev_kernel[NOMSBC_ADACONV_KERNEL_SIZE];  /* for crossfade */
    float prev_gain;
    int   initialized;
};

NomsbcAdaConv *nomsbc_adaconv_create(int max_frame_size)
{
    (void)max_frame_size;
    NomsbcAdaConv *ac = calloc(1, sizeof(*ac));
    /* Initialize prev_kernel as identity (passthrough) */
    if (ac) {
        ac->prev_kernel[0] = 1.0f;
        ac->prev_gain = 1.0f;
    }
    return ac;
}

void nomsbc_adaconv_destroy(NomsbcAdaConv *ac)
{
    free(ac);
}

void nomsbc_adaconv_process(NomsbcAdaConv *ac,
                            float *signal, int frame_size,
                            const AdaConvParams *params)
{
    int klen = NOMSBC_ADACONV_KERNEL_SIZE;
    float out[512]; /* max frame_size */

    /* Apply FIR convolution */
    for (int n = 0; n < frame_size; n++) {
        float sum = 0.0f;
        for (int k = 0; k < klen; k++) {
            int idx = n - k;
            float sample;
            if (idx >= 0)
                sample = signal[idx];
            else
                sample = ac->overlap[klen - 1 + idx]; /* from previous frame tail */
            /* Crossfade between previous and current kernel over first few samples */
            float alpha = (n < klen) ? (float)n / klen : 1.0f;
            float w = alpha * params->kernel[k] + (1.0f - alpha) * ac->prev_kernel[k];
            sum += w * sample;
        }
        out[n] = sum * params->gain;
    }

    /* Save tail of input for next frame's overlap */
    if (frame_size >= klen - 1)
        memcpy(ac->overlap, signal + frame_size - (klen - 1),
               (klen - 1) * sizeof(float));

    /* Save current kernel */
    memcpy(ac->prev_kernel, params->kernel, klen * sizeof(float));
    ac->prev_gain = params->gain;

    memcpy(signal, out, frame_size * sizeof(float));
}
