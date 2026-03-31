#include "modules/adashape.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAX_DELAY 512

struct NomsbcAdaShape {
    float delay_line[MAX_DELAY];
    int   write_pos;
};

NomsbcAdaShape *nomsbc_adashape_create(int max_frame_size)
{
    (void)max_frame_size;
    return calloc(1, sizeof(NomsbcAdaShape));
}

void nomsbc_adashape_destroy(NomsbcAdaShape *as)
{
    free(as);
}

/*
 * Learnable activation: a parameterized non-linearity.
 * Uses a sum-of-tanh basis: f(x) = sum_i a_i * tanh(b_i * x)
 * with parameters packed as [a0, b0, a1, b1, ...].
 */
static float shape_activation(float x, const float *params, int dim)
{
    float y = 0.0f;
    for (int i = 0; i < dim; i += 2) {
        float a = params[i];
        float b = params[i + 1];
        y += a * tanhf(b * x);
    }
    return y;
}

static float read_delay(const NomsbcAdaShape *as, int offset)
{
    int idx = as->write_pos - offset;
    while (idx < 0) idx += MAX_DELAY;
    return as->delay_line[idx % MAX_DELAY];
}

void nomsbc_adashape_process(NomsbcAdaShape *as,
                             float *signal, int frame_size,
                             int pitch_lag,
                             const AdaShapeParams *params)
{
    for (int n = 0; n < frame_size; n++) {
        float x = signal[n];
        as->delay_line[as->write_pos] = x;

        /* SELECT: generate basis signals from delay line */
        float bases[NOMSBC_ADASHAPE_NUM_BASES];
        bases[0] = x;                                      /* current sample */
        bases[1] = read_delay(as, pitch_lag);               /* pitch-delayed */
        bases[2] = read_delay(as, pitch_lag / 2);           /* half-pitch */
        bases[3] = 0.5f * (read_delay(as, 1) + read_delay(as, -1 + MAX_DELAY));
                                                            /* smoothed neighbor */

        /* Weighted sum of bases */
        float selected = 0.0f;
        for (int b = 0; b < NOMSBC_ADASHAPE_NUM_BASES; b++)
            selected += params->select_weights[b] * bases[b];

        /* SHAPE: apply non-linear activation */
        float shaped = shape_activation(selected, params->shape_params,
                                        NOMSBC_ADASHAPE_SHAPE_DIM);

        /* MIX: blend with input */
        signal[n] = x + params->mix_gain * (shaped - x);

        as->write_pos = (as->write_pos + 1) % MAX_DELAY;
    }
}
