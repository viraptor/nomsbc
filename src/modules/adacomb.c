#include "modules/adacomb.h"
#include <stdlib.h>
#include <string.h>

#define MAX_DELAY 512  /* > NOMSBC_PITCH_MAX_LAG + kernel half-width */

struct NomsbcAdaComb {
    float delay_line[MAX_DELAY];
    int   write_pos;
};

NomsbcAdaComb *nomsbc_adacomb_create(int max_frame_size)
{
    (void)max_frame_size;
    NomsbcAdaComb *ac = calloc(1, sizeof(*ac));
    return ac;
}

void nomsbc_adacomb_destroy(NomsbcAdaComb *ac)
{
    free(ac);
}

void nomsbc_adacomb_process(NomsbcAdaComb *ac,
                            float *signal, int frame_size,
                            int pitch_lag,
                            const AdaCombParams *params)
{
    int half = NOMSBC_ADACOMB_KERNEL_SIZE / 2;

    for (int n = 0; n < frame_size; n++) {
        /* Read the current sample into the delay line */
        ac->delay_line[ac->write_pos] = signal[n];

        /* Compute comb filter contribution */
        float comb = 0.0f;
        for (int k = -half; k <= half; k++) {
            int idx = ac->write_pos - pitch_lag + k;
            while (idx < 0) idx += MAX_DELAY;
            idx %= MAX_DELAY;
            comb += params->kernel[k + half] * ac->delay_line[idx];
        }

        signal[n] += params->global_gain * comb;

        ac->write_pos = (ac->write_pos + 1) % MAX_DELAY;
    }
}
