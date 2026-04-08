#include "data/sbc_codec.h"
#include <sbc.h>
#include <stdlib.h>
#include <string.h>

struct NomsbcSBC {
    sbc_t encoder;
    sbc_t decoder;
    struct sbc_frame frame;
};

NomsbcSBC *nomsbc_sbc_create(void)
{
    NomsbcSBC *s = calloc(1, sizeof(*s));
    if (!s) return NULL;

    sbc_reset(&s->encoder);
    sbc_reset(&s->decoder);

    /* mSBC frame description (HFP Appendix A). */
    s->frame.msbc = true;
    s->frame.freq = SBC_FREQ_16K;
    s->frame.mode = SBC_MODE_MONO;
    s->frame.bam  = SBC_BAM_LOUDNESS;
    s->frame.nblocks   = 15;
    s->frame.nsubbands = 8;
    s->frame.bitpool   = 26;

    return s;
}

void nomsbc_sbc_destroy(NomsbcSBC *s)
{
    free(s);
}

int nomsbc_sbc_encode(NomsbcSBC *s,
                      const int16_t *pcm, int num_samples,
                      uint8_t *out, int *out_len)
{
    if (num_samples != SBC_MSBC_SAMPLES) return -1;
    if (sbc_encode(&s->encoder, pcm, 1, NULL, 0,
                   &s->frame, out, NOMSBC_SBC_FRAME_BYTES) < 0)
        return -1;
    *out_len = NOMSBC_SBC_FRAME_BYTES;
    return 0;
}

int nomsbc_sbc_decode(NomsbcSBC *s,
                      const uint8_t *data, int data_len,
                      int16_t *pcm, int *num_samples)
{
    struct sbc_frame f;
    if (data_len < (int)NOMSBC_SBC_FRAME_BYTES) return -1;
    if (sbc_decode(&s->decoder, data, data_len, &f,
                   pcm, 1, NULL, 0) < 0)
        return -1;
    *num_samples = f.nblocks * f.nsubbands;
    return 0;
}
