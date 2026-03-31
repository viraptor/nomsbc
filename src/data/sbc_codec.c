#include "data/sbc_codec.h"
#include <sbc/sbc.h>
#include <stdlib.h>
#include <string.h>

struct NomsbcSBC {
    sbc_t encoder;
    sbc_t decoder;
};

NomsbcSBC *nomsbc_sbc_create(void)
{
    NomsbcSBC *s = calloc(1, sizeof(*s));
    if (!s) return NULL;

    if (sbc_init(&s->encoder, 0) < 0 ||
        sbc_init(&s->decoder, 0) < 0) {
        free(s);
        return NULL;
    }

    /* Configure for mSBC */
    s->encoder.frequency  = SBC_FREQ_16000;
    s->encoder.mode       = SBC_MODE_MONO;
    s->encoder.subbands   = SBC_SB_8;
    s->encoder.blocks     = SBC_BLK_15;
    s->encoder.bitpool    = 26;
    s->encoder.allocation = SBC_AM_LOUDNESS;
    s->encoder.endian     = SBC_LE;

    /* Decoder auto-configures from the bitstream header */

    return s;
}

void nomsbc_sbc_destroy(NomsbcSBC *s)
{
    if (!s) return;
    sbc_finish(&s->encoder);
    sbc_finish(&s->decoder);
    free(s);
}

int nomsbc_sbc_encode(NomsbcSBC *s,
                      const int16_t *pcm, int num_samples,
                      uint8_t *out, int *out_len)
{
    ssize_t written = 0;
    ssize_t consumed = sbc_encode(&s->encoder,
                                  pcm, num_samples * sizeof(int16_t),
                                  out, NOMSBC_SBC_FRAME_BYTES,
                                  &written);
    if (consumed < 0) return -1;
    *out_len = (int)written;
    return 0;
}

int nomsbc_sbc_decode(NomsbcSBC *s,
                      const uint8_t *data, int data_len,
                      int16_t *pcm, int *num_samples)
{
    size_t written = 0;
    ssize_t consumed = sbc_decode(&s->decoder,
                                  data, data_len,
                                  pcm, NOMSBC_SBC_FRAME_SAMPLES * sizeof(int16_t),
                                  &written);
    if (consumed < 0) return -1;
    *num_samples = (int)(written / sizeof(int16_t));
    return 0;
}
