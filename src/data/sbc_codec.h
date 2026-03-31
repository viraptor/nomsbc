#ifndef NOMSBC_SBC_CODEC_H
#define NOMSBC_SBC_CODEC_H

/*
 * Wrapper around libsbc for mSBC encode/decode.
 *
 * mSBC is a specific SBC configuration mandated by the HFP spec:
 *   - 16 kHz sample rate
 *   - 1 channel (mono)
 *   - 8 subbands
 *   - 15 blocks
 *   - Loudness allocation
 *   - Bitpool 26 (fixed)
 *   - Frame size: 57 bytes (including 1-byte sync + 2-byte header)
 *
 * This produces ~62 kbps, decoded to 120 samples (7.5 ms) per frame.
 */

#include <stddef.h>
#include <stdint.h>

#define NOMSBC_SBC_FRAME_SAMPLES  120   /* samples per mSBC frame */
#define NOMSBC_SBC_FRAME_BYTES    57    /* encoded frame size */

typedef struct NomsbcSBC NomsbcSBC;

NomsbcSBC *nomsbc_sbc_create(void);
void       nomsbc_sbc_destroy(NomsbcSBC *s);

/*
 * Encode PCM samples to mSBC.
 *   pcm:       120 int16 samples at 16 kHz
 *   out:       buffer for encoded frame (at least 57 bytes)
 *   out_len:   actual encoded length
 *   Returns 0 on success.
 */
int nomsbc_sbc_encode(NomsbcSBC *s,
                      const int16_t *pcm, int num_samples,
                      uint8_t *out, int *out_len);

/*
 * Decode mSBC frame to PCM.
 *   data:      encoded frame (57 bytes)
 *   data_len:  length of data
 *   pcm:       output buffer for decoded PCM (at least 120 samples)
 *   num_samples: actual decoded sample count
 *   Returns 0 on success.
 */
int nomsbc_sbc_decode(NomsbcSBC *s,
                      const uint8_t *data, int data_len,
                      int16_t *pcm, int *num_samples);

#endif
