#ifndef NOMSBC_SPEECH_DETECT_H
#define NOMSBC_SPEECH_DETECT_H

/*
 * Detect whether an audio stream is likely playing speech from a
 * bandwidth-limited codec such as mSBC (Bluetooth HFP).
 *
 * Detection strategy:
 *   - Compute a short FFT of the incoming audio at system sample rate.
 *   - Measure the ratio of spectral energy above ~8 kHz to total energy.
 *   - mSBC is limited to 0-8 kHz; fullband audio has significant energy
 *     above 8 kHz, so a low high-band ratio signals codec-limited audio.
 *   - Also require non-trivial total energy (reject silence).
 *
 * To avoid rapid switching between passthrough and enhancement, a
 * hysteresis counter requires multiple consecutive frames of consistent
 * detection before the mode actually changes.
 *
 * The detector runs every CHECK_INTERVAL frames to save CPU.
 */

#include <stdbool.h>

typedef struct NomsbcSpeechDetect NomsbcSpeechDetect;

/*
 * Create a speech detector.
 *   sample_rate:  system audio sample rate (e.g. 48000)
 *   frame_size:   samples per audio callback frame at system rate
 */
NomsbcSpeechDetect *nomsbc_speech_detect_create(int sample_rate,
                                                 int frame_size);
void                nomsbc_speech_detect_destroy(NomsbcSpeechDetect *sd);

/*
 * Feed one frame of audio and return current detection state.
 *   samples:  frame_size float samples at system sample rate
 *   Returns true if the stream is currently classified as mSBC speech.
 *
 * The internal FFT analysis runs only every few frames; between checks
 * the previous result is returned.
 */
bool nomsbc_speech_detect_feed(NomsbcSpeechDetect *sd, const float *samples);

/*
 * Reset detector state (e.g. when switching output devices).
 */
void nomsbc_speech_detect_reset(NomsbcSpeechDetect *sd);

#endif
