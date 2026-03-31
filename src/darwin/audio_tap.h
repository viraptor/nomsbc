#ifndef NOMSBC_AUDIO_TAP_H
#define NOMSBC_AUDIO_TAP_H

/*
 * CoreAudio output device tap for Darwin.
 *
 * Creates an aggregate device that wraps the target output device with
 * a process tap.  Audio flowing to the aggregate is intercepted in the
 * tap callback, where it is analysed for mSBC-like speech.  When
 * detected, frames are downsampled to 16 kHz, enhanced through the
 * nomsbc pipeline, and upsampled back to the system rate.
 *
 * Requires macOS 14.2+ for AudioHardwareCreateProcessTap.
 */

#include <stdbool.h>
#include <CoreAudio/CoreAudio.h>
#include <AudioToolbox/AudioToolbox.h>

typedef struct NomsbcAudioTap NomsbcAudioTap;

/*
 * Create an audio tap on the given output device.
 *   device_uid:   CoreAudio device UID string, or NULL for default output.
 *   weights_path: path to the nomsbc_weights.bin file.
 * Returns NULL on failure.
 */
NomsbcAudioTap *nomsbc_audio_tap_create(const char *device_uid,
                                         const char *weights_path);

/* Start processing audio. */
int  nomsbc_audio_tap_start(NomsbcAudioTap *tap);

/* Stop processing (idempotent). */
void nomsbc_audio_tap_stop(NomsbcAudioTap *tap);

/* Tear down everything. */
void nomsbc_audio_tap_destroy(NomsbcAudioTap *tap);

/* Query whether the tap is currently enhancing (vs passthrough). */
bool nomsbc_audio_tap_is_enhancing(const NomsbcAudioTap *tap);

#endif
