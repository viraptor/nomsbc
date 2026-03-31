/*
 * nomsbc_darwin: real-time mSBC speech enhancement for macOS.
 *
 * Creates an aggregate audio output device that taps into the original
 * output.  Audio is analysed for bandwidth-limited codec characteristics
 * (like mSBC from Bluetooth HFP).  When detected, frames are downsampled
 * to 16 kHz, enhanced through the NoLACE-mSBC DDSP pipeline, and
 * upsampled back to the system rate.  Non-speech audio passes through
 * unmodified.
 *
 * Usage:
 *   nomsbc_darwin -w nomsbc_weights.bin [-d <device-uid>]
 */

#include "darwin/audio_tap.h"
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>

static volatile sig_atomic_t g_running = 1;

static void sighandler(int sig)
{
    (void)sig;
    g_running = 0;
}

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s -w <weights.bin> [-d <device-uid>]\n"
        "\n"
        "Options:\n"
        "  -w PATH   Path to nomsbc_weights.bin (required)\n"
        "  -d UID    CoreAudio output device UID (default: system default)\n"
        "  -h        Show this help\n"
        "\n"
        "The utility creates an aggregate output device that intercepts\n"
        "audio, detects mSBC-encoded speech, and enhances it in real time.\n"
        "Press Ctrl-C to stop.\n",
        prog);
}

int main(int argc, char **argv)
{
    const char *weights_path = NULL;
    const char *device_uid   = NULL;

    int opt;
    while ((opt = getopt(argc, argv, "w:d:h")) != -1) {
        switch (opt) {
        case 'w': weights_path = optarg; break;
        case 'd': device_uid   = optarg; break;
        case 'h':
        default:
            usage(argv[0]);
            return opt == 'h' ? 0 : 1;
        }
    }

    if (!weights_path) {
        fprintf(stderr, "Error: -w <weights.bin> is required\n\n");
        usage(argv[0]);
        return 1;
    }

    /* Set up signal handling for clean shutdown */
    signal(SIGINT,  sighandler);
    signal(SIGTERM, sighandler);

    fprintf(stderr, "nomsbc: loading weights from %s\n", weights_path);

    NomsbcAudioTap *tap = nomsbc_audio_tap_create(device_uid, weights_path);
    if (!tap) {
        fprintf(stderr, "nomsbc: failed to create audio tap\n");
        return 1;
    }

    if (nomsbc_audio_tap_start(tap) != 0) {
        nomsbc_audio_tap_destroy(tap);
        return 1;
    }

    fprintf(stderr, "nomsbc: running (Ctrl-C to stop)\n");

    /* Main loop: just wait, audio processing happens in the IO callback */
    while (g_running) {
        sleep(1);

        /* Periodic status (to stderr so it doesn't pollute piped output) */
        fprintf(stderr, "\rnomsbc: %s",
                nomsbc_audio_tap_is_enhancing(tap)
                    ? "enhancing (mSBC speech detected)"
                    : "passthrough                     ");
    }

    fprintf(stderr, "\nnomsbc: shutting down\n");

    nomsbc_audio_tap_stop(tap);
    nomsbc_audio_tap_destroy(tap);

    return 0;
}
