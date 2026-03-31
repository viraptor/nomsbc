#include "darwin/audio_tap.h"
#include "darwin/weights.h"
#include "darwin/dnn_infer.h"
#include "darwin/speech_detect.h"
#include "pipeline/enhance.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/*  Resampler: 48 kHz <-> 16 kHz (factor-of-3 decimation/interp)     */
/* ------------------------------------------------------------------ */

/*
 * Simple polyphase FIR lowpass for 3:1 decimation / 1:3 interpolation.
 * Cutoff at 8 kHz (1/3 of 48 kHz Nyquist), 48-tap windowed-sinc.
 */
#define RESAMP_TAPS    48
#define RESAMP_FACTOR  3

typedef struct {
    float coeffs[RESAMP_TAPS];
    float history[RESAMP_TAPS];   /* ring buffer for input samples */
    int   pos;
} Resampler;

static void resampler_design_lowpass(float *h, int n, float fc_norm)
{
    int mid = n / 2;
    for (int i = 0; i < n; i++) {
        float x = (float)(i - mid);
        /* sinc */
        float sinc;
        if (fabsf(x) < 1e-6f)
            sinc = 1.0f;
        else
            sinc = sinf((float)M_PI * fc_norm * x) / ((float)M_PI * x);
        /* Hamming window */
        float win = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * i / (n - 1));
        h[i] = fc_norm * sinc * win;
    }
}

static void resampler_init(Resampler *r)
{
    /* fc_norm = 2 * 8000 / 48000 = 1/3 */
    resampler_design_lowpass(r->coeffs, RESAMP_TAPS, 1.0f / 3.0f);
    memset(r->history, 0, sizeof(r->history));
    r->pos = 0;
}

static void resampler_reset(Resampler *r)
{
    memset(r->history, 0, sizeof(r->history));
    r->pos = 0;
}

/* Push one sample into the FIR history */
static void resampler_push(Resampler *r, float sample)
{
    r->history[r->pos] = sample;
    r->pos = (r->pos + 1) % RESAMP_TAPS;
}

/* Compute one FIR output from current history */
static float resampler_convolve(const Resampler *r)
{
    float sum = 0.0f;
    int idx = r->pos;  /* oldest sample */
    for (int i = 0; i < RESAMP_TAPS; i++) {
        sum += r->coeffs[i] * r->history[idx];
        idx = (idx + 1) % RESAMP_TAPS;
    }
    return sum;
}

/*
 * Decimate: in_count samples at 48 kHz -> out_count samples at 16 kHz.
 * out_count = in_count / 3.  Caller ensures in_count is a multiple of 3.
 */
static void resample_decimate(Resampler *r, const float *in, float *out,
                              int in_count)
{
    int j = 0;
    for (int i = 0; i < in_count; i++) {
        resampler_push(r, in[i]);
        if ((i % RESAMP_FACTOR) == RESAMP_FACTOR - 1)
            out[j++] = resampler_convolve(r);
    }
}

/*
 * Interpolate: in_count samples at 16 kHz -> out_count samples at 48 kHz.
 * out_count = in_count * 3.
 */
static void resample_interpolate(Resampler *r, const float *in, float *out,
                                 int in_count)
{
    int j = 0;
    for (int i = 0; i < in_count; i++) {
        /* Insert sample then 2 zeros (zero-stuffing for interpolation) */
        resampler_push(r, in[i] * RESAMP_FACTOR);
        out[j++] = resampler_convolve(r);
        resampler_push(r, 0.0f);
        out[j++] = resampler_convolve(r);
        resampler_push(r, 0.0f);
        out[j++] = resampler_convolve(r);
    }
}

/* ------------------------------------------------------------------ */
/*  Tap state                                                         */
/* ------------------------------------------------------------------ */

/*
 * Maximum system-rate frame we handle in one callback.
 * 480 samples = 10 ms at 48 kHz, which maps to 160 samples at 16 kHz
 * (one nomsbc frame).
 */
#define MAX_SYS_FRAME     4800    /* up to 100 ms at 48 kHz */
#define MAX_16K_FRAME     1600    /* up to 100 ms at 16 kHz */

struct NomsbcAudioTap {
    /* CoreAudio objects */
    AudioObjectID       original_device;
    AudioObjectID       tap_object;
    AudioObjectID       aggregate_device;
    AudioDeviceIOProcID io_proc_id;
    bool                running;

    /* Processing */
    NomsbcWeights       *weights;
    NomsbcDNN           *dnn;
    NomsbcEnhancer      *enhancer;
    NomsbcSpeechDetect  *detector;

    /* Resamplers */
    Resampler            decim;     /* 48 kHz -> 16 kHz */
    Resampler            interp;    /* 16 kHz -> 48 kHz */

    /* Stream format */
    int                  sys_rate;
    int                  sys_frame_size;   /* samples per callback frame */

    /* Current state (read from non-audio thread) */
    volatile bool        enhancing;
};

/* ------------------------------------------------------------------ */
/*  Audio processing callback                                         */
/* ------------------------------------------------------------------ */

static OSStatus tap_io_proc(AudioObjectID           device,
                            const AudioTimeStamp    *now,
                            const AudioBufferList   *input_data,
                            const AudioTimeStamp    *input_time,
                            AudioBufferList         *output_data,
                            const AudioTimeStamp    *output_time,
                            void                    *user_data)
{
    (void)device; (void)now; (void)input_time; (void)output_time;

    NomsbcAudioTap *tap = (NomsbcAudioTap *)user_data;

    /* Process each buffer (typically one for mono/interleaved stereo) */
    for (UInt32 buf_idx = 0; buf_idx < output_data->mNumberBuffers; buf_idx++) {
        AudioBuffer *abuf = &output_data->mBuffers[buf_idx];
        float *samples = (float *)abuf->mData;
        int n_frames = abuf->mDataByteSize / (sizeof(float) * abuf->mNumberChannels);
        int channels = abuf->mNumberChannels;

        if (n_frames <= 0 || !samples) continue;

        /*
         * For speech detection + enhancement we work on channel 0
         * (mono speech from HFP is typically identical across channels).
         */
        float mono[MAX_SYS_FRAME];
        int count = n_frames < MAX_SYS_FRAME ? n_frames : MAX_SYS_FRAME;

        /* Extract channel 0 */
        if (channels == 1) {
            memcpy(mono, samples, count * sizeof(float));
        } else {
            for (int i = 0; i < count; i++)
                mono[i] = samples[i * channels];
        }

        /* Speech detection (operates at system rate) */
        bool is_speech = nomsbc_speech_detect_feed(tap->detector, mono);
        tap->enhancing = is_speech;

        if (!is_speech) {
            /* Passthrough: copy input to output unchanged */
            if (input_data && input_data->mNumberBuffers > buf_idx) {
                const AudioBuffer *ibuf = &input_data->mBuffers[buf_idx];
                if (ibuf->mData && ibuf->mDataByteSize == abuf->mDataByteSize)
                    memcpy(abuf->mData, ibuf->mData, abuf->mDataByteSize);
            }
            continue;
        }

        /* --- Enhancement path --- */

        /* Downsample channel 0: sys_rate -> 16 kHz */
        int n16k = count / RESAMP_FACTOR;
        float down[MAX_16K_FRAME];
        resample_decimate(&tap->decim, mono, down, count);

        /* Process through enhancer in 160-sample (10 ms) chunks */
        float enhanced[MAX_16K_FRAME];
        int pos = 0;
        while (pos + NOMSBC_FRAME_SIZE <= n16k) {
            nomsbc_enhancer_process_frame(tap->enhancer,
                                          down + pos,
                                          enhanced + pos);
            pos += NOMSBC_FRAME_SIZE;
        }
        /* Handle any trailing samples (< 1 frame): passthrough */
        if (pos < n16k)
            memcpy(enhanced + pos, down + pos, (n16k - pos) * sizeof(float));

        /* Upsample back: 16 kHz -> sys_rate */
        float up[MAX_SYS_FRAME];
        resample_interpolate(&tap->interp, enhanced, up, n16k);

        /* Write enhanced signal back to all channels */
        if (channels == 1) {
            memcpy(samples, up, count * sizeof(float));
        } else {
            for (int i = 0; i < count; i++) {
                for (int c = 0; c < channels; c++)
                    samples[i * channels + c] = up[i];
            }
        }
    }

    return noErr;
}

/* ------------------------------------------------------------------ */
/*  Device discovery helpers                                          */
/* ------------------------------------------------------------------ */

static AudioObjectID get_default_output_device(void)
{
    AudioObjectPropertyAddress addr = {
        kAudioHardwarePropertyDefaultOutputDevice,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    AudioObjectID dev = kAudioObjectUnknown;
    UInt32 size = sizeof(dev);
    OSStatus err = AudioObjectGetPropertyData(kAudioObjectSystemObject,
                                              &addr, 0, NULL, &size, &dev);
    if (err != noErr) {
        fprintf(stderr, "nomsbc: cannot get default output device (%d)\n", (int)err);
        return kAudioObjectUnknown;
    }
    return dev;
}

static AudioObjectID find_device_by_uid(const char *uid)
{
    CFStringRef cf_uid = CFStringCreateWithCString(kCFAllocatorDefault, uid,
                                                    kCFStringEncodingUTF8);
    if (!cf_uid) return kAudioObjectUnknown;

    AudioObjectPropertyAddress addr = {
        kAudioHardwarePropertyDeviceForUID,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };

    AudioValueTranslation trans;
    trans.mInputData      = &cf_uid;
    trans.mInputDataSize  = sizeof(CFStringRef);
    AudioObjectID dev     = kAudioObjectUnknown;
    trans.mOutputData     = &dev;
    trans.mOutputDataSize = sizeof(AudioObjectID);

    UInt32 size = sizeof(trans);
    AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr,
                               0, NULL, &size, &trans);
    CFRelease(cf_uid);
    return dev;
}

static int get_device_sample_rate(AudioObjectID dev)
{
    AudioObjectPropertyAddress addr = {
        kAudioDevicePropertyNominalSampleRate,
        kAudioObjectPropertyScopeOutput,
        kAudioObjectPropertyElementMain
    };
    Float64 rate = 0;
    UInt32 size = sizeof(rate);
    OSStatus err = AudioObjectGetPropertyData(dev, &addr, 0, NULL, &size, &rate);
    if (err != noErr) return 48000;  /* fallback */
    return (int)rate;
}

/* ------------------------------------------------------------------ */
/*  Aggregate device + process tap creation                           */
/* ------------------------------------------------------------------ */

/*
 * Create an aggregate device with a mutating process tap on the
 * target output device.  The aggregate becomes the new default output
 * so apps' audio flows through our tap callback.
 */
static int create_aggregate_with_tap(NomsbcAudioTap *tap)
{
    OSStatus err;

    /* --- Build tap description --- */
    CFStringRef tap_uid = CFStringCreateWithCString(
        kCFAllocatorDefault, "com.nomsbc.speech-tap", kCFStringEncodingUTF8);

    CFMutableDictionaryRef tap_desc = CFDictionaryCreateMutable(
        kCFAllocatorDefault, 0,
        &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks);

    CFDictionarySetValue(tap_desc,
        CFSTR(kAudioAggregateDeviceUIDKey), tap_uid);

    /* The device to tap */
    AudioObjectPropertyAddress uid_addr = {
        kAudioDevicePropertyDeviceUID,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    CFStringRef orig_uid = NULL;
    UInt32 size = sizeof(orig_uid);
    err = AudioObjectGetPropertyData(tap->original_device, &uid_addr,
                                     0, NULL, &size, &orig_uid);
    if (err != noErr) {
        fprintf(stderr, "nomsbc: cannot get device UID (%d)\n", (int)err);
        CFRelease(tap_desc);
        CFRelease(tap_uid);
        return -1;
    }

    /* Aggregate device name */
    CFDictionarySetValue(tap_desc,
        CFSTR(kAudioAggregateDeviceNameKey),
        CFSTR("nomsbc Enhanced Output"));

    /* Sub-device list: the original output */
    CFMutableArrayRef sub_list = CFArrayCreateMutable(
        kCFAllocatorDefault, 1, &kCFTypeArrayCallBacks);
    CFArrayAppendValue(sub_list, orig_uid);
    CFDictionarySetValue(tap_desc,
        CFSTR(kAudioAggregateDeviceSubDeviceListKey), sub_list);
    CFDictionarySetValue(tap_desc,
        CFSTR(kAudioAggregateDeviceMainSubDeviceKey), orig_uid);

    /* Mark as stacked (output-only aggregate) */
    int is_stacked = 1;
    CFNumberRef stacked_num = CFNumberCreate(kCFAllocatorDefault,
                                             kCFNumberIntType, &is_stacked);
    CFDictionarySetValue(tap_desc,
        CFSTR(kAudioAggregateDeviceIsStackedKey), stacked_num);

    /* Create the aggregate device */
    err = AudioHardwareCreateAggregateDevice(tap_desc, &tap->aggregate_device);
    CFRelease(sub_list);
    CFRelease(stacked_num);
    CFRelease(tap_desc);
    CFRelease(tap_uid);
    if (orig_uid) CFRelease(orig_uid);

    if (err != noErr) {
        fprintf(stderr, "nomsbc: cannot create aggregate device (%d)\n", (int)err);
        return -1;
    }

    /* Install our IO proc on the aggregate device */
    err = AudioDeviceCreateIOProcID(tap->aggregate_device,
                                    tap_io_proc, tap,
                                    &tap->io_proc_id);
    if (err != noErr) {
        fprintf(stderr, "nomsbc: cannot install IO proc (%d)\n", (int)err);
        return -1;
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/*  Public API                                                        */
/* ------------------------------------------------------------------ */

NomsbcAudioTap *nomsbc_audio_tap_create(const char *device_uid,
                                         const char *weights_path)
{
    NomsbcAudioTap *tap = calloc(1, sizeof(*tap));
    if (!tap) return NULL;

    /* Find target device */
    if (device_uid)
        tap->original_device = find_device_by_uid(device_uid);
    else
        tap->original_device = get_default_output_device();

    if (tap->original_device == kAudioObjectUnknown) {
        fprintf(stderr, "nomsbc: no output device found\n");
        goto fail;
    }

    tap->sys_rate = get_device_sample_rate(tap->original_device);
    fprintf(stderr, "nomsbc: target device sample rate: %d Hz\n", tap->sys_rate);

    if (tap->sys_rate % NOMSBC_SAMPLE_RATE != 0) {
        fprintf(stderr, "nomsbc: device rate %d is not a multiple of %d\n",
                tap->sys_rate, NOMSBC_SAMPLE_RATE);
        goto fail;
    }

    /* 10 ms frame at system rate */
    tap->sys_frame_size = tap->sys_rate / 100;

    /* Load model weights */
    tap->weights = nomsbc_weights_load(weights_path);
    if (!tap->weights) goto fail;

    /* Create DNN inference engine */
    tap->dnn = nomsbc_dnn_create(tap->weights);
    if (!tap->dnn) goto fail;

    /* Create enhancement pipeline (uses DNN as callback) */
    tap->enhancer = nomsbc_enhancer_create(nomsbc_dnn_infer, tap->dnn);
    if (!tap->enhancer) goto fail;

    /* Create speech detector */
    tap->detector = nomsbc_speech_detect_create(tap->sys_rate,
                                                 tap->sys_frame_size);
    if (!tap->detector) goto fail;

    /* Initialise resamplers */
    resampler_init(&tap->decim);
    resampler_init(&tap->interp);

    /* Set up CoreAudio aggregate device + tap */
    if (create_aggregate_with_tap(tap) != 0)
        goto fail;

    return tap;

fail:
    nomsbc_audio_tap_destroy(tap);
    return NULL;
}

int nomsbc_audio_tap_start(NomsbcAudioTap *tap)
{
    if (tap->running) return 0;

    OSStatus err = AudioDeviceStart(tap->aggregate_device, tap->io_proc_id);
    if (err != noErr) {
        fprintf(stderr, "nomsbc: AudioDeviceStart failed (%d)\n", (int)err);
        return -1;
    }
    tap->running = true;
    fprintf(stderr, "nomsbc: audio tap started\n");
    return 0;
}

void nomsbc_audio_tap_stop(NomsbcAudioTap *tap)
{
    if (!tap || !tap->running) return;
    AudioDeviceStop(tap->aggregate_device, tap->io_proc_id);
    tap->running = false;
    fprintf(stderr, "nomsbc: audio tap stopped\n");
}

void nomsbc_audio_tap_destroy(NomsbcAudioTap *tap)
{
    if (!tap) return;

    nomsbc_audio_tap_stop(tap);

    if (tap->io_proc_id)
        AudioDeviceDestroyIOProcID(tap->aggregate_device, tap->io_proc_id);

    if (tap->aggregate_device != kAudioObjectUnknown)
        AudioHardwareDestroyAggregateDevice(tap->aggregate_device);

    nomsbc_speech_detect_destroy(tap->detector);
    nomsbc_enhancer_destroy(tap->enhancer);
    nomsbc_dnn_destroy(tap->dnn);
    nomsbc_weights_destroy(tap->weights);

    free(tap);
}

bool nomsbc_audio_tap_is_enhancing(const NomsbcAudioTap *tap)
{
    return tap->enhancing;
}
