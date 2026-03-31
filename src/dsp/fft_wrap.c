#include "dsp/fft_wrap.h"
#include <kiss_fftr.h>
#include <stdlib.h>
#include <string.h>

struct NomsbcFFT {
    int             nfft;
    kiss_fftr_cfg   fwd;
    kiss_fftr_cfg   inv;
};

NomsbcFFT *nomsbc_fft_create(int nfft)
{
    NomsbcFFT *f = calloc(1, sizeof(*f));
    if (!f) return NULL;
    f->nfft = nfft;
    f->fwd  = kiss_fftr_alloc(nfft, 0, NULL, NULL);
    f->inv  = kiss_fftr_alloc(nfft, 1, NULL, NULL);
    if (!f->fwd || !f->inv) {
        nomsbc_fft_destroy(f);
        return NULL;
    }
    return f;
}

void nomsbc_fft_destroy(NomsbcFFT *f)
{
    if (!f) return;
    free(f->fwd);
    free(f->inv);
    free(f);
}

void nomsbc_fft_forward(NomsbcFFT *f, const float *in, float *out)
{
    /* kiss_fft_cpx is {float r, i} -- layout-compatible with interleaved. */
    kiss_fftr(f->fwd, in, (kiss_fft_cpx *)out);
}

void nomsbc_fft_inverse(NomsbcFFT *f, const float *in, float *out)
{
    kiss_fftri(f->inv, (const kiss_fft_cpx *)in, out);
    /* kiss_fft inverse is unnormalized -- scale by 1/nfft. */
    float scale = 1.0f / f->nfft;
    for (int i = 0; i < f->nfft; i++)
        out[i] *= scale;
}

void nomsbc_fft_power_spectrum(NomsbcFFT *f, const float *in, float *out)
{
    int nbins = f->nfft / 2 + 1;
    /* Use stack alloc for small sizes, heap for large. */
    float buf[2 * nbins];
    nomsbc_fft_forward(f, in, buf);
    for (int i = 0; i < nbins; i++)
        out[i] = buf[2*i] * buf[2*i] + buf[2*i+1] * buf[2*i+1];
}
