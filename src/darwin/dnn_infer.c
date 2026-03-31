#include "darwin/dnn_infer.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* --- Architecture dimensions --- */

#define FEAT_DIM      NOMSBC_FEATURE_DIM   /* 28 */
#define ENC_H         NOMSBC_ENC_HIDDEN    /* 128 */
#define GRU_H         NOMSBC_GRU_HIDDEN    /* 192 */
#define GRU_LAYERS    NOMSBC_GRU_LAYERS    /* 2 */

/* PARAM_DIM: must match model.py */
#define PARAM_DIM  ((NOMSBC_ADACOMB_KERNEL_SIZE + 1) * 2       \
                    + NOMSBC_ADACONV_KERNEL_SIZE + 1            \
                    + (NOMSBC_ADASHAPE_NUM_BASES                \
                       + NOMSBC_ADASHAPE_SHAPE_DIM + 1) * 2    \
                    + NOMSBC_BWE_ENVELOPE_DIM + 1 + 1)

/* --- Linear layer --- */

typedef struct {
    const float *W;   /* (out_dim, in_dim) row-major */
    const float *b;   /* (out_dim,) */
    int out_dim;
    int in_dim;
} Linear;

static void linear_forward(const Linear *l, const float *x, float *y)
{
    for (int i = 0; i < l->out_dim; i++) {
        float sum = l->b[i];
        const float *row = l->W + i * l->in_dim;
        for (int j = 0; j < l->in_dim; j++)
            sum += row[j] * x[j];
        y[i] = sum;
    }
}

/* --- GRU layer --- */

typedef struct {
    const float *W_ih;   /* (3*hidden, input) */
    const float *W_hh;   /* (3*hidden, hidden) */
    const float *b_ih;   /* (3*hidden,) */
    const float *b_hh;   /* (3*hidden,) */
    int input_dim;
    int hidden_dim;
} GRULayer;

static float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

/*
 * One GRU step matching PyTorch's default GRU equations:
 *   r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
 *   z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
 *   n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
 *   h' = (1 - z) * n + z * h
 */
static void gru_step(const GRULayer *g, const float *x, float *h)
{
    int H = g->hidden_dim;
    int I = g->input_dim;

    /* Compute W_ih @ x + b_ih  and  W_hh @ h + b_hh */
    float *gx = calloc(3 * H, sizeof(float));
    float *gh = calloc(3 * H, sizeof(float));
    if (!gx || !gh) { free(gx); free(gh); return; }

    for (int i = 0; i < 3 * H; i++) {
        float sx = g->b_ih[i];
        const float *row_x = g->W_ih + i * I;
        for (int j = 0; j < I; j++)
            sx += row_x[j] * x[j];
        gx[i] = sx;

        float sh = g->b_hh[i];
        const float *row_h = g->W_hh + i * H;
        for (int j = 0; j < H; j++)
            sh += row_h[j] * h[j];
        gh[i] = sh;
    }

    /* Gates */
    for (int i = 0; i < H; i++) {
        float r = sigmoidf(gx[i] + gh[i]);              /* reset */
        float z = sigmoidf(gx[H + i] + gh[H + i]);      /* update */
        float n = tanhf(gx[2*H + i] + r * gh[2*H + i]); /* new */
        h[i] = (1.0f - z) * n + z * h[i];
    }

    free(gx);
    free(gh);
}

/* --- DNN state --- */

struct NomsbcDNN {
    /* Encoder */
    Linear enc_l1;   /* (28 -> 128) */
    Linear enc_l2;   /* (128 -> 128) */

    /* GRU */
    GRULayer gru[GRU_LAYERS];
    float    gru_state[GRU_LAYERS][GRU_H];

    /* Head */
    Linear head;     /* (192 -> PARAM_DIM) */
};

/* Helper: find a layer or fail */
static const float *find_layer(const NomsbcWeights *w, const char *name,
                               int expected)
{
    const NomsbcWeightLayer *l = nomsbc_weights_find(w, name);
    if (!l) {
        fprintf(stderr, "nomsbc_dnn: missing layer '%s'\n", name);
        return NULL;
    }
    if (l->count != expected) {
        fprintf(stderr, "nomsbc_dnn: layer '%s' has %d params, expected %d\n",
                name, l->count, expected);
        return NULL;
    }
    return l->data;
}

NomsbcDNN *nomsbc_dnn_create(const NomsbcWeights *weights)
{
    NomsbcDNN *d = calloc(1, sizeof(*d));
    if (!d) return NULL;

    /* FeatureEncoder: encoder.net.0 = Linear(28,128), encoder.net.2 = Linear(128,128) */
    const float *p;

    if (!(p = find_layer(weights, "encoder.net.0.weight", ENC_H * FEAT_DIM))) goto fail;
    d->enc_l1.W = p;  d->enc_l1.out_dim = ENC_H;  d->enc_l1.in_dim = FEAT_DIM;

    if (!(p = find_layer(weights, "encoder.net.0.bias", ENC_H))) goto fail;
    d->enc_l1.b = p;

    if (!(p = find_layer(weights, "encoder.net.2.weight", ENC_H * ENC_H))) goto fail;
    d->enc_l2.W = p;  d->enc_l2.out_dim = ENC_H;  d->enc_l2.in_dim = ENC_H;

    if (!(p = find_layer(weights, "encoder.net.2.bias", ENC_H))) goto fail;
    d->enc_l2.b = p;

    /* GRU layers */
    const char *gru_names[][4] = {
        { "gru.gru.weight_ih_l0", "gru.gru.weight_hh_l0",
          "gru.gru.bias_ih_l0",   "gru.gru.bias_hh_l0" },
        { "gru.gru.weight_ih_l1", "gru.gru.weight_hh_l1",
          "gru.gru.bias_ih_l1",   "gru.gru.bias_hh_l1" },
    };
    int gru_input_dims[GRU_LAYERS] = { ENC_H, GRU_H };

    for (int l = 0; l < GRU_LAYERS; l++) {
        int idim = gru_input_dims[l];
        d->gru[l].input_dim  = idim;
        d->gru[l].hidden_dim = GRU_H;

        if (!(p = find_layer(weights, gru_names[l][0], 3*GRU_H*idim))) goto fail;
        d->gru[l].W_ih = p;
        if (!(p = find_layer(weights, gru_names[l][1], 3*GRU_H*GRU_H))) goto fail;
        d->gru[l].W_hh = p;
        if (!(p = find_layer(weights, gru_names[l][2], 3*GRU_H))) goto fail;
        d->gru[l].b_ih = p;
        if (!(p = find_layer(weights, gru_names[l][3], 3*GRU_H))) goto fail;
        d->gru[l].b_hh = p;
    }

    /* ParameterHead: head.linear = Linear(192, PARAM_DIM) */
    if (!(p = find_layer(weights, "head.linear.weight", PARAM_DIM * GRU_H))) goto fail;
    d->head.W = p;  d->head.out_dim = PARAM_DIM;  d->head.in_dim = GRU_H;

    if (!(p = find_layer(weights, "head.linear.bias", PARAM_DIM))) goto fail;
    d->head.b = p;

    return d;

fail:
    free(d);
    return NULL;
}

void nomsbc_dnn_destroy(NomsbcDNN *dnn)
{
    free(dnn);
}

void nomsbc_dnn_reset(NomsbcDNN *dnn)
{
    memset(dnn->gru_state, 0, sizeof(dnn->gru_state));
}

/* --- Unpack raw parameters into EnhanceFrameParams --- */

static void softmax(float *x, int n)
{
    float max_v = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max_v) max_v = x[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_v);
        sum += x[i];
    }
    for (int i = 0; i < n; i++)
        x[i] /= sum;
}

static void unpack_params(const float *raw, EnhanceFrameParams *p)
{
    int idx = 0;

    /* comb1 */
    memcpy(p->comb1.kernel, raw + idx, NOMSBC_ADACOMB_KERNEL_SIZE * sizeof(float));
    idx += NOMSBC_ADACOMB_KERNEL_SIZE;
    p->comb1.global_gain = sigmoidf(raw[idx++]);

    /* comb2 */
    memcpy(p->comb2.kernel, raw + idx, NOMSBC_ADACOMB_KERNEL_SIZE * sizeof(float));
    idx += NOMSBC_ADACOMB_KERNEL_SIZE;
    p->comb2.global_gain = sigmoidf(raw[idx++]);

    /* conv */
    memcpy(p->conv.kernel, raw + idx, NOMSBC_ADACONV_KERNEL_SIZE * sizeof(float));
    idx += NOMSBC_ADACONV_KERNEL_SIZE;
    p->conv.gain = sigmoidf(raw[idx++]) * 2.0f;

    /* shape1 */
    memcpy(p->shape1.select_weights, raw + idx, NOMSBC_ADASHAPE_NUM_BASES * sizeof(float));
    idx += NOMSBC_ADASHAPE_NUM_BASES;
    softmax(p->shape1.select_weights, NOMSBC_ADASHAPE_NUM_BASES);
    memcpy(p->shape1.shape_params, raw + idx, NOMSBC_ADASHAPE_SHAPE_DIM * sizeof(float));
    idx += NOMSBC_ADASHAPE_SHAPE_DIM;
    p->shape1.mix_gain = sigmoidf(raw[idx++]);

    /* shape2 */
    memcpy(p->shape2.select_weights, raw + idx, NOMSBC_ADASHAPE_NUM_BASES * sizeof(float));
    idx += NOMSBC_ADASHAPE_NUM_BASES;
    softmax(p->shape2.select_weights, NOMSBC_ADASHAPE_NUM_BASES);
    memcpy(p->shape2.shape_params, raw + idx, NOMSBC_ADASHAPE_SHAPE_DIM * sizeof(float));
    idx += NOMSBC_ADASHAPE_SHAPE_DIM;
    p->shape2.mix_gain = sigmoidf(raw[idx++]);

    /* bwe */
    memcpy(p->bwe.envelope, raw + idx, NOMSBC_BWE_ENVELOPE_DIM * sizeof(float));
    idx += NOMSBC_BWE_ENVELOPE_DIM;
    p->bwe.excitation_gain = sigmoidf(raw[idx++]);
    p->bwe.voicing_factor  = sigmoidf(raw[idx++]);
}

/* --- Inference --- */

void nomsbc_dnn_infer(const float *features, int feat_dim,
                      EnhanceFrameParams *params, void *user_data)
{
    NomsbcDNN *d = (NomsbcDNN *)user_data;
    (void)feat_dim;

    /* 1. FeatureEncoder: Linear + Tanh + Linear + Tanh */
    float enc1[ENC_H], enc2[ENC_H];
    linear_forward(&d->enc_l1, features, enc1);
    for (int i = 0; i < ENC_H; i++) enc1[i] = tanhf(enc1[i]);
    linear_forward(&d->enc_l2, enc1, enc2);
    for (int i = 0; i < ENC_H; i++) enc2[i] = tanhf(enc2[i]);

    /* 2. GRU: 2 layers */

    /* Layer 0: input is encoder output (ENC_H dims) */
    gru_step(&d->gru[0], enc2, d->gru_state[0]);

    /* Layer 1: input is layer 0 hidden state */
    gru_step(&d->gru[1], d->gru_state[0], d->gru_state[1]);

    /* 3. ParameterHead */
    float raw[PARAM_DIM];
    linear_forward(&d->head, d->gru_state[1], raw);

    /* 4. Unpack with activations */
    unpack_params(raw, params);
}
