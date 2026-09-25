/* Portable single-thread float32 inference for dynamic-routes-v2.
 * No Python extension ABI, BLAS, OpenMP, GPU, simulator, or runtime compiler.
 * Weight layout is defined in submission.py, and checked by the loader.
 */
#include <stddef.h>
#include <math.h>

enum { FEATURES = 96, STATE = 3200, WIDTH = 128 };
unsigned int route_weight_count(void) { return 471681u; }

static float dot(const float *a, const float *b, size_t n) {
    float v = 0.0f;
    for (size_t i = 0; i < n; ++i) v += a[i] * b[i];
    return v;
}
static void layer(const float *x, const float *w, const float *b,
                  size_t n, float *out) {
    for (size_t i = 0; i < WIDTH; ++i)
        out[i] = tanhf(dot(x, w + i * n, n) + b[i]);
}
void route_scores(const float *weights, const float *state,
                  const float *features, size_t count, float *scores) {
    const float *ew = weights;
    const float *eb = ew + WIDTH * FEATURES;
    const float *c0w = eb + WIDTH;
    const float *c0b = c0w + WIDTH * STATE;
    const float *c2w = c0b + WIDTH;
    const float *c2b = c2w + WIDTH * WIDTH;
    const float *a0w = c2b + WIDTH;
    const float *a0b = a0w + WIDTH * (2 * WIDTH);
    const float *a2w = a0b + WIDTH;
    const float *a2b = a2w + WIDTH;
    float c0[WIDTH], context[WIDTH], context_term[WIDTH];
    float encoded[WIDTH], hidden[WIDTH];
    layer(state, c0w, c0b, STATE, c0);
    layer(c0, c2w, c2b, WIDTH, context);
    for (size_t i = 0; i < WIDTH; ++i)
        context_term[i] = dot(context, a0w + i * 2 * WIDTH + WIDTH, WIDTH);
    for (size_t j = 0; j < count; ++j) {
        layer(features + j * FEATURES, ew, eb, FEATURES, encoded);
        for (size_t i = 0; i < WIDTH; ++i)
            hidden[i] = tanhf(dot(encoded, a0w + i * 2 * WIDTH, WIDTH) +
                             context_term[i] + a0b[i]);
        scores[j] = dot(hidden, a2w, WIDTH) + a2b[0];
    }
}
