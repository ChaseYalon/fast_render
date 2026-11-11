#define FEATHER 32.0f

// How much of clip A, clip B and black a pixel gets (they sum to 1).
__device__ void weights(int kind, int reverse, float p, int x, int width,
                        float* wa, float* wb, float* wk)
{
    *wk = 0.0f;
    if (kind == 0) {
        *wb = p;
    } else if (kind == 1) {
        float px = reverse ? (float)(width - 1 - x) : (float)x;
        float edge = p * ((float)width + FEATHER);
        *wb = fminf(fmaxf((edge - px) / FEATHER, 0.0f), 1.0f);
    } else {
        *wb = fmaxf(2.0f * p - 1.0f, 0.0f);
        *wa = fmaxf(1.0f - 2.0f * p, 0.0f);
        *wk = 1.0f - *wa - *wb;
        return;
    }
    *wa = 1.0f - *wb;
}

__device__ unsigned char to_byte(float v)
{
    return (unsigned char)(fminf(fmaxf(v, 0.0f), 255.0f) + 0.5f);
}

extern "C" __global__ void transition(
    const unsigned char* __restrict__ y_a,
    const unsigned char* __restrict__ u_a,
    const unsigned char* __restrict__ v_a,
    const unsigned char* __restrict__ a_a,
    const unsigned char* __restrict__ y_b,
    const unsigned char* __restrict__ u_b,
    const unsigned char* __restrict__ v_b,
    const unsigned char* __restrict__ a_b,
    unsigned char* __restrict__ out_y,
    unsigned char* __restrict__ out_u,
    unsigned char* __restrict__ out_v,
    unsigned char* __restrict__ out_a,
    int num_pairs,      // width * height / 2
    int width,
    int kind,           // 0 crossfade, 1 wipe, 2 dip to black
    int reverse,        // wipe runs right to left
    float progress      // 0..1
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    int p1 = 2 * idx;
    int p2 = p1 + 1;
    int x1 = p1 % width;

    float wa1, wb1, wk1, wa2, wb2, wk2;
    weights(kind, reverse, progress, x1, width, &wa1, &wb1, &wk1);
    weights(kind, reverse, progress, x1 + 1, width, &wa2, &wb2, &wk2);

    // the shared chroma sample uses the average weights of the pair
    float wa = (wa1 + wa2) * 0.5f;
    float wb = (wb1 + wb2) * 0.5f;
    float wk = (wk1 + wk2) * 0.5f;

    out_y[p1] = to_byte(wa1 * y_a[p1] + wb1 * y_b[p1] + wk1 * 16.0f);
    out_y[p2] = to_byte(wa2 * y_a[p2] + wb2 * y_b[p2] + wk2 * 16.0f);
    out_u[idx] = to_byte(wa * u_a[idx] + wb * u_b[idx] + wk * 128.0f);
    out_v[idx] = to_byte(wa * v_a[idx] + wb * v_b[idx] + wk * 128.0f);
    out_a[p1] = to_byte(wa1 * a_a[p1] + wb1 * a_b[p1] + wk1 * 255.0f);
    out_a[p2] = to_byte(wa2 * a_a[p2] + wb2 * a_b[p2] + wk2 * 255.0f);
}
