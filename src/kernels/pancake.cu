// Composites `top` over `bottom` (straight alpha). All planes are row-major;
// Y and alpha are full-res, U and V have one sample per pair of luma pixels.
__device__ float over(float c_top, float c_bot, float a_top, float a_bot, float out_a)
{
    // out_a is kept away from 0 so fully transparent pixels don't divide by zero
    return (c_top * a_top + c_bot * a_bot * (1.0f - a_top)) / fmaxf(out_a, 1e-6f);
}

__device__ unsigned char to_byte(float v)
{
    return (unsigned char)(fminf(fmaxf(v, 0.0f), 255.0f) + 0.5f);
}

extern "C" __global__ void pancake(
    const unsigned char* __restrict__ y_bot,
    const unsigned char* __restrict__ u_bot,
    const unsigned char* __restrict__ v_bot,
    const unsigned char* __restrict__ a_bot,
    const unsigned char* __restrict__ y_top,
    const unsigned char* __restrict__ u_top,
    const unsigned char* __restrict__ v_top,
    const unsigned char* __restrict__ a_top,
    unsigned char* __restrict__ out_y,
    unsigned char* __restrict__ out_u,
    unsigned char* __restrict__ out_v,
    unsigned char* __restrict__ out_a,
    int num_pairs       // width * height / 2
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    int p1 = 2 * idx;
    int p2 = p1 + 1;

    float at1 = a_top[p1] / 255.0f;
    float at2 = a_top[p2] / 255.0f;
    float ab1 = a_bot[p1] / 255.0f;
    float ab2 = a_bot[p2] / 255.0f;

    float oa1 = at1 + ab1 * (1.0f - at1);
    float oa2 = at2 + ab2 * (1.0f - at2);

    float y1 = over((float)y_top[p1], (float)y_bot[p1], at1, ab1, oa1);
    float y2 = over((float)y_top[p2], (float)y_bot[p2], at2, ab2, oa2);

    // the shared chroma sample uses the average alpha of the pair
    float at = (at1 + at2) * 0.5f;
    float ab = (ab1 + ab2) * 0.5f;
    float oa = at + ab * (1.0f - at);

    float u = over((float)u_top[idx], (float)u_bot[idx], at, ab, oa);
    float v = over((float)v_top[idx], (float)v_bot[idx], at, ab, oa);

    out_y[p1] = to_byte(y1);
    out_y[p2] = to_byte(y2);
    out_u[idx] = to_byte(u);
    out_v[idx] = to_byte(v);
    out_a[p1] = to_byte(oa1 * 255.0f);
    out_a[p2] = to_byte(oa2 * 255.0f);
}
