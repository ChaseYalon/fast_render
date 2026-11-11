extern "C" __global__ void saturation_change(
    const unsigned char* __restrict__ in,
    float factor,
    int n,
    unsigned char* __restrict__ out
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float v = 128.0f + (in[idx] - 128.0f) * factor;
    out[idx] = (unsigned char)(fminf(fmaxf(v, 0.0f), 255.0f) + 0.5f);
}
