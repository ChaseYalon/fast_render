extern "C" __global__ void blur(
    const unsigned char* __restrict__ in,
    unsigned char* __restrict__ out,
    const float* __restrict__ weights,
    int width,
    int height,
    int radius,
    int horizontal
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    int x = idx % width;
    int y = idx / width;

    float sum = 0.0f;
    for (int i = -radius; i <= radius; i++) {
        int sx = horizontal ? x + i : x;
        int sy = horizontal ? y : y + i;
        sx = min(max(sx, 0), width - 1);
        sy = min(max(sy, 0), height - 1);
        sum += in[sy * width + sx] * weights[i + radius];
    }

    out[idx] = (unsigned char)(fminf(fmaxf(sum, 0.0f), 255.0f) + 0.5f);
}
