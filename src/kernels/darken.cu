extern "C" __global__ void darken(
    const unsigned char* y,
    float scale_factor,      // 1.0 = unchanged, 0.0 = black
    int bytes_in_plane,
    unsigned char* output_frame)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bytes_in_plane) return; //out of bounds
    float v = ((float)y[i] - 16.0f) * scale_factor + 16.0f;
    output_frame[i] = (unsigned char)(v + 0.5f); // +0.5 rounds instead of truncating
}
