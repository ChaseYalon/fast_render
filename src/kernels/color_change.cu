extern "C" __global__ void color_change(
    const unsigned char* y,
    float amount,
    int bytes_in_plane,
    unsigned char* output_frame,
    int is_chroma)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= bytes_in_plane) return; //out of bounds

    float v = (float)y[i];
    if (is_chroma) {
        float t = fabsf(amount - 0.5f) * 2.0f;
        v = 128.0f + (v - 128.0f) * (1.0f - t);
    } else if (amount <= 0.5f) {
        v = (v - 16.0f) * (amount * 2.0f) + 16.0f;
    } else {
        // toward white: close the gap to the white level (235)
        v = v + (235.0f - v) * ((amount - 0.5f) * 2.0f);
    }

    v = fminf(fmaxf(v, 0.0f), 255.0f); // keep it inside a byte
    output_frame[i] = (unsigned char)(v + 0.5f); // +0.5 rounds instead of truncating
}
