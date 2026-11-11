__device__ void inverse_map(
    float x, float y,
    float trans_x, float trans_y,
    float rot_x, float rot_y, float cos_t, float sin_t,
    float scale_x, float scale_y, float scale_factor,
    float* src_x, float* src_y)
{
    // 1. undo scale: shrink the distance from the scale pivot back down
    x = scale_x + (x - scale_x) / scale_factor;
    y = scale_y + (y - scale_y) / scale_factor;

    // 2. undo rotation: rotate by -theta around the rotation pivot
    float dx = x - rot_x;
    float dy = y - rot_y;
    x = rot_x + cos_t * dx + sin_t * dy;
    y = rot_y - sin_t * dx + cos_t * dy;

    // 3. undo translation
    *src_x = x - trans_x;
    *src_y = y - trans_y;
}

// Bilinear sample of a w x h byte plane; positions outside the plane give `fill`.
__device__ float sample(const unsigned char* plane, int w, int h, float x, float y, float fill)
{
    if (x < 0.0f || y < 0.0f || x > (float)(w - 1) || y > (float)(h - 1)) return fill;
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = min(x0 + 1, w - 1);
    int y1 = min(y0 + 1, h - 1);
    float fx = x - (float)x0;
    float fy = y - (float)y0;
    float top = plane[y0 * w + x0] * (1.0f - fx) + plane[y0 * w + x1] * fx;
    float bottom = plane[y1 * w + x0] * (1.0f - fx) + plane[y1 * w + x1] * fx;
    return top * (1.0f - fy) + bottom * fy;
}

// Alpha of the source at (x, y) for a frame with no alpha plane: opaque inside, transparent outside.
__device__ float opaque_inside(int w, int h, float x, float y)
{
    if (x < 0.0f || y < 0.0f || x > (float)(w - 1) || y > (float)(h - 1)) return 0.0f;
    return 255.0f;
}

extern "C" __global__ void transform_macropixel(
    const unsigned char* __restrict__ in_y,
    const unsigned char* __restrict__ in_u,
    const unsigned char* __restrict__ in_v,
    const unsigned char* __restrict__ in_a,   // only read when has_alpha is 1
    unsigned char* __restrict__ out_y,
    unsigned char* __restrict__ out_u,
    unsigned char* __restrict__ out_v,
    unsigned char* __restrict__ out_a,
    int width,            // luma width; U and V planes are width / 2 wide, alpha is width
    int height,           // same for all planes
    int has_alpha,
    float trans_x,        // translation, in luma pixels
    float trans_y,
    float rot_x,          // rotation is around (rot_x, rot_y), in luma pixels
    float rot_y,
    float degrees_of_rot,
    float scale_x,        // scale is from (scale_x, scale_y), in luma pixels
    float scale_y,
    float scale_factor
)
{
    int chroma_x = blockIdx.x * blockDim.x + threadIdx.x;
    int chroma_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (chroma_x >= width / 2 || chroma_y >= height) return; //out of bounds

    // 4:2:2: chroma is half width but full height, so a macropixel is two
    // luma pixels in the same row
    int luma_x1 = 2 * chroma_x;
    int luma_x2 = luma_x1 + 1;
    int luma_y = chroma_y;

    float radians = degrees_of_rot * 0.017453292519943295f; // pi / 180
    float cos_t = cosf(radians);
    float sin_t = sinf(radians);

    // where each of this macropixel's samples came from in the source frame
    // (fractional luma coordinates; they may be outside the frame)
    float original_x1, original_y1;
    float original_x2, original_y2;
    float original_cx, original_cy;
    inverse_map((float)luma_x1, (float)luma_y,
                trans_x, trans_y, rot_x, rot_y, cos_t, sin_t,
                scale_x, scale_y, scale_factor, &original_x1, &original_y1);
    inverse_map((float)luma_x2, (float)luma_y,
                trans_x, trans_y, rot_x, rot_y, cos_t, sin_t,
                scale_x, scale_y, scale_factor, &original_x2, &original_y2);
    // the shared chroma sample sits between the two luma pixels
    inverse_map((float)luma_x1 + 0.5f, (float)luma_y,
                trans_x, trans_y, rot_x, rot_y, cos_t, sin_t,
                scale_x, scale_y, scale_factor, &original_cx, &original_cy);

    int chroma_w = width / 2;
    // chroma sample i sits at luma x = 2i + 0.5
    float chroma_src_x = (original_cx - 0.5f) / 2.0f;

    float y1 = sample(in_y, width, height, original_x1, original_y1, 16.0f);
    float y2 = sample(in_y, width, height, original_x2, original_y2, 16.0f);
    float u = sample(in_u, chroma_w, height, chroma_src_x, original_cy, 128.0f);
    float v = sample(in_v, chroma_w, height, chroma_src_x, original_cy, 128.0f);

    float a1 = has_alpha ? sample(in_a, width, height, original_x1, original_y1, 0.0f)
                         : opaque_inside(width, height, original_x1, original_y1);
    float a2 = has_alpha ? sample(in_a, width, height, original_x2, original_y2, 0.0f)
                         : opaque_inside(width, height, original_x2, original_y2);

    out_y[luma_y * width + luma_x1] = (unsigned char)(y1 + 0.5f);
    out_y[luma_y * width + luma_x2] = (unsigned char)(y2 + 0.5f);
    out_a[luma_y * width + luma_x1] = (unsigned char)(a1 + 0.5f);
    out_a[luma_y * width + luma_x2] = (unsigned char)(a2 + 0.5f);
    out_u[chroma_y * chroma_w + chroma_x] = (unsigned char)(u + 0.5f);
    out_v[chroma_y * chroma_w + chroma_x] = (unsigned char)(v + 0.5f);
}
