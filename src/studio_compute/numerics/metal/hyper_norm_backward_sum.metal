// Purpose: Input gradient of the hyper-connection grouped RMS norm, with the
//     sum of all normed-row gradients in one pass.
// Input: x (N, G * D) BF16, the weight w (G * D), the inverse RMS r (N, G), the
//     normed-row gradients of the stream mix dm and the down projection dd
//     (N, G * D), the injection logit gradient di (N, G), and the dequantized
//     injection weight wi (G, G * D).
// Output: dx (N, G * D) BF16.
// Method:
//     1. One threadgroup owns one (row, group) pair. Each value of the normed-row
//        gradient is dy = dm + dd + sum_h di[h] * wi[h], in FP32 registers.
//     2. With g = 1 + w, it sums s = sum(g * dy * x) in FP32.
//     3. Each value is dx = r * g * dy - x * r^3 * s / D.
// Note: Separate passes would write and read two (N, G * D) sums and the
//     injection term.
    const uint pair = threadgroup_position_in_grid.x;
    const uint row = pair / G;
    const ushort tid = thread_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const ushort sg = simdgroup_index_in_threadgroup;
    const uint group = pair % G;
    const size_t base = size_t(pair) * D;
    const device auto* wp = w + group * D;
    threadgroup float partial[TG / 32];

    float injection[G];
    for (short h = 0; h < G; ++h) injection[h] = float(di[row * G + h]);
    float xs[D / TG], gdy[D / TG];
    float sum = 0.0f;
    for (short k = 0; k < D / TG; ++k) {
        const int column = tid + TG * k;
        float dy = float(dm[base + column]) + float(dd[base + column]);
        for (short h = 0; h < G; ++h)
            dy += injection[h] * float(wi[size_t(h) * G * D + group * D + column]);
        xs[k] = float(x[base + column]);
        gdy[k] = (1.0f + float(wp[column])) * dy;
        sum += gdy[k] * xs[k];
    }
    sum = simd_sum(sum);
    if (lane == 0) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (short k = 0; k < TG / 32; ++k) total += partial[k];
    const float inverse = r[pair];
    const float correction = inverse * inverse * inverse * total / float(D);
    for (short k = 0; k < D / TG; ++k)
        dx[base + tid + TG * k] = bfloat(inverse * gdy[k] - xs[k] * correction);
