// Purpose: Input gradient of the grouped RMS norm of the hyper-connection.
// Input: x (N, G * D) BF16, the weight w (G * D), the inverse RMS r (N, G), and
//     the output gradient dy (N, G * D).
// Output: dx (N, G * D) BF16.
// Method:
//     1. One threadgroup owns one (row, group) pair. With g = 1 + w, it sums
//        s = sum(g * dy * x) in FP32.
//     2. Each value is dx = r * g * dy - x * r^3 * s / D.
    const uint pair = threadgroup_position_in_grid.x;
    const ushort tid = thread_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const ushort sg = simdgroup_index_in_threadgroup;
    const uint group = pair % G;
    const device bfloat* xp = x + size_t(pair) * D;
    const device auto* dyp = dy + size_t(pair) * D;
    const device auto* wp = w + group * D;
    threadgroup float partial[TG / 32];

    float xs[D / TG], gdy[D / TG];
    float sum = 0.0f;
    for (short k = 0; k < D / TG; ++k) {
        const int column = tid + TG * k;
        xs[k] = float(xp[column]);
        gdy[k] = (1.0f + float(wp[column])) * float(dyp[column]);
        sum += gdy[k] * xs[k];
    }
    sum = simd_sum(sum);
    if (lane == 0) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (short k = 0; k < TG / 32; ++k) total += partial[k];
    const float inverse = r[pair];
    const float correction = inverse * inverse * inverse * total / float(D);
    device bfloat* dxp = dx + size_t(pair) * D;
    for (short k = 0; k < D / TG; ++k)
        dxp[tid + TG * k] = bfloat(inverse * gdy[k] - xs[k] * correction);
