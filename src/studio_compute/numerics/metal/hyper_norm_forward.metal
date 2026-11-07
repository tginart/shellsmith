// Purpose: Grouped RMS norm of the Flash hyper-connection input.
// Input: x (N, G * D) BF16 and the zero-centered weight w (G * D).
// Output: y (N, G * D) BF16 and the FP32 inverse RMS r (N, G).
// Method:
//     1. One threadgroup owns one (row, group) pair of D values.
//     2. The threadgroup sums x^2 in FP32 and calculates r = rsqrt(mean + eps).
//     3. Each value is y = x * r * (1 + w) in FP32, then rounded to BF16.
    const uint pair = threadgroup_position_in_grid.x;
    const ushort tid = thread_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const ushort sg = simdgroup_index_in_threadgroup;
    const uint group = pair % G;
    const device bfloat* xp = x + size_t(pair) * D;
    const device auto* wp = w + group * D;
    threadgroup float partial[TG / 32];

    float local[D / TG];
    float sum = 0.0f;
    for (short k = 0; k < D / TG; ++k) {
        local[k] = float(xp[tid + TG * k]);
        sum += local[k] * local[k];
    }
    sum = simd_sum(sum);
    if (lane == 0) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (short k = 0; k < TG / 32; ++k) total += partial[k];
    const float inverse = metal::rsqrt(total / float(D) + eps[0]);
    if (tid == 0) r[pair] = inverse;
    device bfloat* yp = y + size_t(pair) * D;
    for (short k = 0; k < D / TG; ++k) {
        const int column = tid + TG * k;
        yp[column] = bfloat(local[k] * inverse * (1.0f + float(wp[column])));
    }
