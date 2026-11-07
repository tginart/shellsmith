// Purpose: Gradients of y = h + w o of the hyper residual injection.
// Input: branch output o (N, D), injection weights w (N, G), and the output
//     gradient dy (N, G * D), all BF16.
// Output: do (N, D) BF16 and dw (N, G) FP32.
// Method: One threadgroup owns one row n. Each thread owns D / TG / 4 groups of
//     four columns. do = sum over g of w dy, and dw[g] = dy[g] . o, with the
//     row sums over simdgroups in threadgroup memory.
    const int n = threadgroup_position_in_grid.x;
    const ushort tid = thread_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const ushort sg = simdgroup_index_in_threadgroup;
    threadgroup float partial[TG / 32][G];
    float weight[G], dot[G];
    for (short g = 0; g < G; ++g) {
        weight[g] = float(w[n * G + g]);
        dot[g] = 0.0f;
    }
    for (int d = 4 * tid; d < D; d += 4 * TG) {
        const float4 branch = float4(
            *reinterpret_cast<const device vec<bfloat, 4>*>(o + size_t(n) * D + d));
        float4 total = 0.0f;
        for (short g = 0; g < G; ++g) {
            const float4 grad = float4(*reinterpret_cast<const device vec<bfloat, 4>*>(
                dy + (size_t(n) * G + g) * D + d));
            total += weight[g] * grad;
            dot[g] += metal::dot(grad, branch);
        }
        *reinterpret_cast<device vec<bfloat, 4>*>(d_o + size_t(n) * D + d) =
            vec<bfloat, 4>(total);
    }
    for (short g = 0; g < G; ++g) {
        const float value = simd_sum(dot[g]);
        if (lane == 0) partial[sg][g] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < G) {
        float value = 0.0f;
        for (short s = 0; s < TG / 32; ++s) value += partial[s][tid];
        d_w[n * G + tid] = value;
    }
