// Purpose: Gated RMS norm of the GDN output heads.
// Input: y and z (N, V) BF16, the weight w (V) BF16, and eps (1) FP32.
// Output: o (N, V) BF16 with o = rms_norm(y) * w * gate(z).
// Method: One simdgroup owns one row. The gate is SiLU, or the sigmoid when
//     SIGMOID is true. All values are FP32 until the one rounding of o.
    const uint row = thread_position_in_grid.x / 32;
    const ushort lane = thread_index_in_simdgroup;
    if (row >= uint(y_shape[0])) return;
    float values[V / 32];
    float sum = 0.0f;
    for (short k = 0; k < V / 32; ++k) {
        values[k] = float(y[size_t(row) * V + lane + 32 * k]);
        sum += values[k] * values[k];
    }
    const float inverse = metal::rsqrt(simd_sum(sum) / float(V) + eps[0]);
    for (short k = 0; k < V / 32; ++k) {
        const int column = lane + 32 * k;
        const float gate_input = float(z[size_t(row) * V + column]);
        const float sigmoid = 1.0f / (1.0f + metal::exp(-gate_input));
        const float gate = SIGMOID ? sigmoid : gate_input * sigmoid;
        o[size_t(row) * V + column] = bfloat(values[k] * inverse * float(w[column]) * gate);
    }
