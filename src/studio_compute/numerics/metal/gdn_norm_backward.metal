// Purpose: Input gradients of the gated RMS norm of the GDN output heads.
// Input: y and z (N, V) BF16, w (V) BF16, eps (1) FP32, and do (N, V).
// Output: dy and dz (N, V) BF16.
// Method: With n = y r, a = n w, and o = a g(z):
//     dz = do a g'(z), dn = do g w, and dy = r (dn - n mean(dn n)).
//     g' is s (1 - s) for the sigmoid and s (1 + z (1 - s)) for SiLU.
    const uint row = thread_position_in_grid.x / 32;
    const ushort lane = thread_index_in_simdgroup;
    if (row >= uint(y_shape[0])) return;
    float normed[V / 32], dnormed[V / 32];
    float sum = 0.0f;
    for (short k = 0; k < V / 32; ++k) {
        normed[k] = float(y[size_t(row) * V + lane + 32 * k]);
        sum += normed[k] * normed[k];
    }
    const float inverse = metal::rsqrt(simd_sum(sum) / float(V) + eps[0]);
    float projection = 0.0f;
    for (short k = 0; k < V / 32; ++k) {
        const int column = lane + 32 * k;
        const size_t offset = size_t(row) * V + column;
        const float gate_input = float(z[offset]);
        const float sigmoid = 1.0f / (1.0f + metal::exp(-gate_input));
        const float gate = SIGMOID ? sigmoid : gate_input * sigmoid;
        const float slope = SIGMOID ? sigmoid * (1.0f - sigmoid)
                                    : sigmoid * (1.0f + gate_input * (1.0f - sigmoid));
        const float upstream = float(dout[offset]);
        normed[k] *= inverse;
        dz[offset] = bfloat(upstream * normed[k] * float(w[column]) * slope);
        dnormed[k] = upstream * gate * float(w[column]);
        projection += dnormed[k] * normed[k];
    }
    const float mean = simd_sum(projection) / float(V);
    for (short k = 0; k < V / 32; ++k) {
        const size_t offset = size_t(row) * V + lane + 32 * k;
        dy[offset] = bfloat(inverse * (dnormed[k] - normed[k] * mean));
    }
