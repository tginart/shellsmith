// Purpose: Causal depthwise convolution and SiLU, for the GDN input projection
//     (DIL 1) and the PLE short convolution (DIL 3).
// Input: the carried taps s (B, (K - 1) DIL, C), the rows x (B, L, C), and the
//     depthwise weight w (C, K), all BF16.
// Output: y (B, L, C) BF16.
// Method: One thread owns four channels of one (b, t). Row t reads the K rows
//     t + j DIL of the sequence [s; x], four channels per vector load. The
//     sum and the SiLU are in FP32, and only the result is rounded to BF16.
    const uint index = thread_position_in_grid.x;
    const int L = x_shape[1];
    const int c = 4 * (index % (C / 4));
    const int t = (index / (C / 4)) % L;
    const int b = index / (C / 4 * L);
    float4 pre = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (short j = 0; j < K; ++j) {
        const int source = t + (j - (K - 1)) * DIL;
        const vec<bfloat, 4> value = source >= 0
            ? *reinterpret_cast<const device vec<bfloat, 4>*>(x + (b * L + source) * C + c)
            : *reinterpret_cast<const device vec<bfloat, 4>*>(
                  s + (b * (K - 1) * DIL + (K - 1) * DIL + source) * C + c);
        const float4 weight = float4(float(w[c * K + j]), float(w[(c + 1) * K + j]),
                                     float(w[(c + 2) * K + j]), float(w[(c + 3) * K + j]));
        pre += float4(value) * weight;
    }
    const float4 out = pre / (1.0f + metal::exp(-pre));
    *reinterpret_cast<device vec<bfloat, 4>*>(y + (b * L + t) * C + c) = vec<bfloat, 4>(out);
