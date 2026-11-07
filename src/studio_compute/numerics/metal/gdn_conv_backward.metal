// Purpose: Input gradients of the causal depthwise convolution and SiLU.
// Input: s (B, S, C) with S = (K - 1) DIL, x (B, L, C), w (C, K), and the
//     output gradient dy (B, L, C).
// Output: ds (B, S, C) and dx (B, L, C) BF16.
// Method: One thread owns one row p of the sequence [s; x] and four channels.
//     The thread recomputes the K pre-activations that read row p, applies
//     the SiLU derivative s (1 + a (1 - s)), and sums dpre[t] w[p - t].
    const uint index = thread_position_in_grid.x;
    const int L = x_shape[1];
    constexpr int S = (K - 1) * DIL;
    const int P = L + S;
    const int c = 4 * (index % (C / 4));
    const int p = (index / (C / 4)) % P;
    const int b = index / (C / 4 * P);
    float4 weight[K];
    _Pragma("clang loop unroll(full)")
    for (short j = 0; j < K; ++j)
        weight[j] = float4(float(w[c * K + j]), float(w[(c + 1) * K + j]),
                           float(w[(c + 2) * K + j]), float(w[(c + 3) * K + j]));
    float4 gradient = 0.0f;
    _Pragma("clang loop unroll(full)")
    for (short j = 0; j < K; ++j) {
        // Output row t reads row p through tap j when p = t + j DIL.
        const int t = p - j * DIL;
        if (t < 0 || t >= L) continue;
        float4 pre = 0.0f;
        _Pragma("clang loop unroll(full)")
        for (short i = 0; i < K; ++i) {
            const int source = t + (i - (K - 1)) * DIL;
            const vec<bfloat, 4> value = source >= 0
                ? *reinterpret_cast<const device vec<bfloat, 4>*>(x + (b * L + source) * C + c)
                : *reinterpret_cast<const device vec<bfloat, 4>*>(
                      s + (b * S + S + source) * C + c);
            pre += float4(value) * weight[i];
        }
        const float4 gate = 1.0f / (1.0f + metal::exp(-pre));
        const float4 slope = gate * (1.0f + pre * (1.0f - gate));
        const float4 upstream = float4(
            *reinterpret_cast<const device vec<bfloat, 4>*>(dy + (b * L + t) * C + c));
        gradient += upstream * slope * weight[j];
    }
    if (p < S) {
        *reinterpret_cast<device vec<bfloat, 4>*>(ds + (b * S + p) * C + c) =
            vec<bfloat, 4>(gradient);
    } else {
        *reinterpret_cast<device vec<bfloat, 4>*>(dx + (b * L + p - S) * C + c) =
            vec<bfloat, 4>(gradient);
    }
