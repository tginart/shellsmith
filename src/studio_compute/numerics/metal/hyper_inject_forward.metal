// Purpose: Add the injected branch output to the hyper residual.
// Input: hyper h (N, G * D), branch output o (N, D), and injection weights
//     w (N, G), all BF16.
// Output: y (N, G * D) BF16 with y[n, g, d] = h[n, g, d] + w[n, g] o[n, d].
// Method: One thread owns four values. The sum is FP32, rounded once.
    const uint index = thread_position_in_grid.x;
    const int d = 4 * (index % (D / 4));
    const int g = (index / (D / 4)) % G;
    const int n = index / (D / 4 * G);
    const size_t offset = (size_t(n) * G + g) * D + d;
    const float4 hyper = float4(*reinterpret_cast<const device vec<bfloat, 4>*>(h + offset));
    const float4 branch = float4(
        *reinterpret_cast<const device vec<bfloat, 4>*>(o + size_t(n) * D + d));
    *reinterpret_cast<device vec<bfloat, 4>*>(y + offset) =
        vec<bfloat, 4>(hyper + float(w[n * G + g]) * branch);
