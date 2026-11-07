// Purpose: The hyper-connection stream mix in one pass.
// Input: The projected gate logits u and the normed streams n, both (N, G * D)
//     BF16.
// Output: m (N, D) BF16, m = mean_g(sigmoid(u_g) * n_g).
// Method: One thread owns four columns of one row. Each step rounds to BF16 as
//     the upstream compiled mix does: the MLX sigmoid formula, the product, and
//     a running BF16 sum over the streams. So the result is the upstream one.
    const uint index = thread_position_in_grid.x;
    const uint row = index / (D / 4);
    const uint column = 4 * (index % (D / 4));
    vec<bfloat, 4> total;
    for (short g = 0; g < G; ++g) {
        const size_t offset = size_t(row) * G * D + g * D + column;
        const vec<bfloat, 4> logit = *reinterpret_cast<const device vec<bfloat, 4>*>(u + offset);
        const vec<bfloat, 4> stream = *reinterpret_cast<const device vec<bfloat, 4>*>(n + offset);
        vec<bfloat, 4> product;
        for (short j = 0; j < 4; ++j) {
            const bfloat e = bfloat(metal::exp(float(metal::abs(logit[j]))));
            const bfloat y = bfloat(1.0f / float(bfloat(1.0f + float(e))));
            const bfloat gate = logit[j] < bfloat(0.0f) ? y : bfloat(1.0f - float(y));
            product[j] = bfloat(float(gate) * float(stream[j]));
        }
        total = g == 0 ? product : vec<bfloat, 4>(float4(total) + float4(product));
    }
    *reinterpret_cast<device vec<bfloat, 4>*>(m + size_t(row) * D + column) =
        vec<bfloat, 4>(float4(total) / float(G));
