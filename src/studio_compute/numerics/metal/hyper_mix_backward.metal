// Purpose: Input gradients of the hyper-connection stream mix.
// Input: The projected gate logits u and the normed streams n, both (N, G * D)
//     BF16, and the mixed-row gradient dm (N, D).
// Output: du and dn (N, G * D) BF16.
// Method: The mix is m = mean_g(sigmoid(u_g) * n_g). With s = sigmoid(u_g),
//     du_g = dm * n_g * s * (1 - s) / G and dn_g = dm * s / G, in FP32.
    const uint index = thread_position_in_grid.x;
    const uint row = index / D;
    const uint column = index % D;
    const float share = float(dm[index]) / float(G);
    for (short g = 0; g < G; ++g) {
        const size_t offset = size_t(row) * G * D + g * D + column;
        const float gate = 1.0f / (1.0f + metal::exp(-float(u[offset])));
        du[offset] = bfloat(share * float(n[offset]) * gate * (1.0f - gate));
        dn[offset] = bfloat(share * gate);
    }
