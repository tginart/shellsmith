// Purpose: Intra-chunk transforms of the gated delta rule, for one chunk.
// Input: q and k (B, T, Hk, 128), v (B, T, Hv, 128), the gate and beta
//     logits a and b (B, T, Hv), all BF16, and the per-head A_log and dt_bias.
// Output: In (B, Hv, N, 64, 128) BF16 chunk layout: mu0, q_eff, u0, kt, and
//     a copy of k; decay_last (B, Hv, N) and ratio_last (B, Hv, N, 64) BF16.
// Method: With log gates lg (FP32 prefix sums), R_ij = exp(lg_i - lg_j),
//     beta = sigmoid(b), and X = inverse(I + beta R (k k^T), strict lower):
//     1. u0 = X (beta v) and kt = X (beta exp(lg) k).
//     2. m = R (q k^T), lower with the diagonal.
//     3. mu0 = m u0 and q_eff = exp(lg) q - m kt.
//     The scores use row bands. Steps 1 and 3 use column slices: simdgroup sg
//     owns columns 32 sg .. 32 sg + 31, so u0 and kt stay in its registers.
// Note: The gates come from the logits in FP32, so a gate near 1 keeps its
//     exact log. Matrix operands are BF16 with FP32 sums.
    const int n = threadgroup_position_in_grid.x;
    const int h = threadgroup_position_in_grid.y;
    const int bb = threadgroup_position_in_grid.z;
    const ushort tid = thread_index_in_threadgroup;
    const short sg = simdgroup_index_in_threadgroup;
    const short2 sc = BaseNAXFrag::get_coord();
    const int length = q_shape[1];
    const int hk = q_shape[2];
    const int hv = v_shape[2];
    const int chunks = length / CHUNK;
    const int ks = hk * WIDTH;
    const int vs = hv * WIDTH;
    const size_t row0 = size_t(bb) * length + n * CHUNK;
    const device bfloat* qb = q + (row0 * hk + h / GROUP) * WIDTH;
    const device bfloat* kb = k + (row0 * hk + h / GROUP) * WIDTH;
    const device bfloat* vb = v + (row0 * hv + h) * WIDTH;
    const size_t block = (size_t(bb) * hv + h) * chunks + n;
    const size_t out0 = block * CHUNK * WIDTH;
    const int band = 16 * sg;
    const int slice = 32 * sg;

    threadgroup float lower[CHUNK * CHUNK];
    threadgroup bfloat mixer[CHUNK * CHUNK];
    threadgroup float lg[CHUNK], beta[CHUNK], decay[CHUNK];

    load_gates(a + row0 * hv + h, b + row0 * hv + h, hv, float(a_log[h]),
               float(dt_bias[h]), lg, beta, decay, tid);
    // The staged keys share memory with the lower matrix, which comes later.
    threadgroup bfloat* keys = reinterpret_cast<threadgroup bfloat*>(lower);
    stage_keys(keys, kb, ks, k_chunks + out0, tid);
    {
        ffrag kk[4], qk[4];
        chunk_scores(kk, qk, qb, ks, keys, sg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (short cb = 0; cb < 4; ++cb)
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < 8; ++e) {
                const int i = band + frag_row(sc, e);
                const int j = 16 * cb + frag_col(sc, e);
                const float r = j <= i ? metal::exp(lg[i] - lg[j]) : 0.0f;
                lower[i * CHUNK + j] = j < i ? beta[i] * r * kk[cb][e] : 0.0f;
                mixer[i * CHUNK + j] = bfloat(r * qk[cb][e]);
            }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    invert_unit_lower(lower, tid);

    // pass 0: u0 = X (beta v), then mu0 = m u0.
    // pass 1: kt = X (beta e^lg k), then q_eff = e^lg q - m kt.
    STEEL_PRAGMA_UNROLL
    for (short pass = 0; pass < 2; ++pass) {
        const device bfloat* source = pass == 0 ? vb : kb;
        const int stride = pass == 0 ? vs : ks;
        ffrag acc[4][2];
        STEEL_PRAGMA_UNROLL
        for (short r = 0; r < 4; ++r) acc[r][0] = acc[r][1] = zero_ffrag();
        STEEL_PRAGMA_UNROLL
        for (short js = 0; js < 4; ++js) {
            ffrag r0, r1;
            BaseNAXFrag::load(r0, source + 16 * js * stride + slice, stride, Int<1>{});
            BaseNAXFrag::load(r1, source + 16 * js * stride + slice + 16, stride, Int<1>{});
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < 8; ++e) {
                const int j = 16 * js + frag_row(sc, e);
                const float scale = pass == 0 ? beta[j] : beta[j] * decay[j];
                r0[e] *= scale;
                r1[e] *= scale;
            }
            STEEL_PRAGMA_UNROLL
            for (short r = 0; r < 4; ++r) {
                if (js > r) continue;
                ffrag xa;
                BaseNAXFrag::load(xa, lower + 16 * r * CHUNK + 16 * js, CHUNK, Int<1>{});
                BaseNAXFrag::mma(acc[r][0], acc[r][1], xa, metal::false_type{}, r0, r1,
                                 metal::false_type{});
            }
        }
        device bfloat* solved = (pass == 0 ? u0 : kt) + out0 + slice;
        STEEL_PRAGMA_UNROLL
        for (short r = 0; r < 4; ++r) {
            BaseNAXFrag::store(acc[r][0], solved + 16 * r * WIDTH, WIDTH, Int<1>{});
            BaseNAXFrag::store(acc[r][1], solved + 16 * r * WIDTH + 16, WIDTH, Int<1>{});
        }
        device bfloat* mixed = (pass == 0 ? mu0 : q_eff) + out0 + slice;
        STEEL_PRAGMA_UNROLL
        for (short r = 0; r < 4; ++r) {
            ffrag out0f = zero_ffrag(), out1f = zero_ffrag();
            STEEL_PRAGMA_UNROLL
            for (short js = 0; js <= r; ++js) {
                bfrag ma;
                BaseNAXFrag::load(ma, mixer + 16 * r * CHUNK + 16 * js, CHUNK, Int<1>{});
                BaseNAXFrag::mma(out0f, out1f, ma, metal::false_type{}, acc[js][0],
                                 acc[js][1], metal::false_type{});
            }
            if (pass == 1) {
                ffrag q0, q1;
                BaseNAXFrag::load(q0, qb + 16 * r * ks + slice, ks, Int<1>{});
                BaseNAXFrag::load(q1, qb + 16 * r * ks + slice + 16, ks, Int<1>{});
                STEEL_PRAGMA_UNROLL
                for (short e = 0; e < 8; ++e) {
                    const float scale = decay[16 * r + frag_row(sc, e)];
                    out0f[e] = scale * q0[e] - out0f[e];
                    out1f[e] = scale * q1[e] - out1f[e];
                }
            }
            BaseNAXFrag::store(out0f, mixed + 16 * r * WIDTH, WIDTH, Int<1>{});
            BaseNAXFrag::store(out1f, mixed + 16 * r * WIDTH + 16, WIDTH, Int<1>{});
        }
    }

    if (tid < CHUNK) ratio_last[block * CHUNK + tid] = bfloat(metal::exp(lg[CHUNK - 1] - lg[tid]));
    if (tid == 0) decay_last[block] = bfloat(decay[CHUNK - 1]);
