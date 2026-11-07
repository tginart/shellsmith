// Purpose: Input gradients of the fused GDN chunk transforms, for one chunk.
// Input: The forward inputs, the forward outputs u0 and kt, and the output
//     gradients of mu0, q_eff, u0, kt, the k copy, decay_last, and ratio_last.
// Output: dq, dk, dv (B, T, Hv, 128) per value head, and the gate and beta
//     logit gradients da and db (B, T, Hv). s_du, s_dkt, and s_dkb are scratch.
// Method: The kernel recomputes the gates, R, A = beta R (k k^T) (strict
//     lower), m = R (q k^T), and X = inverse(I + A). Then, by bands of rows:
//     1. dm = dmu0 u0^T - dq_eff kt^T, lower with the diagonal.
//     2. du0' = du0 + m^T dmu0 and dkt' = dkt - m^T dq_eff (scratch).
//     3. dW = X^T du0' and dKb = X^T dkt', with W = beta v, Kb = beta e^lg k.
//        dA = -(dW u0^T + dKb kt^T), strict lower.
//     4. dkk = dA beta R and dS = dm R. P = (dA beta kk + dm qk) R gives the
//        R terms of the log gate gradient: + row sums, - column sums.
//     5. dq = e^lg dq_eff + dS k and dk = (dkk + dkk^T) k + dS^T q
//        + beta e^lg dKb + dk_copy.
//     6. The log gate gradient becomes a suffix sum, then da and db.
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
    const device bfloat* u0b = u0 + out0;
    const device bfloat* ktb = kt + out0;
    const device bfloat* dmu = d_mu0 + out0;
    const device bfloat* dqe = d_q_eff + out0;
    const int band = 16 * sg;

    threadgroup float lower[CHUNK * CHUNK];
    threadgroup bfloat mixer[CHUNK * CHUNK];
    threadgroup float lg[CHUNK], beta[CHUNK], decay[CHUNK];
    threadgroup float dlg[CHUNK], dbeta[CHUNK], ddecay[CHUNK], rowk[CHUNK];

    if (tid < CHUNK) {
        dlg[tid] = 0.0f;
        dbeta[tid] = 0.0f;
        ddecay[tid] = 0.0f;
        rowk[tid] = 0.0f;
    }
    load_gates(a + row0 * hv + h, b + row0 * hv + h, hv, float(a_log[h]),
               float(dt_bias[h]), lg, beta, decay, tid);
    threadgroup bfloat* keys = reinterpret_cast<threadgroup bfloat*>(lower);
    stage_keys(keys, kb, ks, nullptr, tid);
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

    // 1. dm = dmu0 u0^T - dq_eff kt^T.
    ffrag dm[4], dA[4];
    STEEL_PRAGMA_UNROLL
    for (short f = 0; f < 4; ++f) {
        dm[f] = zero_ffrag();
        dA[f] = zero_ffrag();
    }
    STEEL_PRAGMA_UNROLL
    for (short kslice = 0; kslice < FRAGS; ++kslice) {
        ffrag am, aq;
        BaseNAXFrag::load(am, dmu + band * WIDTH + 16 * kslice, WIDTH, Int<1>{});
        BaseNAXFrag::load(aq, dqe + band * WIDTH + 16 * kslice, WIDTH, Int<1>{});
        aq = -aq;
        STEEL_PRAGMA_UNROLL
        for (short cb = 0; cb < 4; cb += 2) {
            bfrag r0, r1, t0, t1;
            BaseNAXFrag::load(r0, u0b + 16 * cb * WIDTH + 16 * kslice, WIDTH, Int<1>{});
            BaseNAXFrag::load(r1, u0b + 16 * (cb + 1) * WIDTH + 16 * kslice, WIDTH, Int<1>{});
            BaseNAXFrag::load(t0, ktb + 16 * cb * WIDTH + 16 * kslice, WIDTH, Int<1>{});
            BaseNAXFrag::load(t1, ktb + 16 * (cb + 1) * WIDTH + 16 * kslice, WIDTH, Int<1>{});
            BaseNAXFrag::mma(dm[cb], dm[cb + 1], am, metal::false_type{}, r0, r1,
                             metal::true_type{});
            BaseNAXFrag::mma(dm[cb], dm[cb + 1], aq, metal::false_type{}, t0, t1,
                             metal::true_type{});
        }
    }
    STEEL_PRAGMA_UNROLL
    for (short cb = 0; cb < 4; ++cb)
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 8; ++e)
            if (16 * cb + frag_col(sc, e) > band + frag_row(sc, e)) dm[cb][e] = 0.0f;

    // 2. du0' = du0 + m^T dmu0 and dkt' = dkt - m^T dq_eff, through scratch.
    //    A transposed operand is a plain fragment load with transpose_a set.
    STEEL_PRAGMA_UNROLL
    for (short pass = 0; pass < 2; ++pass) {
        const device bfloat* seed = (pass == 0 ? d_u0 : d_kt) + out0;
        const device bfloat* source = pass == 0 ? dmu : dqe;
        ffrag acc[FRAGS];
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < FRAGS; ++d)
            BaseNAXFrag::load(acc[d], seed + band * WIDTH + 16 * d, WIDTH, Int<1>{});
        STEEL_PRAGMA_UNROLL
        for (short js = 0; js < 4; ++js) {
            ffrag mt;
            BaseNAXFrag::load(mt, mixer + 16 * js * CHUNK + band, CHUNK, Int<1>{});
            if (pass == 1) mt = -mt;
            STEEL_PRAGMA_UNROLL
            for (short d = 0; d < FRAGS; d += 2) {
                bfrag r0, r1;
                BaseNAXFrag::load(r0, source + 16 * js * WIDTH + 16 * d, WIDTH, Int<1>{});
                BaseNAXFrag::load(r1, source + 16 * js * WIDTH + 16 * (d + 1), WIDTH, Int<1>{});
                BaseNAXFrag::mma(acc[d], acc[d + 1], mt, metal::true_type{}, r0, r1,
                                 metal::false_type{});
            }
        }
        device bfloat* target = (pass == 0 ? s_du : s_dkt) + out0 + band * WIDTH;
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < FRAGS; ++d)
            BaseNAXFrag::store(acc[d], target + 16 * d, WIDTH, Int<1>{});
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // 3. dW and dKb, their direct terms, and dA.
    STEEL_PRAGMA_UNROLL
    for (short pass = 0; pass < 2; ++pass) {
        const device bfloat* source = (pass == 0 ? s_du : s_dkt) + out0;
        ffrag acc[FRAGS];
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < FRAGS; ++d) acc[d] = zero_ffrag();
        STEEL_PRAGMA_UNROLL
        for (short js = 0; js < 4; ++js) {
            ffrag xt;
            BaseNAXFrag::load(xt, lower + 16 * js * CHUNK + band, CHUNK, Int<1>{});
            STEEL_PRAGMA_UNROLL
            for (short d = 0; d < FRAGS; d += 2) {
                bfrag r0, r1;
                BaseNAXFrag::load(r0, source + 16 * js * WIDTH + 16 * d, WIDTH, Int<1>{});
                BaseNAXFrag::load(r1, source + 16 * js * WIDTH + 16 * (d + 1), WIDTH, Int<1>{});
                BaseNAXFrag::mma(acc[d], acc[d + 1], xt, metal::true_type{}, r0, r1,
                                 metal::false_type{});
            }
        }
        // pass 0: dv = beta dW and dbeta += sum(dW v).
        // pass 1: rowk = sum(dKb k), and s_dkb = beta e^lg dKb for step 5.
        ffrag rows = zero_ffrag();
        const device bfloat* other = pass == 0 ? vb : kb;
        const int stride = pass == 0 ? vs : ks;
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < FRAGS; ++d) {
            ffrag x, scaled;
            BaseNAXFrag::load(x, other + band * stride + 16 * d, stride, Int<1>{});
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < 8; ++e) {
                const int i = band + frag_row(sc, e);
                rows[e] += acc[d][e] * x[e];
                scaled[e] = acc[d][e] * (pass == 0 ? beta[i] : beta[i] * decay[i]);
            }
            if (pass == 0) {
                BaseNAXFrag::store(scaled, dv + (row0 * hv + h) * WIDTH + band * vs + 16 * d,
                                   vs, Int<1>{});
            } else {
                BaseNAXFrag::store(scaled, s_dkb + out0 + band * WIDTH + 16 * d, WIDTH,
                                   Int<1>{});
            }
        }
        add_band_row_sums(rows, pass == 0 ? dbeta : rowk, sg, sc);
        const device bfloat* right = pass == 0 ? u0b : ktb;
        STEEL_PRAGMA_UNROLL
        for (short kslice = 0; kslice < FRAGS; ++kslice) {
            STEEL_PRAGMA_UNROLL
            for (short cb = 0; cb < 4; cb += 2) {
                bfrag r0, r1;
                BaseNAXFrag::load(r0, right + 16 * cb * WIDTH + 16 * kslice, WIDTH, Int<1>{});
                BaseNAXFrag::load(r1, right + 16 * (cb + 1) * WIDTH + 16 * kslice, WIDTH,
                                  Int<1>{});
                BaseNAXFrag::mma(dA[cb], dA[cb + 1], acc[kslice], metal::false_type{}, r0, r1,
                                 metal::true_type{});
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    // 4. dkk, dS, the beta term of A, and P = dR R in the lower buffer.
    stage_keys(keys, kb, ks, nullptr, tid);
    {
        ffrag kk[4], qk[4];
        chunk_scores(kk, qk, qb, ks, keys, sg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        ffrag rows = zero_ffrag(), terms = zero_ffrag();
        STEEL_PRAGMA_UNROLL
        for (short cb = 0; cb < 4; ++cb)
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < 8; ++e) {
                const int i = band + frag_row(sc, e);
                const int j = 16 * cb + frag_col(sc, e);
                const float r = j <= i ? metal::exp(lg[i] - lg[j]) : 0.0f;
                const float grad_a = j < i ? -dA[cb][e] : 0.0f;
                const float term = (grad_a * beta[i] * kk[cb][e] + dm[cb][e] * qk[cb][e]) * r;
                rows[e] += grad_a * r * kk[cb][e];
                terms[e] += term;
                lower[i * CHUNK + j] = term;
                dA[cb][e] = grad_a * beta[i] * r;
                dm[cb][e] *= r;
            }
        add_band_row_sums(rows, dbeta, sg, sc);
        add_band_row_sums(terms, dlg, sg, sc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < CHUNK) {
        float column_sum = 0.0f;
        for (int i = 0; i < CHUNK; ++i) column_sum += lower[i * CHUNK + tid];
        dlg[tid] -= column_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup bfloat* dkk = reinterpret_cast<threadgroup bfloat*>(lower);
    STEEL_PRAGMA_UNROLL
    for (short cb = 0; cb < 4; ++cb)
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 8; ++e) {
            const int index = (band + frag_row(sc, e)) * CHUNK + 16 * cb + frag_col(sc, e);
            dkk[index] = bfloat(dA[cb][e]);
            mixer[index] = bfloat(dm[cb][e]);
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 5a. dq = e^lg dq_eff + dS k, and ddecay += sum(dq_eff q).
    {
        ffrag acc[FRAGS];
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < FRAGS; ++d) acc[d] = zero_ffrag();
        STEEL_PRAGMA_UNROLL
        for (short js = 0; js < 4; ++js) {
            bfrag sa;
            BaseNAXFrag::load(sa, mixer + band * CHUNK + 16 * js, CHUNK, Int<1>{});
            STEEL_PRAGMA_UNROLL
            for (short d = 0; d < FRAGS; d += 2) {
                bfrag k0, k1;
                BaseNAXFrag::load(k0, kb + 16 * js * ks + 16 * d, ks, Int<1>{});
                BaseNAXFrag::load(k1, kb + 16 * js * ks + 16 * (d + 1), ks, Int<1>{});
                BaseNAXFrag::mma(acc[d], acc[d + 1], sa, metal::false_type{}, k0, k1,
                                 metal::false_type{});
            }
        }
        ffrag rows = zero_ffrag();
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < FRAGS; ++d) {
            ffrag g, x;
            BaseNAXFrag::load(g, dqe + band * WIDTH + 16 * d, WIDTH, Int<1>{});
            BaseNAXFrag::load(x, qb + band * ks + 16 * d, ks, Int<1>{});
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < 8; ++e) {
                rows[e] += g[e] * x[e];
                acc[d][e] += decay[band + frag_row(sc, e)] * g[e];
            }
            BaseNAXFrag::store(acc[d], dq + (row0 * hv + h) * WIDTH + band * vs + 16 * d, vs,
                               Int<1>{});
        }
        add_band_row_sums(rows, ddecay, sg, sc);
    }
    // 5b. dk = (dkk + dkk^T) k + dS^T q + beta e^lg dKb + dk_copy.
    {
        ffrag acc[FRAGS];
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < FRAGS; ++d) {
            ffrag x, y;
            BaseNAXFrag::load(x, s_dkb + out0 + band * WIDTH + 16 * d, WIDTH, Int<1>{});
            BaseNAXFrag::load(y, d_k_chunks + out0 + band * WIDTH + 16 * d, WIDTH, Int<1>{});
            acc[d] = x + y;
        }
        STEEL_PRAGMA_UNROLL
        for (short js = 0; js < 4; ++js) {
            bfrag a1, a2, a3;
            BaseNAXFrag::load(a1, dkk + band * CHUNK + 16 * js, CHUNK, Int<1>{});
            BaseNAXFrag::load(a2, dkk + 16 * js * CHUNK + band, CHUNK, Int<1>{});
            BaseNAXFrag::load(a3, mixer + 16 * js * CHUNK + band, CHUNK, Int<1>{});
            STEEL_PRAGMA_UNROLL
            for (short d = 0; d < FRAGS; d += 2) {
                bfrag k0, k1, q0, q1;
                BaseNAXFrag::load(k0, kb + 16 * js * ks + 16 * d, ks, Int<1>{});
                BaseNAXFrag::load(k1, kb + 16 * js * ks + 16 * (d + 1), ks, Int<1>{});
                BaseNAXFrag::load(q0, qb + 16 * js * ks + 16 * d, ks, Int<1>{});
                BaseNAXFrag::load(q1, qb + 16 * js * ks + 16 * (d + 1), ks, Int<1>{});
                BaseNAXFrag::mma(acc[d], acc[d + 1], a1, metal::false_type{}, k0, k1,
                                 metal::false_type{});
                BaseNAXFrag::mma(acc[d], acc[d + 1], a2, metal::true_type{}, k0, k1,
                                 metal::false_type{});
                BaseNAXFrag::mma(acc[d], acc[d + 1], a3, metal::true_type{}, q0, q1,
                                 metal::false_type{});
            }
        }
        STEEL_PRAGMA_UNROLL
        for (short d = 0; d < FRAGS; ++d)
            BaseNAXFrag::store(acc[d], dk + (row0 * hv + h) * WIDTH + band * vs + 16 * d, vs,
                               Int<1>{});
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 6. Gate terms, the suffix sum, and the logit gradients.
    const size_t chunk_rows = block * CHUNK;
    if (tid < CHUNK) {
        const float ratio = metal::exp(lg[CHUNK - 1] - lg[tid]);
        float total = dlg[tid] + (ddecay[tid] + beta[tid] * rowk[tid]) * decay[tid]
            - float(d_ratio_last[chunk_rows + tid]) * ratio;
        if (tid == CHUNK - 1) {
            total += float(d_decay_last[block]) * decay[CHUNK - 1];
            for (int j = 0; j < CHUNK; ++j)
                total += float(d_ratio_last[chunk_rows + j])
                    * metal::exp(lg[CHUNK - 1] - lg[j]);
        }
        dbeta[tid] += decay[tid] * rowk[tid];
        ddecay[tid] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < CHUNK) {
        float suffix = 0.0f;
        for (int t = tid; t < CHUNK; ++t) suffix += ddecay[t];
        const size_t gate = (row0 + tid) * hv + h;
        const float slope = sampler_gate(a[gate], float(a_log[h]), float(dt_bias[h])).slope;
        da[gate] = bfloat(suffix * slope);
        db[gate] = bfloat(dbeta[tid] * beta[tid] * (1.0f - beta[tid]));
    }
