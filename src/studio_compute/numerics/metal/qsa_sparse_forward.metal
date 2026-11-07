// Purpose: Sparse QSA attention forward for one query row and one KV head.
// Input: q (L * H + 16, D), k and v (J, Lk, D), sorted block slots (L, S),
//     valid slot counts (L), query ends (L), dims [L, Lk, S], and scale.
// Output: o (L, H, D) BF16 and the log2 normalizer lse (L, H) FP32.
// Method:
//     1. The query heads of the KV group are the 16 fragment rows. Each
//        simdgroup keeps its 64-wide Q slice in registers.
//     2. A chunk reads eight selected blocks (32 keys) straight from device rows.
//        Each simdgroup forms partial scores; the four partials are summed.
//     3. One online softmax step runs on the score fragments in registers.
//     4. Each simdgroup adds P V to its 64 output columns.
    constexpr int CK = 32;
    const int i = threadgroup_position_in_grid.x;
    const int j = threadgroup_position_in_grid.y;
    const ushort tid = thread_index_in_threadgroup;
    const ushort sg = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const short2 sc = BaseNAXFrag::get_coord();
    const int Lk = dims[1], S = dims[2];
    const float qscale = scale[0] * LOG2E;

    threadgroup float xchg[4 * 2 * 8 * 32];
    threadgroup int keys[CK];

    const int head = i * H + j * G;
    const device bfloat* kp = k + j * Lk * D;
    const device bfloat* vp = v + j * Lk * D;
    const int end = ends[i];
    const int count = counts[i];

    bfrag qf[4];
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < 4; ++id)
        BaseNAXFrag::load(qf[id], q + head * D + 64 * sg + 16 * id, D, Int<1>{});
    ffrag of[4];
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < 4; ++id) of[id] = zero_ffrag();
    float m_row[2] = {FINITE_MIN, FINITE_MIN};
    float l_row[2] = {0.0f, 0.0f};

    for (int base = 0; base < count; base += CK / 4) {
        if (tid < CK) {
            int slot = base + tid / 4;
            int block = slot < count ? slots[i * S + slot] : -1;
            int token = block * 4 + tid % 4;
            keys[tid] = (block >= 0 && token < end) ? token : -1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        ffrag sf[2] = {zero_ffrag(), zero_ffrag()};
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < 4; ++id) {
            bfrag k0, k1;
            load_gathered(k0, kp, keys, 64 * sg + 16 * id, sc);
            load_gathered(k1, kp, keys + 16, 64 * sg + 16 * id, sc);
            BaseNAXFrag::mma(sf[0], sf[1], qf[id], metal::false_type{}, k0, k1,
                             metal::true_type{});
        }
        exchange_four(sf, 2, xchg, sg, lane);

        float m_new[2] = {m_row[0], m_row[1]};
        STEEL_PRAGMA_UNROLL
        for (short f = 0; f < 2; ++f)
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < 8; ++e) {
                int key = 16 * f + sc.x + e % 4;
                float value = keys[key] >= 0 ? sf[f][e] * qscale : NEG_INF;
                sf[f][e] = value;
                m_new[e / 4] = max(m_new[e / 4], value);
            }
        STEEL_PRAGMA_UNROLL
        for (short r = 0; r < 2; ++r) {
            m_new[r] = row_max4(m_new[r]);
            float factor = fast::exp2(m_row[r] - m_new[r]);
            float total = 0.0f;
            STEEL_PRAGMA_UNROLL
            for (short f = 0; f < 2; ++f)
                STEEL_PRAGMA_UNROLL
                for (short e = 4 * r; e < 4 * r + 4; ++e) {
                    float p = fast::exp2(sf[f][e] - m_new[r]);
                    sf[f][e] = p;
                    total += p;
                }
            l_row[r] = l_row[r] * factor + row_sum4(total);
            m_row[r] = m_new[r];
            STEEL_PRAGMA_UNROLL
            for (short id = 0; id < 4; ++id)
                STEEL_PRAGMA_UNROLL
                for (short e = 4 * r; e < 4 * r + 4; ++e) of[id][e] *= factor;
        }

        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < 4; id += 2) {
            STEEL_PRAGMA_UNROLL
            for (short f = 0; f < 2; ++f) {
                bfrag v0, v1;
                load_gathered(v0, vp, keys + 16 * f, 64 * sg + 16 * id, sc);
                load_gathered(v1, vp, keys + 16 * f, 64 * sg + 16 * (id + 1), sc);
                BaseNAXFrag::mma(of[id], of[id + 1], sf[f], metal::false_type{}, v0, v1,
                                 metal::false_type{});
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < 2; ++r) {
        int row = sc.y + 8 * r;
        if (row >= G) continue;
        float inverse = l_row[r] > 0.0f ? 1.0f / l_row[r] : 0.0f;
        if (sg == 0 && sc.x == 0) lse[head + row] = m_row[r] + log2(l_row[r]);
        device bfloat* op = o + (head + row) * D + 64 * sg + sc.x;
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < 4; ++id)
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < 4; ++e) op[16 * id + e] = bfloat(of[id][4 * r + e] * inverse);
    }
