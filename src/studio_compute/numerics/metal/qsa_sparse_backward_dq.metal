// Purpose: Sparse QSA attention dQ for one query row and one KV head.
// Input: q and dy (L * H + 16, D), k and v (J, Lk, D), lse and delta (L * H + 16),
//     sorted block slots (L, S), valid slot counts (L), query ends (L),
//     dims [L, Lk, S], and scale.
// Output: BF16 dq (L, H, D), and BF16 P and dS (L, S, H, 4) for the dK and dV
//     kernel. Entry (i, s, h, t) belongs to key t of the block in slot s of query i.
// Method:
//     1. Each simdgroup keeps its 64-wide Q and dY slices in registers.
//     2. A chunk reads eight selected blocks (32 keys) of K and V from device
//        rows, forms partial scores and dP, and sums the four slices.
//     3. P comes from the forward normalizer. dS = P (dP - delta).
//     4. dQ += dS K stays in registers. The threadgroup owns its dQ rows, so it
//        writes them once, without atomics. It also writes P and dS of each key.
    constexpr int CK = 32;
    const int i = threadgroup_position_in_grid.x;
    const int j = threadgroup_position_in_grid.y;
    const ushort tid = thread_index_in_threadgroup;
    const ushort sg = simdgroup_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const short2 sc = BaseNAXFrag::get_coord();
    const int Lk = dims[1], S = dims[2];
    const float qscale = scale[0] * LOG2E;

    threadgroup float xchg[5 * 32 * 32];
    threadgroup int keys[CK];

    const int head = i * H + j * G;
    const device bfloat* kp = k + j * Lk * D;
    const device bfloat* vp = v + j * Lk * D;
    const int end = ends[i];
    const int count = counts[i];

    bfrag qf[4], yf[4];
    ffrag of[4];
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < 4; ++id) {
        BaseNAXFrag::load(qf[id], q + head * D + 64 * sg + 16 * id, D, Int<1>{});
        BaseNAXFrag::load(yf[id], dy + head * D + 64 * sg + 16 * id, D, Int<1>{});
        of[id] = zero_ffrag();
    }
    float lse_row[2], delta_row[2];
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < 2; ++r) {
        lse_row[r] = lse[head + sc.y + 8 * r];
        delta_row[r] = delta[head + sc.y + 8 * r];
    }

    for (int base = 0; base < count; base += CK / 4) {
        if (tid < CK) {
            int slot = base + tid / 4;
            int block = slot < count ? slots[i * S + slot] : -1;
            int token = block * 4 + tid % 4;
            keys[tid] = (block >= 0 && token < end) ? token : -1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Fragments 0-1 hold scores and 2-3 hold dP for keys 0-15 and 16-31.
        ffrag sf[4] = {zero_ffrag(), zero_ffrag(), zero_ffrag(), zero_ffrag()};
        // kf[2 * id + f] holds keys 16f to 16f + 15 of column block id. The dQ
        // product reuses them.
        bfrag kf[8];
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < 4; ++id) {
            bfrag v0, v1;
            load_gathered(kf[2 * id], kp, keys, 64 * sg + 16 * id, sc);
            load_gathered(kf[2 * id + 1], kp, keys + 16, 64 * sg + 16 * id, sc);
            load_gathered(v0, vp, keys, 64 * sg + 16 * id, sc);
            load_gathered(v1, vp, keys + 16, 64 * sg + 16 * id, sc);
            BaseNAXFrag::mma(sf[0], sf[1], qf[id], metal::false_type{}, kf[2 * id],
                             kf[2 * id + 1], metal::true_type{});
            BaseNAXFrag::mma(sf[2], sf[3], yf[id], metal::false_type{}, v0, v1,
                             metal::true_type{});
        }
        reduce_scatter_four(sf, xchg, sg, lane);

        STEEL_PRAGMA_UNROLL
        for (short f = 0; f < 2; ++f)
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < 8; ++e) {
                int key = 16 * f + sc.x + e % 4;
                int row = e / 4;
                float p = keys[key] >= 0
                    ? fast::exp2(sf[f][e] * qscale - lse_row[row]) : 0.0f;
                sf[f][e] = p * (sf[f + 2][e] - delta_row[row]);
                const int slot = base + key / 4;
                const int h = sc.y + 8 * row;
                // All four simdgroups hold the same sums; one writes them.
                if (sg == 0 && slot < count && h < G) {
                    const int index = ((i * S + slot) * H + j * G + h) * 4 + key % 4;
                    p_store[index] = bfloat(p);
                    ds_store[index] = bfloat(sf[f][e]);
                }
            }

        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < 4; id += 2) {
            STEEL_PRAGMA_UNROLL
            for (short f = 0; f < 2; ++f)
                BaseNAXFrag::mma(of[id], of[id + 1], sf[f], metal::false_type{},
                                 kf[2 * id + f], kf[2 * id + 2 + f], metal::false_type{});
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < 2; ++r) {
        int row = sc.y + 8 * r;
        if (row >= G) continue;
        device bfloat* dqp = dq + (head + row) * D + 64 * sg + sc.x;
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < 4; ++id)
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < 4; ++e)
                dqp[16 * id + e] = bfloat(of[id][4 * r + e] * scale[0]);
    }
