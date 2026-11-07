// Purpose: Sparse QSA attention dK and dV for one 16-key tile and one KV head.
// Input: q and dy (L * H + 16, D), P and dS (L, S, H, 4) from the dQ kernel,
//     the query lists of the tiles (entries, offsets), the tile order,
//     dims [L, Lk, T, S], and scale.
// Output: BF16 dk and dv (J, Lk, D).
// Method:
//     1. A tile is four key blocks. Its list holds each query that selected one
//        of the blocks, as query << 14 | first slot << 4 | block mask. The
//        selected blocks of the tile sit in consecutive slots from the first
//        slot on. Each query gives G list rows.
//     2. A step takes 64 list rows. The threadgroup gathers their P and dS for
//        the 16 keys into threadgroup memory, transposed. One thread reads the
//        four keys of one block and row as one vector.
//     3. Each simdgroup owns 64 columns of D. dV += P^T dY and dK += dS^T Q stay
//        in registers. The threadgroup owns its key rows, so it writes them
//        once, without atomics.
    constexpr int TK = 16;
    constexpr int RS = 64;
    constexpr int PS = RS + 8;
    const int t = order[threadgroup_position_in_grid.x];
    const int j = threadgroup_position_in_grid.y;
    const ushort tid = thread_index_in_threadgroup;
    const ushort sg = simdgroup_index_in_threadgroup;
    const short2 sc = BaseNAXFrag::get_coord();
    const int Lk = dims[1], S = dims[3];

    threadgroup bfloat pt[TK * PS];
    threadgroup bfloat dst[TK * PS];
    threadgroup int rows[RS];
    threadgroup int masks[RS];
    threadgroup int bases[RS];

    const int begin = offsets[t];
    const int total = (offsets[t + 1] - begin) * G;
    ffrag dkf[4], dvf[4];
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < 4; ++id) {
        dkf[id] = zero_ffrag();
        dvf[id] = zero_ffrag();
    }

    for (int base = 0; base < total; base += RS) {
        if (tid < RS) {
            const int r = base + tid;
            int row = -1, mask = 0, start = 0;
            if (r < total) {
                const int entry = entries[begin + r / G];
                const int i = entry >> 14;
                const int h = r % G;
                mask = entry & 15;
                row = i * H + j * G + h;
                start = ((i * S + ((entry >> 4) & 1023)) * H + j * G + h) * 4;
            }
            rows[tid] = row;
            masks[tid] = mask;
            bases[tid] = start;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_UNROLL
        for (short pass = 0; pass < TK / 4 * RS / 128; ++pass) {
            const int index = pass * 128 + tid;
            const int column = index % RS;
            const int block = index / RS;
            const int mask = masks[column];
            const bool valid = rows[column] >= 0 && ((mask >> block) & 1);
            const int slot = metal::popcount(mask & ((1 << block) - 1));
            const int offset = bases[column] + slot * H * 4;
            const vec<bfloat, 4> p = valid
                ? *reinterpret_cast<const device vec<bfloat, 4>*>(p_store + offset)
                : vec<bfloat, 4>(0);
            const vec<bfloat, 4> ds = valid
                ? *reinterpret_cast<const device vec<bfloat, 4>*>(ds_store + offset)
                : vec<bfloat, 4>(0);
            STEEL_PRAGMA_UNROLL
            for (short t = 0; t < 4; ++t) {
                pt[(4 * block + t) * PS + column] = p[t];
                dst[(4 * block + t) * PS + column] = ds[t];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_UNROLL
        for (short part = 0; part < RS / 16; ++part) {
            bfrag pf, df, qf[4], yf[4];
            BaseNAXFrag::load(pf, pt + 16 * part, PS, Int<1>{});
            BaseNAXFrag::load(df, dst + 16 * part, PS, Int<1>{});
            STEEL_PRAGMA_UNROLL
            for (short id = 0; id < 4; ++id) {
                load_gathered(qf[id], q, rows + 16 * part, 64 * sg + 16 * id, sc);
                load_gathered(yf[id], dy, rows + 16 * part, 64 * sg + 16 * id, sc);
            }
            STEEL_PRAGMA_UNROLL
            for (short id = 0; id < 4; id += 2) {
                BaseNAXFrag::mma(dvf[id], dvf[id + 1], pf, metal::false_type{}, yf[id],
                                 yf[id + 1], metal::false_type{});
                BaseNAXFrag::mma(dkf[id], dkf[id + 1], df, metal::false_type{}, qf[id],
                                 qf[id + 1], metal::false_type{});
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < 2; ++r) {
        const int token = t * TK + sc.y + 8 * r;
        if (token >= Lk) continue;
        const int offset = (j * Lk + token) * D + 64 * sg + sc.x;
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < 4; ++id)
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < 4; ++e) {
                dk[offset + 16 * id + e] = bfloat(dkf[id][4 * r + e] * scale[0]);
                dv[offset + 16 * id + e] = bfloat(dvf[id][4 * r + e]);
            }
    }
