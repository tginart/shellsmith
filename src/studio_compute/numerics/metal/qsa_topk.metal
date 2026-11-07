// Purpose: Select the K largest block scores of one query row.
// Input: FP32 scores (L, B), dims [L, B, K].
// Output: int32 block indices (L, K) in ascending order.
// Method:
//     1. Map each score to an unsigned key with the same order.
//     2. Four 8-bit radix passes find the key of the K-th largest score, and
//        how many scores equal to it are part of the selection.
//     3. Each thread owns a contiguous run of blocks. Prefix sums over the
//        threads give each selected block its output place.
// Note: Of equal scores at the cut-off, the lowest block indices win.
    constexpr int THREADS = 256;
    const int row = threadgroup_position_in_grid.x;
    const ushort tid = thread_index_in_threadgroup;
    const ushort lane = thread_index_in_simdgroup;
    const ushort sg = simdgroup_index_in_threadgroup;
    const int B = dims[1], K = dims[2];
    const auto s = scores + size_t(row) * B;

    threadgroup atomic_uint histogram[256];
    threadgroup uint chosen[2];
    threadgroup uint sums[2][THREADS / 32];

    uint prefix = 0, mask = 0, want = uint(K);
    for (int shift = 24; shift >= 0; shift -= 8) {
        atomic_store_explicit(&histogram[tid], 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < B; i += THREADS) {
            const uint key = order_key(s[i]);
            if ((key & mask) == prefix)
                atomic_fetch_add_explicit(&histogram[(key >> shift) & 255u], 1u,
                                          memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint above = 0;
            int bin = 255;
            for (; bin > 0; --bin) {
                const uint count = atomic_load_explicit(&histogram[bin], memory_order_relaxed);
                if (above + count >= want) break;
                above += count;
            }
            chosen[0] = uint(bin);
            chosen[1] = want - above;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix |= chosen[0] << shift;
        mask |= 255u << shift;
        want = chosen[1];
    }

    // The cut-off key is `prefix`. `want` scores equal to it are selected.
    const int run = (B + THREADS - 1) / THREADS;
    const int begin = min(B, int(tid) * run);
    const int end = min(B, begin + run);
    uint above = 0, equal = 0;
    for (int i = begin; i < end; ++i) {
        const uint key = order_key(s[i]);
        above += key > prefix;
        equal += key == prefix;
    }
    // Exclusive prefix sums over the threads, for both counts.
    const uint above_scan = simd_prefix_exclusive_sum(above);
    const uint equal_scan = simd_prefix_exclusive_sum(equal);
    if (lane == 31) {
        sums[0][sg] = above_scan + above;
        sums[1][sg] = equal_scan + equal;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint above_base = above_scan, equal_base = equal_scan;
    for (ushort g = 0; g < sg; ++g) {
        above_base += sums[0][g];
        equal_base += sums[1][g];
    }
    // Blocks before this run: the selected larger ones, and the selected equal ones.
    uint place = above_base + min(equal_base, want);
    device int* out = selected + size_t(row) * K;
    for (int i = begin; i < end; ++i) {
        const uint key = order_key(s[i]);
        bool take = key > prefix;
        if (key == prefix) {
            take = equal_base < want;
            ++equal_base;
        }
        if (take) out[place++] = i;
    }
