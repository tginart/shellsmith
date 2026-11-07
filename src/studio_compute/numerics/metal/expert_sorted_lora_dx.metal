// Purpose: Input gradient of one or two sorted routed-expert LoRA projections
//     that read the same input rows.
// Input: For pair p < PAIRS: dy_p (R, KR) BF16 rows sorted by expert, the 4-bit
//     weights w_p (E, KR, NO / 8) with scales_p and biases_p (E, KR, NO / GS),
//     and the LoRA factor lora_p (E, RANK, NO). dlow (R, 2 * RANK) holds
//     scale * dy_p B_p[e] for each pair, and zeros after the last pair. Row and tile offsets per expert (E + 1),
//     and dims [R, KR, NO, E].
// Output: dx (R, NO) BF16, with dx[r] = sum_p dy_p[r] W_p[e] + dlow_p[r] A_p[e].
// Method:
//     1. The tiles follow expert_sorted_dx.metal: 64 rows of one expert and 128
//        output columns per threadgroup, 32 x 64 per simdgroup.
//     2. One step sequence runs over the KR rows of all pairs. The next packed
//        words load while the NAX fragments use the current stage.
//     3. A first 32-deep step adds the LoRA terms of both pairs from one stage.
// Note: The kernel writes dx once. Separate kernels would write one (R, NO)
//     array per term and add them.
    constexpr int BM = 64, BN = 128, BK = 32, LD = BN + 8, RANK = 16;
    const int KR = dims[1], NO = dims[2], E = dims[3];
    const int n0 = threadgroup_position_in_grid.x * BN;
    const int t = threadgroup_position_in_grid.y;
    const ushort tid = thread_index_in_threadgroup;
    const ushort sg = simdgroup_index_in_threadgroup;
    if (t >= tile_offsets[E]) return;
    int lo = 0, hi = E;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (tile_offsets[mid] <= t) lo = mid; else hi = mid;
    }
    const int e = lo;
    const int row0 = row_offsets[e] + (t - tile_offsets[e]) * BM;
    const int rows = min(BM, row_offsets[e + 1] - row0);

    threadgroup bfloat ws[2 * BK * LD];
    const int words = NO / 8, groups = NO / GS;
    const short drow = tid / 4, dq = tid % 4;
    const size_t wrow = size_t(e) * KR + drow;
    const size_t woff = wrow * words + n0 / 8 + dq * 4;
    const size_t soff = wrow * groups + (n0 + 32 * dq) / GS;
    const int steps = KR / BK, total = PAIRS * steps;

    const short sm = sg / 2, sn = sg % 2;
    const short srows = short(rows - 32 * sm);
    const size_t xoff = size_t(row0 + 32 * sm) * KR;
    ffrag c[2][4];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; ++i)
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 4; ++j) c[i][j] = zero_ffrag();

    // The LoRA step runs first. Rows 0-15 of its stage are A_0 and rows 16-31
    // are A_1. For one pair, A_1 is A_0 and the dlow columns 16-31 are zero.
    threadgroup bfloat* stage = ws + BK * LD;
    {
        const device uint4* lrow = (const device uint4*)((drow < RANK ? lora0 : lora1)
            + (size_t(e) * RANK + drow % RANK) * NO + n0 + 32 * dq);
        threadgroup uint4* dst = (threadgroup uint4*)(stage + drow * LD + dq * 32);
        STEEL_PRAGMA_UNROLL
        for (short q = 0; q < 4; ++q) dst[q] = lrow[q];
    }
    uint4 packed = *((const device uint4*)(w0 + woff));
    float scale = float(scales0[soff]), bias = float(biases0[soff]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (srows > 0)
        mma_step(c, dlow + size_t(row0 + 32 * sm) * BK, BK, srows, stage, sn);
    stage = ws;
    dequant_32(stage + drow * LD + dq * 32, packed, scale, bias);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short p = 0; p < PAIRS; ++p) {
        const device uint32_t* wp = (p == 0 ? w0 : w1) + woff;
        const auto sp = (p == 0 ? scales0 : scales1) + soff;
        const auto bp = (p == 0 ? biases0 : biases1) + soff;
        const device bfloat* xp = (p == 0 ? dy0 : dy1) + xoff;
        for (int k0 = 0; k0 < KR; k0 += BK) {
            const bool last = k0 + BK >= KR;
            const bool has_next = !last || p + 1 < PAIRS;
            if (has_next) {
                // The first step of pair 1 follows the last step of pair 0.
                const size_t k = last ? 0 : size_t(k0 + BK);
                packed = *((const device uint4*)((last ? w1 + woff : wp) + k * words));
                scale = float((last ? scales1 + soff : sp)[k * groups]);
                bias = float((last ? biases1 + soff : bp)[k * groups]);
            }
            if (srows > 0) mma_step(c, xp + k0, KR, srows, stage, sn);
            // The other stage is free: every simdgroup passed the last barrier
            // after its products on it.
            if (has_next) {
                stage = stage == ws ? ws + BK * LD : ws;
                dequant_32(stage + drow * LD + dq * 32, packed, scale, bias);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (srows <= 0) return;
    device bfloat* yp = dx + size_t(row0 + 32 * sm) * NO + n0 + 64 * sn;
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < 4; ++j) {
        BaseNAXFrag::store_rows(c[0][j], yp + 16 * j, NO, Int<1>{}, srows);
        BaseNAXFrag::store_rows(c[1][j], yp + 16 * NO + 16 * j, NO, Int<1>{},
                                short(srows - 16));
    }
