// Purpose: Input gradient of the sorted routed-expert 4-bit projection.
// Input: dy (R, KR) BF16 rows sorted by expert, the expert weights w (E, KR, NO / 8)
//     with scales and biases (E, KR, NO / GS), row and tile offsets per expert
//     (E + 1), and dims [R, KR, NO, E].
// Output: dx (R, NO) BF16, with dx[r] = dy[r] W[e(r)].
// Method:
//     1. One threadgroup owns a tile of up to 64 rows of one expert and 128
//        output columns. A binary search of the tile offsets gives the expert.
//     2. Each step dequantizes 32 weight rows into one of two threadgroup stages.
//        The next packed words load while the NAX fragments use this stage.
//     3. Each simdgroup owns 32 rows and 64 columns. A 16-row half that has no
//        valid row skips its products.
// Note: The forward weight is W (KR, NO), quantized along NO, so the tile
//     columns follow the quantization groups.
    constexpr int BM = 64, BN = 128, BK = 32, LD = BN + 8;
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
    const device uint32_t* wp = w + wrow * words + n0 / 8 + dq * 4;
    const auto sp = scales + wrow * groups + (n0 + 32 * dq) / GS;
    const auto bp = biases + wrow * groups + (n0 + 32 * dq) / GS;

    const short sm = sg / 2, sn = sg % 2;
    const short srows = short(rows - 32 * sm);
    const device bfloat* xp = dy + size_t(row0 + 32 * sm) * KR;
    ffrag c[2][4];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; ++i)
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 4; ++j) c[i][j] = zero_ffrag();

    uint4 packed = *((const device uint4*)wp);
    float scale = float(sp[0]), bias = float(bp[0]);
    threadgroup bfloat* stage = ws;
    dequant_32(stage + drow * LD + dq * 32, packed, scale, bias);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k0 = 0; k0 < KR; k0 += BK) {
        const bool has_next = k0 + BK < KR;
        if (has_next) {
            packed = *((const device uint4*)(wp + size_t(k0 + BK) * words));
            scale = float(sp[size_t(k0 + BK) * groups]);
            bias = float(bp[size_t(k0 + BK) * groups]);
        }
        if (srows > 0) {
            STEEL_PRAGMA_UNROLL
            for (short kk = 0; kk < BK; kk += 16) {
                bfrag a0, a1, bf[4];
                BaseNAXFrag::load_rows(a0, xp + k0 + kk, KR, Int<1>{}, srows);
                BaseNAXFrag::load_rows(a1, xp + 16 * KR + k0 + kk, KR, Int<1>{},
                                       short(srows - 16));
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; ++j)
                    BaseNAXFrag::load(bf[j], stage + kk * LD + 64 * sn + 16 * j, LD,
                                      Int<1>{});
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; j += 2) {
                    BaseNAXFrag::mma(c[0][j], c[0][j + 1], a0, metal::false_type{}, bf[j],
                                     bf[j + 1], metal::false_type{});
                    if (srows > 16)
                        BaseNAXFrag::mma(c[1][j], c[1][j + 1], a1, metal::false_type{},
                                         bf[j], bf[j + 1], metal::false_type{});
                }
            }
        }
        // The other stage is free: every simdgroup passed the last barrier after
        // its products on it.
        if (has_next) {
            stage = stage == ws ? ws + BK * LD : ws;
            dequant_32(stage + drow * LD + dq * 32, packed, scale, bias);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (srows > 0) {
        device bfloat* yp = dx + size_t(row0 + 32 * sm) * NO + n0 + 64 * sn;
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 4; ++j) {
            BaseNAXFrag::store_rows(c[0][j], yp + 16 * j, NO, Int<1>{}, srows);
            BaseNAXFrag::store_rows(c[1][j], yp + 16 * NO + 16 * j, NO, Int<1>{},
                                    short(srows - 16));
        }
    }
