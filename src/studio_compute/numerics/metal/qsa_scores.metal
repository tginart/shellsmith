// Purpose: QSA indexer block scores of four query rows against a range of blocks.
// Input: rotated query heads q (N = 4, L, 128) BF16, pooled block keys k (B, 128)
//     BF16, complete block counts (L), dims [L, B, width, tiles per simdgroup].
// Output: FP32 scores (L, width): sum over heads of ReLU(q . k) / sqrt(128) for
//     a visible block, and -inf for the other blocks and the padding columns.
// Method:
//     1. Fragment row r is head r / 4 of query r % 4. Each simdgroup keeps the
//        16 query rows in registers as 8 fragments over the width.
//     2. A tile is 32 blocks. Eight MMAs give the 16 x 32 products in FP32.
//     3. Element e holds heads h and h + 2 of one query, and lane l ^ 16 holds
//        heads h + 1 and h + 3. One add and one shuffle sum the four ReLU terms.
// Note: The products of BF16 values are exact in FP32, as in the upstream
//     FP32 matmul. Only the order of the sum differs.
    constexpr int WIDTH = 128;
    constexpr int HEADS = 4;
    const int L = dims[0], B = dims[1], Bp = dims[2], T = dims[3];
    const ushort lane = thread_index_in_simdgroup;
    const ushort sg = simdgroup_index_in_threadgroup;
    const int q0 = 4 * threadgroup_position_in_grid.y;
    const int first_tile = (threadgroup_position_in_grid.x * 4 + sg) * T;
    const short2 sc = BaseNAXFrag::get_coord();

    bfrag qf[WIDTH / 16];
    STEEL_PRAGMA_UNROLL
    for (short d = 0; d < WIDTH / 16; ++d)
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; ++i) {
            const int row = sc.y + 8 * i;
            const int query = min(q0 + row % 4, L - 1);
            const auto source = q + (size_t(row / 4) * L + query) * WIDTH + 16 * d + sc.x;
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; ++j) qf[d][4 * i + j] = source[j];
        }
    const int query = q0 + sc.y % 4;
    const int visible = query < L ? complete[query] : 0;
    const float scale = metal::sqrt(float(WIDTH));

    for (int tile = first_tile; tile < first_tile + T; ++tile) {
        const int b0 = 32 * tile;
        if (b0 >= Bp) break;
        ffrag c[2] = {zero_ffrag(), zero_ffrag()};
        if (b0 < B) {
            STEEL_PRAGMA_UNROLL
            for (short d = 0; d < WIDTH / 16; ++d) {
                bfrag k0, k1;
                STEEL_PRAGMA_UNROLL
                for (short i = 0; i < 2; ++i) {
                    const int r0 = b0 + sc.y + 8 * i, r1 = r0 + 16;
                    const auto p0 = k + size_t(min(r0, B - 1)) * WIDTH + 16 * d + sc.x;
                    const auto p1 = k + size_t(min(r1, B - 1)) * WIDTH + 16 * d + sc.x;
                    STEEL_PRAGMA_UNROLL
                    for (short j = 0; j < 4; ++j) {
                        k0[4 * i + j] = p0[j];
                        k1[4 * i + j] = p1[j];
                    }
                }
                BaseNAXFrag::mma(c[0], c[1], qf[d], metal::false_type{}, k0, k1,
                                 metal::true_type{});
            }
        }
        STEEL_PRAGMA_UNROLL
        for (short f = 0; f < 2; ++f)
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; ++j) {
                float total = metal::max(c[f][j], 0.0f) + metal::max(c[f][4 + j], 0.0f);
                total += simd_shuffle_xor(total, ushort(16));
                const int block = b0 + 16 * f + sc.x + j;
                if ((sc.y & 4) == 0 && query < L && block < Bp)
                    scores[size_t(query) * Bp + block] =
                        block < min(visible, B) ? total / scale : -INFINITY;
            }
    }
