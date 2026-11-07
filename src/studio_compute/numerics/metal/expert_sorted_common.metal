// Purpose: Shared helpers for the sorted routed-expert kernels.
// Note: The header follows gdn_nax_common.metal, which defines the 16x16 NAX
//     fragments. Element e of a fragment is row fm + 8 * (e / 4) and column
//     fn + e % 4, with (fn, fm) = BaseNAXFrag::get_coord().

typedef metal::vec<float, 8> ffrag;
typedef metal::vec<bfloat, 8> bfrag;

inline ffrag zero_ffrag() {
    return ffrag(0.0f);
}

// Write 32 BF16 weights from four packed 4-bit words. Nibble j of a word is
// column j, as in the MLX affine format: value = scale * q + bias.
inline void dequant_32(threadgroup bfloat* dst_, uint4 packed, float scale, float bias) {
    threadgroup vec<bfloat, 8>* dst = (threadgroup vec<bfloat, 8>*)dst_;
    STEEL_PRAGMA_UNROLL
    for (short q = 0; q < 4; ++q) {
        uint32_t word = packed[q];
        vec<bfloat, 8> out;
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 8; ++j) out[j] = bfloat(scale * float((word >> (4 * j)) & 0xF) + bias);
        dst[q] = out;
    }
}

// Add one 32-deep step to the 32 x 64 tile of a simdgroup. The x rows are at
// xp with row stride ld, srows of them valid. The stage holds 32 rows of
// 128 BF16 columns with row stride 136; the simdgroup uses columns 64 sn.
inline void mma_step(thread ffrag (&c)[2][4], const device bfloat* xp, int ld,
                     short srows, const threadgroup bfloat* stage, short sn) {
    constexpr int LD = 136;
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < 32; kk += 16) {
        bfrag a0, a1, bf[4];
        BaseNAXFrag::load_rows(a0, xp + kk, ld, Int<1>{}, srows);
        BaseNAXFrag::load_rows(a1, xp + 16 * ld + kk, ld, Int<1>{}, short(srows - 16));
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 4; ++j)
            BaseNAXFrag::load(bf[j], stage + kk * LD + 64 * sn + 16 * j, LD, Int<1>{});
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 4; j += 2) {
            BaseNAXFrag::mma(c[0][j], c[0][j + 1], a0, metal::false_type{}, bf[j],
                             bf[j + 1], metal::false_type{});
            if (srows > 16)
                BaseNAXFrag::mma(c[1][j], c[1][j + 1], a1, metal::false_type{}, bf[j],
                                 bf[j + 1], metal::false_type{});
        }
    }
}
