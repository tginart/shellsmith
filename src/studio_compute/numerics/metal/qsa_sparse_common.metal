// Purpose: Shared helpers for the sparse QSA attention kernels.
// Note: The header follows gdn_nax_common.metal, which defines the 16x16 NAX
//     fragments. Element e of a fragment is row fm + 8 * (e / 4) and column
//     fn + e % 4, with (fn, fm) = BaseNAXFrag::get_coord(). Each of the four
//     simdgroups owns one 64-wide slice of the head width D.

constant constexpr int D = 256;
constant constexpr float LOG2E = 1.4426950408889634f;
constant constexpr float NEG_INF = -metal::numeric_limits<float>::infinity();
constant constexpr float FINITE_MIN = -metal::numeric_limits<float>::max();

typedef metal::vec<float, 8> ffrag;
typedef metal::vec<bfloat, 8> bfrag;

inline ffrag zero_ffrag() {
    return ffrag(0.0f);
}

// Load a 16-row BF16 fragment whose row r is the device row tokens[r].
// A negative token gives a zero row.
inline void load_gathered(
    thread bfrag& dst, const device bfloat* base, threadgroup const int* tokens,
    int column, short2 sc) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < 2; ++i) {
        int token = tokens[sc.y + 8 * i];
        const device bfloat* row = base + token * D + column + sc.x;
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 4; ++j) dst[i * 4 + j] = token >= 0 ? row[j] : bfloat(0);
    }
}

// Sum four simdgroup partials of `count` fragments in a fixed order.
// Element e of lane l is word (f * 8 + e) * 32 + l, so the 32 lanes of one
// access hit 32 different banks.
inline void exchange_four(
    thread ffrag* values, short count, threadgroup float* scratch, short group,
    ushort lane) {
    const int stride = 32 * 8 * count;
    STEEL_PRAGMA_UNROLL
    for (short f = 0; f < count; ++f)
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 8; ++e)
            scratch[group * stride + (f * 8 + e) * 32 + lane] = values[f][e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short f = 0; f < count; ++f)
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 8; ++e) {
            int index = (f * 8 + e) * 32 + lane;
            values[f][e] = (scratch[index] + scratch[stride + index])
                + (scratch[2 * stride + index] + scratch[3 * stride + index]);
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Sum four simdgroup partials of four fragments in a fixed order. Simdgroup g
// sums fragment g, then every simdgroup reads the four sums. scratch holds
// 5 * 32 * 32 floats, with the bank-free layout of exchange_four.
inline void reduce_scatter_four(
    thread ffrag* values, threadgroup float* scratch, short group, ushort lane) {
    threadgroup float* sums = scratch + 4 * 32 * 32;
    STEEL_PRAGMA_UNROLL
    for (short f = 0; f < 4; ++f)
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 8; ++e)
            scratch[((group * 4 + f) * 8 + e) * 32 + lane] = values[f][e];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < 8; ++e) {
        const int index = (group * 8 + e) * 32 + lane;
        sums[index] = (scratch[index] + scratch[32 * 32 + index])
            + (scratch[2 * 32 * 32 + index] + scratch[3 * 32 * 32 + index]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short f = 0; f < 4; ++f)
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 8; ++e) values[f][e] = sums[(f * 8 + e) * 32 + lane];
}

// Reduce one value over the four lanes that hold one fragment row.
inline float row_max4(float value) {
    value = max(value, simd_shuffle_xor(value, ushort(1)));
    return max(value, simd_shuffle_xor(value, ushort(8)));
}

inline float row_sum4(float value) {
    value += simd_shuffle_xor(value, ushort(1));
    return value + simd_shuffle_xor(value, ushort(8));
}
