// Purpose: Shared helpers of the fused GDN chunk kernels.
// Note: The header follows gdn_nax_common.metal. One threadgroup owns one
//     (batch, value head, chunk) and has four simdgroups. Simdgroup sg owns
//     the 16-row band 16 sg .. 16 sg + 15 of the 64-row chunk. Element e of a
//     16x16 fragment is row fm + 8 (e / 4) and column fn + e % 4, with
//     (fn, fm) = BaseNAXFrag::get_coord().

constant constexpr int CHUNK = 64;
constant constexpr int WIDTH = 128;
constant constexpr int FRAGS = WIDTH / 16;
// The chunk path clamps each gate at 1e-6 before its log.
constant constexpr float LOG_GATE_MIN = -13.815510557964274f;

typedef metal::vec<float, 8> ffrag;
typedef metal::vec<bfloat, 8> bfrag;

inline ffrag zero_ffrag() {
    return ffrag(0.0f);
}

inline int frag_row(short2 sc, short e) {
    return sc.y + 8 * (e / 4);
}

inline int frag_col(short2 sc, short e) {
    return sc.x + e % 4;
}

// Sum one value over the four lanes that hold one fragment row.
inline float row_sum4(float value) {
    value += simd_shuffle_xor(value, ushort(1));
    return value + simd_shuffle_xor(value, ushort(8));
}

// Add the row sums of one band fragment to target[16 sg + row].
inline void add_band_row_sums(const thread ffrag& frag, threadgroup float* target,
                              short sg, short2 sc) {
    float sums[2] = {0.0f, 0.0f};
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < 8; ++e) sums[e / 4] += frag[e];
    STEEL_PRAGMA_UNROLL
    for (short r = 0; r < 2; ++r) {
        const float total = row_sum4(sums[r]);
        if (sc.x == 0) target[16 * sg + sc.y + 8 * r] += total;
    }
}

// The log gate of one row and its derivative by the logit a.
struct Gate {
    float log_gate;
    float slope;
};

inline float round_bf16(float x) {
    return float(bfloat(x));
}

// The gate math of the BF16 sampler (bf16.gdn_inner), with each step rounded to
// BF16 as MLX does, so replay sees the gates that the sampler saw. Only the log
// and the prefix sums are FP32. The slope ignores the rounding.
inline Gate sampler_gate(bfloat a, float a_log, float dt_bias) {
    const float scale = round_bf16(metal::exp(round_bf16(a_log)));
    const float x = round_bf16(float(a) + round_bf16(dt_bias));
    const float softplus = round_bf16(
        metal::max(x, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(x))));
    const float gate = round_bf16(metal::exp(round_bf16(-scale * softplus)));
    const bool clamped = gate < 1e-6f;
    const float sigmoid = 1.0f / (1.0f + metal::exp(-x));
    return {clamped ? LOG_GATE_MIN : metal::log(gate), clamped ? 0.0f : -scale * sigmoid};
}

// The BF16 sigmoid of a beta logit, as the sampler makes it.
inline float sampler_beta(bfloat b) {
    return round_bf16(1.0f / (1.0f + metal::exp(-float(b))));
}

// Rows j of the chunk: the log gate prefix sums lg, beta, and exp(lg).
// a and b are the gate and beta logits of the chunk rows, with row stride hv.
inline void load_gates(const device bfloat* a, const device bfloat* b, int hv,
                       float a_log, float dt_bias, threadgroup float* lg,
                       threadgroup float* beta, threadgroup float* decay, ushort tid) {
    if (tid < CHUNK) {
        lg[tid] = sampler_gate(a[tid * hv], a_log, dt_bias).log_gate;
        beta[tid] = sampler_beta(b[tid * hv]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float prefix = 0.0f;
    if (tid < CHUNK)
        for (int t = 0; t <= int(tid); ++t) prefix += lg[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < CHUNK) {
        lg[tid] = prefix;
        decay[tid] = metal::exp(prefix);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Copy the 64 x 128 k rows of a chunk to threadgroup memory, and to `copy`
// when it is not null. All 128 threads take part.
inline void stage_keys(threadgroup bfloat* staged, const device bfloat* k, int stride,
                       device bfloat* copy, ushort tid) {
    for (int index = tid; index < CHUNK * WIDTH / 4; index += 128) {
        const int row = index / (WIDTH / 4);
        const int column = 4 * (index % (WIDTH / 4));
        const vec<bfloat, 4> value = *reinterpret_cast<const device vec<bfloat, 4>*>(
            k + row * stride + column);
        *reinterpret_cast<threadgroup vec<bfloat, 4>*>(staged + row * WIDTH + column) = value;
        if (copy) *reinterpret_cast<device vec<bfloat, 4>*>(copy + row * WIDTH + column) = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Band fragments of k k^T and q k^T (16 rows by 64 columns), FP32. The keys
// come from threadgroup memory, the queries of the band from the device.
inline void chunk_scores(thread ffrag* kk, thread ffrag* qk, const device bfloat* q,
                         int stride, threadgroup const bfloat* keys, short sg) {
    STEEL_PRAGMA_UNROLL
    for (short f = 0; f < 4; ++f) {
        kk[f] = zero_ffrag();
        qk[f] = zero_ffrag();
    }
    STEEL_PRAGMA_UNROLL
    for (short ks = 0; ks < FRAGS; ++ks) {
        bfrag ka, qa;
        BaseNAXFrag::load(ka, keys + 16 * sg * WIDTH + 16 * ks, WIDTH, Int<1>{});
        BaseNAXFrag::load(qa, q + 16 * sg * stride + 16 * ks, stride, Int<1>{});
        STEEL_PRAGMA_UNROLL
        for (short cb = 0; cb < 4; cb += 2) {
            bfrag k0, k1;
            BaseNAXFrag::load(k0, keys + 16 * cb * WIDTH + 16 * ks, WIDTH, Int<1>{});
            BaseNAXFrag::load(k1, keys + 16 * (cb + 1) * WIDTH + 16 * ks, WIDTH, Int<1>{});
            BaseNAXFrag::mma(kk[cb], kk[cb + 1], ka, metal::false_type{}, k0, k1,
                             metal::true_type{});
            BaseNAXFrag::mma(qk[cb], qk[cb + 1], qa, metal::false_type{}, k0, k1,
                             metal::true_type{});
        }
    }
}

// Invert the unit lower matrix I + L in place. lower holds the strict part L
// and zeros elsewhere; it then holds the inverse X. Thread j owns column j.
inline void invert_unit_lower(threadgroup float* lower, ushort tid) {
    if (tid < CHUNK) lower[tid] = tid == 0 ? 1.0f : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 1; i < CHUNK; ++i) {
        float value = 0.0f;
        if (tid < CHUNK) {
            float4 sums = 0.0f;
            int k = 0;
            for (; k + 4 <= i; k += 4) {
                sums += float4(lower[i * CHUNK + k], lower[i * CHUNK + k + 1],
                               lower[i * CHUNK + k + 2], lower[i * CHUNK + k + 3])
                    * float4(lower[k * CHUNK + tid], lower[(k + 1) * CHUNK + tid],
                             lower[(k + 2) * CHUNK + tid], lower[(k + 3) * CHUNK + tid]);
            }
            value = (int(tid) == i ? 1.0f : 0.0f) - ((sums.x + sums.y) + (sums.z + sums.w));
            for (; k < i; ++k) value -= lower[i * CHUNK + k] * lower[k * CHUNK + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < CHUNK) lower[i * CHUNK + tid] = value;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
