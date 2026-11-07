// NAX helpers include Apple MIT-licensed code. See THIRD_PARTY_LICENSES.txt.
// MFA_REQUIRE_MSL4
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// Apple steel/* helpers (verbatim from ~/code/mlx-source).

// === defines.h ===
#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

// === utils/type_traits.h (subset) ===
#pragma METAL internals : enable
namespace metal {
template <typename T> struct is_empty : metal::bool_constant<__is_empty(T)> {};
template <typename T> struct pointer_element {};
template <typename T> struct pointer_element<thread T*> { using type = remove_cv_t<T>; };
template <typename T> struct pointer_element<device T*> { using type = remove_cv_t<T>; };
template <typename T> struct pointer_element<constant T*> { using type = remove_cv_t<T>; };
template <typename T> struct pointer_element<threadgroup T*> { using type = remove_cv_t<T>; };
template <typename T> using pointer_element_t = typename pointer_element<remove_cv_t<T>>::type;
}
#pragma METAL internals : disable

// === utils/integral_constant.h (subset) ===
#pragma METAL internals : enable
namespace mlx { namespace steel {
template <typename T, T v> struct integral_constant {
  static constexpr constant T value = v;
  using value_type = T;
  using type = integral_constant;
  METAL_FUNC constexpr operator value_type() const thread noexcept { return value; }
};
template <bool B> using bool_constant = integral_constant<bool, B>;
using true_type = bool_constant<true>;
using false_type = bool_constant<false>;
template <int val> using Int = integral_constant<int, val>;
#define integral_const_binop(__op__, __operator__)          \
  template <typename T, T tv, typename U, U uv>             \
  METAL_FUNC constexpr auto __operator__(                   \
      integral_constant<T, tv>, integral_constant<U, uv>) { \
    constexpr auto res = tv __op__ uv;                      \
    return integral_constant<decltype(tv __op__ uv), res>{}; \
  }
integral_const_binop(+, operator+);
integral_const_binop(-, operator-);
integral_const_binop(*, operator*);
integral_const_binop(/, operator/);
template <int start, int stop, int step, typename F>
constexpr void const_for_loop(F f) {
  if constexpr (start < stop) {
    constexpr auto idx = Int<start>{};
    f(idx);
    const_for_loop<start + step, stop, step, F>(f);
  }
}
#undef integral_const_binop
}}
#pragma METAL internals : disable

// === ShellsmithLimits<float/half/bfloat> (Apple kernels/utils.h:55-70) ===
template <typename U> struct ShellsmithLimits {
  static const constant U max = metal::numeric_limits<U>::max();
  static const constant U min = metal::numeric_limits<U>::min();
  static const constant U finite_max = metal::numeric_limits<U>::max();
  static const constant U finite_min = metal::numeric_limits<U>::min();
};
template <> struct ShellsmithLimits<float> {
  static constexpr constant float max = metal::numeric_limits<float>::infinity();
  static constexpr constant float min = -metal::numeric_limits<float>::infinity();
  static constexpr constant float finite_max = metal::numeric_limits<float>::max();
  static constexpr constant float finite_min = -metal::numeric_limits<float>::max();
};

// === Apple steel/attn/nax.h — BaseNAXFrag + NAXTile (verbatim, nax.h:27-817) ===
namespace mlx { namespace steel {

struct BaseNAXFrag {
  STEEL_CONST short kFragRows = 16;
  STEEL_CONST short kFragCols = 16;
  STEEL_CONST short kElemsPerFrag = (kFragRows * kFragCols) / 32;
  STEEL_CONST short kElemRows = 2;
  STEEL_CONST short kElemCols = 4;
  STEEL_CONST short kElemRowsJump = 8;

  template <typename U>
  using dtype_frag_t = typename metal::vec<U, kElemsPerFrag>;

  METAL_FUNC static short2 get_coord() {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4;
    return short2{fn, fm};
  }

  template <typename T, typename SrcPtrType, typename StrX, typename StrY,
            typename OffX = Int<0>, typename OffY = Int<0>>
  METAL_FUNC static constexpr void load(
      thread dtype_frag_t<T>& dst, SrcPtrType src,
      StrX str_x, StrY str_y, OffX off_x = {}, OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + c + j]);
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + (c + j) * str_y]);
        }
      }
    }
  }

  template <typename T, typename SrcPtrType, typename StrX, typename StrY,
            typename LimX, typename OffX = Int<0>, typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_rows(
      thread dtype_frag_t<T>& dst, SrcPtrType src,
      StrX str_x, StrY str_y, LimX lim_x, OffX off_x = {}, OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if (r < lx) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + (c + j)]);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + (c + j) * str_y]);
          }
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <typename T, typename SrcPtrType, typename StrX, typename StrY,
            typename LimX, typename LimY, typename OffX = Int<0>, typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_safe(
      thread dtype_frag_t<T>& dst, SrcPtrType src,
      StrX str_x, StrY str_y, LimX lim_x, LimY lim_y,
      OffX off_x = {}, OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    auto ly = lim_y - sc.x;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if (r < lx && (c + j) < ly) {
          dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + (c + j) * str_y]);
        } else {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <typename T, typename DstPtrType, typename StrX, typename StrY,
            typename OffX = Int<0>, typename OffY = Int<0>>
  METAL_FUNC static constexpr void store(
      const thread dtype_frag_t<T>& src, DstPtrType dst,
      StrX str_x, StrY str_y, OffX off_x = {}, OffY off_y = {}) {
    using U = metal::pointer_element_t<DstPtrType>;
    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + (c + j) * str_y] = static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <typename T, typename DstPtrType, typename StrX, typename StrY,
            typename LimX, typename OffX = Int<0>, typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_rows(
      const thread dtype_frag_t<T>& src, DstPtrType dst,
      StrX str_x, StrY str_y, LimX lim_x, OffX off_x = {}, OffY off_y = {}) {
    using U = metal::pointer_element_t<DstPtrType>;
    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      if (r < lx) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[r * str_x + (c + j) * str_y] = static_cast<U>(src[i * kElemCols + j]);
          }
        }
      }
    }
  }

  template <typename T, typename DstPtrType, typename StrX, typename StrY,
            typename LimX, typename LimY, typename OffX = Int<0>, typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_safe(
      const thread dtype_frag_t<T>& src, DstPtrType dst,
      StrX str_x, StrY str_y, LimX lim_x, LimY lim_y,
      OffX off_x = {}, OffY off_y = {}) {
    using U = metal::pointer_element_t<DstPtrType>;
    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    auto ly = lim_y - sc.x;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if (r < lx && (c + j) < ly) {
          dst[r * str_x + (c + j) * str_y] = static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_reduce(
      thread const dtype_frag_t<T>& inp_vals, thread T* reduced_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      T thr_reduce = Op::apply(
          Op::apply(inp_vals[i * kElemCols + 0], inp_vals[i * kElemCols + 1]),
          Op::apply(inp_vals[i * kElemCols + 2], inp_vals[i * kElemCols + 3]));
      T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
      qgr_reduce = Op::apply(thr_reduce, qgr_reduce);
      T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
      sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);
      reduced_vals[i] = Op::apply(reduced_vals[i], sgr_reduce);
    }
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_bin_op(
      thread dtype_frag_t<T>& inp_vals, thread T* row_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        inp_vals[i * kElemCols + j] = Op::apply(inp_vals[i * kElemCols + j], row_vals[i]);
      }
    }
  }

  template <typename CType, typename AType, typename BType,
            bool transpose_a = false, bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread dtype_frag_t<CType>& Cn0, thread dtype_frag_t<CType>& Cn1,
      const thread dtype_frag_t<AType>& A, metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& Bn0, const thread dtype_frag_t<BType>& Bn1,
      metal::bool_constant<transpose_b>) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, transpose_a, transpose_b, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;
    // prvalue decltype: never address-space qualified (MSL 4.1 puts `thread` on
    // decltype(local)); avoids metal::remove_addrspace_t, absent from pre-0.32.1 MLX
    // headers and unverified on older OS Metal compilers (dense V6 compiles at runtime).
    using ct_a_t = decltype(gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, CType>());
    using ct_b_t = decltype(gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, CType>());
    auto ct_a = gemm_op.template get_left_input_cooperative_tensor<bfloat, bfloat, CType>();
    auto ct_b = gemm_op.template get_right_input_cooperative_tensor<bfloat, bfloat, CType>();
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<ct_a_t, ct_b_t, CType>();
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) ct_a[i] = bfloat(A[i]);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_b[i] = bfloat(Bn0[i]);
      ct_b[kElemsPerFrag + i] = bfloat(Bn1[i]);
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_c[i] = Cn0[i];
      ct_c[kElemsPerFrag + i] = Cn1[i];
    }
    gemm_op.run(ct_a, ct_b, ct_c);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      Cn0[i] = ct_c[i];
      Cn1[i] = ct_c[kElemsPerFrag + i];
    }
  }
};

template <typename T, short kTileRows_, short kTileCols_, class NAXFrag_ = BaseNAXFrag>
struct NAXTile {
  using NAXFrag_t = NAXFrag_;
  using elem_type = T;
  STEEL_CONST short kFragRows = NAXFrag_t::kFragRows;
  STEEL_CONST short kFragCols = NAXFrag_t::kFragCols;
  STEEL_CONST short kElemsPerFrag = NAXFrag_t::kElemsPerFrag;
  STEEL_CONST short kTileRows = kTileRows_;
  STEEL_CONST short kTileCols = kTileCols_;
  STEEL_CONST short kRows = kTileRows * kFragRows;
  STEEL_CONST short kCols = kTileCols * kFragCols;
  STEEL_CONST short kNumFrags = kTileRows * kTileCols;
  STEEL_CONST short kElemsPerTile = kNumFrags * kElemsPerFrag;
  STEEL_CONST short kFragThrRows = NAXFrag_t::kElemRows;
  STEEL_CONST short kFragThrCols = NAXFrag_t::kElemCols;
  STEEL_CONST short kFragRowsJump = NAXFrag_t::kElemRowsJump;
  STEEL_CONST short kRowsPerThread = kTileRows * NAXFrag_t::kElemRows;
  STEEL_CONST short kColsPerThread = kTileCols * NAXFrag_t::kElemCols;

  typedef typename NAXFrag_t::template dtype_frag_t<T> frag_type;
  frag_type val_frags[kNumFrags];

  METAL_FUNC NAXTile() thread {}

  METAL_FUNC constexpr void clear() thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) val_frags[i] = frag_type(0);
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j) thread {
    return val_frags[i * kTileCols + j];
  }
  METAL_FUNC constexpr const thread frag_type& frag_at(const short i, const short j) const thread {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC thread elem_type* elems() thread {
    return reinterpret_cast<thread elem_type*>(val_frags);
  }

  template <typename Op>
  METAL_FUNC void row_reduce(thread metal::vec<T, kRowsPerThread>& vals) const thread {
    auto vptr = (thread T*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        NAXFrag_t::template row_reduce<Op>(frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(thread metal::vec<T, kRowsPerThread>& vals) thread {
    auto vptr = (thread T*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        NAXFrag_t::template row_bin_op<Op>(frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename SrcPtrType>
  METAL_FUNC void load(SrcPtrType src, const int ld) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load(frag_at(idx_row.value, idx_col.value), src, ld, Int<1>{},
                        idx_row * Int<kFragRows>{}, idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store(device U* dst, const int ld) const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store(frag_at(idx_row.value, idx_col.value), dst, ld, Int<1>{},
                         idx_row * Int<kFragRows>{}, idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename SrcPtrType>
  METAL_FUNC void load_rows(SrcPtrType src, const int ld, const short n_rows) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load_rows(frag_at(idx_row.value, idx_col.value), src, ld, Int<1>{},
                             n_rows, idx_row * Int<kFragRows>{}, idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename SrcPtrType>
  METAL_FUNC void load_safe(SrcPtrType src, const int ld, const short2 src_tile_dims) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load_safe(frag_at(idx_row.value, idx_col.value), src, ld, Int<1>{},
                             src_tile_dims.y, src_tile_dims.x,
                             idx_row * Int<kFragRows>{}, idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store_rows(device U* dst, const int ld, const short n_rows) const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store_rows(frag_at(idx_row.value, idx_col.value), dst, ld, Int<1>{},
                              n_rows, idx_row * Int<kFragRows>{}, idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store_safe(device U* dst, const int ld, const short2 dst_tile_dims) const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store_safe(frag_at(idx_row.value, idx_col.value), dst, ld, Int<1>{},
                              dst_tile_dims.y, dst_tile_dims.x,
                              idx_row * Int<kFragRows>{}, idx_col * Int<kFragCols>{});
      });
    });
  }
};

}}  // namespace mlx::steel

// === Operator structs (steel_attention_nax.h:31-71) ===
struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return metal::max(x, y); }
};
struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return x + y; }
};
struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return x * y; }
};
struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return fast::exp2(x - y); }
};


using T=bfloat;
using namespace mlx::steel;
// Purpose: Reduce head-dimension partial contractions without device atomics.
// Input: Register tiles and one shared slot per SIMD group. Output: Complete tiles.
METAL_FUNC void shell_exchange(thread float* values, short count,
        threadgroup float* scratch, short group, short lane) {
    short base=lane*count;
    STEEL_PRAGMA_UNROLL
    for(short i=0;i<count;++i) scratch[group*32*count+base+i]=values[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for(short i=0;i<count;++i) {
        values[i]=scratch[base+i]+scratch[32*count+base+i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// Purpose: Contract a 16-by-16 BF16 tile with float register accumulation.
// Input: Distributed fragments. Output: One updated result fragment.
template<bool TA,bool TB,typename AType,typename BType>
METAL_FUNC void shell_mma_one(thread metal::vec<float,8>& result,
        const thread metal::vec<AType,8>& left,
        const thread metal::vec<BType,8>& right) {
    metal::vec<float,8> ignored(0);
    metal::vec<bfloat,8> zero(bfloat(0));
    BaseNAXFrag::mma(result,ignored,left,metal::bool_constant<TA>{},right,zero,metal::bool_constant<TB>{});
}
