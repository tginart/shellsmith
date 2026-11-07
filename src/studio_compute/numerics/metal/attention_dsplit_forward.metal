// Copyright © 2024-26 Apple Inc. Adapted under the MIT license.
// See THIRD_PARTY_LICENSES.txt for the copyright and permission notices.

    uint simd_lane_id=thread_index_in_simdgroup;
    uint simd_group_id=simdgroup_index_in_threadgroup;
    uint3 tid=threadgroup_position_in_grid;
    uint3 lid=thread_position_in_threadgroup;
    const device T* Q=input_q; const device T* K=input_k; const device T* V=input_v;
    device T* O=output;
    ShellForwardParams params=*reinterpret_cast<const device ShellForwardParams*>(parameters);
    constexpr int BQ=64, BK=32, BD=256, WM=4, WN=2;
    using AccumType=float;
    constexpr bool align_Q=ALIGN_Q, align_K=ALIGN_K, do_causal=true;
  // Pacifying compiler
  (void)lid;

  // Move to correct block
  ulong3 tidl{tid.x, tid.y, tid.z};

  Q += tidl.z * params.Q_strides[0] + // Batch
      tidl.y * params.Q_strides[1] + // Head
      tidl.x * BQ * params.Q_strides[2]; // Sequence

  ulong kv_head_idx = int(tid.y) / params.gqa_factor;
  K += tidl.z * params.K_strides[0] + // Batch
      kv_head_idx * params.K_strides[1]; // Head

  V += tidl.z * params.V_strides[0] + // Batch
      kv_head_idx * params.V_strides[1]; // Head

  O += tidl.z * params.O_strides[0] + // Batch
      tidl.y * params.O_strides[1] + // Head
      tidl.x * BQ * params.O_strides[2]; // Sequence



  const metal::uniform<float> scale2 =
      make_uniform(0.0625f) * make_uniform(1.44269504089f);

  // Prepare MMA tiles
  constexpr short kU = 16;

  // The WM simdgroups along the first warp dimension split the Q sequence;
  // the WN simdgroups along the second split the head dim.
  static_assert(
      WN == 2 || WN == 4, "The head-dim split kernel needs WN == 2 or WN == 4");
  constexpr int kNWarps = WM;
  static_assert(
      BQ >= (kNWarps * kU) && BQ % (kNWarps * kU) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kU);
  // HeadDim frags over the full head dim
  constexpr int TD = BD / kU;
  // KV seq frags per warp
  constexpr short TK = BK / kU;

  static_assert(TQ == 1, "Check TQ");
  static_assert(TD % WN == 0, "The head dim must split evenly across WN");

  // HeadDim frags / columns owned by each of the WN simdgroups of a row group
  constexpr int TDh = TD / WN;
  constexpr int BDh = BD / WN;

  static_assert(TDh % 2 == 0, "P@V accumulates output fragments in pairs");
  static_assert(TK % 2 == 0, "S fragments are computed pair by pair");

  const short row_group = simd_group_id / WN;
  const short d_group = simd_group_id % WN;

  using otile_t = NAXTile<AccumType, TQ, TDh>;
  otile_t Otile;
  Otile.clear();

  const short tm = kU * TQ * row_group;
  Q += tm * int(params.Q_strides[2]) + d_group * BDh;
  K += d_group * BDh;
  V += d_group * BDh;
  O += tm * int(params.O_strides[2]) + d_group * BDh;

  constexpr short kRowsPT = otile_t::kRowsPerThread;

  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = ShellsmithLimits<AccumType>::finite_min;
  }



  int kb_lim = params.NK;
  int kb_min_causal = params.NK;

  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params.qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params.NK, kb_lim);

    int q_min = tid.x * BQ + params.qL_off;
    q_min = max(0, q_min);
    kb_min_causal = (q_min / BK);
  }

  const bool is_last_q = int(tid.x) == ((params.qL/BQ));
  const short lim_rows_q = max(0, (params.qL_rem ? params.qL_rem : BQ)-int(tm));
  const short lim_rows_k = params.kL_rem;

  using stile_t = NAXTile<AccumType, TQ, TK>;
  constexpr short kEPF = stile_t::NAXFrag_t::kElemsPerFrag;

  // One slot per (row group, D group) in per-lane-linear layout. All groups
  // share the fragment-to-lane mapping.
  threadgroup AccumType s_xchg[WM][WN][TK * kEPF * 32];

  // Keep the simdgroup's Q slice resident in registers for the whole KV
  // loop: TDh fragments of T are cheap next to the accumulators.
  NAXTile<T, 1, 1> Qtiles[TDh];
  STEEL_PRAGMA_UNROLL
  for (short id = 0; id < TDh; id++) {
    const int Q_load_off = id * kU;
    if (!align_Q && is_last_q) {
      Qtiles[id].load_rows(
          Q + Q_load_off, int(params.Q_strides[2]), lim_rows_q);
    } else {
      Qtiles[id].load(Q + Q_load_off, int(params.Q_strides[2]));
    }
  }

  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  // Loop over KV seq length
  for (int kb = 0; kb < kb_lim; kb++) {
    const int is_last_k = (kb == ((params.kL/BK)));

    stile_t Stile;
    Stile.clear();

    // S = Q @ K.T for this D slice.
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik += 2) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TDh; id++) {
        NAXTile<T, 2, 1> Ktile;
        const int K_load_off = ik * kU * int(params.K_strides[2]) + id * kU;

        if (!align_K && is_last_k) {
          Ktile.load_rows(
              K + K_load_off, int(params.K_strides[2]), lim_rows_k - ik * kU);
        } else {
          Ktile.load(K + K_load_off, int(params.K_strides[2]));
        }

        stile_t::NAXFrag_t::mma(
            Stile.frag_at(0, ik),
            Stile.frag_at(0, ik + 1),
            Qtiles[id].frag_at(0, 0),
            metal::false_type{},
            Ktile.frag_at(0, 0),
            Ktile.frag_at(1, 0),
            metal::true_type{});
      }
    }

    // Exchange all partial scores and reduce them in a fixed order.
    threadgroup AccumType* slot = s_xchg[row_group][d_group];
    const short base = short(simd_lane_id) * (TK * kEPF);
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik++) {
      thread auto& s = Stile.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kEPF; i++) {
        slot[base + ik * kEPF + i] = s[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for (short ik = 0; ik < TK; ik++) {
      thread auto& s = Stile.frag_at(0, ik);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < kEPF; i++) {
        s[i] = s_xchg[row_group][0][base + ik * kEPF + i];
      }
      STEEL_PRAGMA_UNROLL
      for (short peer_group = 1; peer_group < WN; peer_group++) {
        const threadgroup AccumType* peer = s_xchg[row_group][peer_group];
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kEPF; i++) {
          s[i] += peer[base + ik * kEPF + i];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scale S
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= float(scale2);
    }

    // Mask out length sequence
    if (!align_K && is_last_k) {
      constexpr auto neg_inf = ShellsmithLimits<AccumType>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        const short col_pos = ik * kU + sn;
        thread auto& fg = Stile.frag_at(0, ik);

        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const auto loc = ii * stile_t::kFragThrCols + jj;
            fg[loc] = ((col_pos + jj) < params.kL_rem) ? fg[loc] : neg_inf;
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal && kb >= kb_min_causal) {
      constexpr auto neg_inf = ShellsmithLimits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + params.qL_off + tm;
      const int base_col = kb * BK;

      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        thread auto& fg = Stile.frag_at(0, ik);

        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
            const auto r = base_row + ii * stile_t::kFragRowsJump + sm;
            const auto c = base_col + ik * kU + jj + sn;
            const auto loc = ii * stile_t::kFragThrCols + jj;
            fg[loc] = (r < c) ? neg_inf : fg[loc];
          }
        }
      }
    }

    // Other masking as needed


    // Do softmax in each D group; the row statistics are cheap.
    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i];
    }

    Stile.template row_reduce<SumOp>(sum_score);

    Otile.template row_bin_op<MulOp>(factor);

    simdgroup_barrier(mem_flags::mem_none);

    // O = P @ V for this D slice.
    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDh; id += 2) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        NAXTile<T, 1, 2> Vtile;

        const int V_load_off = ik * kU * int(params.V_strides[2]) + id * kU;

        if (!align_K && is_last_k) {
          Vtile.load_rows(
              V + V_load_off, int(params.V_strides[2]), lim_rows_k - ik * kU);
        } else {
          Vtile.load(V + V_load_off, int(params.V_strides[2]));
        }

        otile_t::NAXFrag_t::mma(
            Otile.frag_at(0, id),
            Otile.frag_at(0, id + 1),
            Stile.frag_at(0, ik),
            metal::false_type{},
            Vtile.frag_at(0, 0),
            Vtile.frag_at(0, 1),
            metal::false_type{});
      }
    }

    // Next block
    K += BK * int(params.K_strides[2]);
    V += BK * int(params.V_strides[2]);
  }


    if(d_group==0 && simd_coord.x==0) {
        device T* row_stats=statistics+tidl.z*params.L_strides[0]
            +tidl.y*params.L_strides[1]+(tidl.x*BQ+tm)*2;
        STEEL_PRAGMA_UNROLL
        for(short i=0;i<2;++i) {
            int row=simd_coord.y+i*8;
            if(tidl.x*BQ+tm+row < params.qL) {
                float log_normalizer=max_score[i]+fast::log2(sum_score[i]);
                T high=T(log_normalizer);
                row_stats[row*2]=high;
                row_stats[row*2+1]=T(log_normalizer-float(high));
            }
        }
    }

  // Normalize output
  threadgroup_barrier(mem_flags::mem_none);

  metal::vec<AccumType, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.f / sum_score[i];
  }

  Otile.template row_bin_op<MulOp>(rcp);

  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0)
      return;
    Otile.store_rows(O, int(params.O_strides[2]), lim_rows_q);
  } else {
    Otile.store(O, int(params.O_strides[2]));
  }
