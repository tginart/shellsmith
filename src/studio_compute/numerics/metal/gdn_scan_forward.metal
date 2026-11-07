    uint lane=thread_index_in_simdgroup;
    uint dg=simdgroup_index_in_threadgroup;
    uint3 tid=threadgroup_position_in_grid;
    constexpr short TD=DK/32,WN=2,TR=C/16;
    int bh=tid.z, vb=tid.x*16;
    int chunks=q_shape[2];
    using State=NAXTile<T,1,TD>;
    using Rows=NAXTile<float,TR,1>;
    using Local=NAXTile<T,TR,1>;
    State state;
    state.load(initial+(bh*DV+vb)*DK+dg*(DK/2),DK);
    threadgroup float exchange[WN*32*Rows::kElemsPerTile];
    short2 coord=BaseNAXFrag::get_coord();
    for(int c=0;c<chunks;++c) {
        int block=bh*chunks+c;
        state.store(history+(block*DV+vb)*DK+dg*(DK/2),DK);
        Rows y;y.clear();
        STEEL_PRAGMA_UNROLL
        for(short row=0;row<TR;++row) {
            STEEL_PRAGMA_UNROLL
            for(short d=0;d<TD;++d) {
                NAXTile<T,1,1> qt;
                qt.load(q+(block*C+row*16)*DK+dg*(DK/2)+d*16,DK);
                shell_mma_one<false,true>(y.frag_at(row,0),qt.frag_at(0,0),state.frag_at(0,d));
            }
        }
        shell_exchange(y.elems(),Rows::kElemsPerTile,exchange,dg,lane);
        STEEL_PRAGMA_UNROLL
        for(short row=0;row<TR;++row) {
            NAXTile<T,1,1> base;
            base.load(mu0+(block*C+row*16)*DV+vb,DV);
            STEEL_PRAGMA_UNROLL
            for(short i=0;i<8;++i) base.elems()[i]=T(float(base.elems()[i])+float(T(y.frag_at(row,0)[i])));
            if(dg==0)base.store(output+(block*C+row*16)*DV+vb,DV);
        }
        Rows memory;memory.clear();
        STEEL_PRAGMA_UNROLL
        for(short row=0;row<TR;++row) {
            STEEL_PRAGMA_UNROLL
            for(short d=0;d<TD;++d) {
                NAXTile<T,1,1> qt;
                qt.load(kt+(block*C+row*16)*DK+dg*(DK/2)+d*16,DK);
                shell_mma_one<false,true>(memory.frag_at(row,0),qt.frag_at(0,0),state.frag_at(0,d));
            }
        }
        shell_exchange(memory.elems(),Rows::kElemsPerTile,exchange,dg,lane);
        Local weighted;
        STEEL_PRAGMA_UNROLL
        for(short row=0;row<TR;++row) {
            NAXTile<T,1,1> base;
            base.load(u0+(block*C+row*16)*DV+vb,DV);
            STEEL_PRAGMA_UNROLL
            for(short i=0;i<2;++i) {
                T r=ratio[block*C+row*16+coord.y+i*8];
                STEEL_PRAGMA_UNROLL
                for(short j=0;j<4;++j) {
                    int index=i*4+j;
                    T u=T(float(base.elems()[index])-float(T(memory.frag_at(row,0)[index])));
                    weighted.frag_at(row,0)[index]=T(float(r)*float(u));
                }
            }
        }
        NAXTile<float,1,TD> delta;delta.clear();
        STEEL_PRAGMA_UNROLL
        for(short d=0;d<TD;d+=2) {
            STEEL_PRAGMA_UNROLL
            for(short row=0;row<TR;++row) {
                NAXTile<T,1,2> keys;
                keys.load(k+(block*C+row*16)*DK+dg*(DK/2)+d*16,DK);
                BaseNAXFrag::mma(delta.frag_at(0,d),delta.frag_at(0,d+1),weighted.frag_at(row,0),
                    metal::true_type{},keys.frag_at(0,0),keys.frag_at(0,1),metal::false_type{});
            }
        }
        STEEL_PRAGMA_UNROLL
        for(short i=0;i<State::kElemsPerTile;++i) {
            T decayed=T(float(decay[block])*float(state.elems()[i]));
            state.elems()[i]=T(float(decayed)+float(T(delta.elems()[i])));
        }
    }
    state.store(final_state+(bh*DV+vb)*DK+dg*(DK/2),DK);
