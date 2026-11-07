    uint lane=thread_index_in_simdgroup;
    uint dg=simdgroup_index_in_threadgroup;
    uint3 tid=threadgroup_position_in_grid;
    constexpr short TD=DK/32,WN=2,TR=C/16;
    int bh=tid.z,vb=tid.x*16;
    int chunks=q_shape[2];
    using State=NAXTile<T,1,TD>;
    using Rows=NAXTile<float,TR,1>;
    using Local=NAXTile<T,TR,1>;
    State adjoint;
    adjoint.load(dfinal+(bh*DV+vb)*DK+dg*(DK/2),DK);
    threadgroup float exchange[WN*32*Rows::kElemsPerTile];
    short2 coord=BaseNAXFrag::get_coord();
    for(int c=chunks-1;c>=0;--c) {
        int block=bh*chunks+c;
        adjoint.store(ds+(block*DV+vb)*DK+dg*(DK/2),DK);
        Rows dw_tile;dw_tile.clear();
        STEEL_PRAGMA_UNROLL
        for(short row=0;row<TR;++row) {
            STEEL_PRAGMA_UNROLL
            for(short d=0;d<TD;++d) {
                NAXTile<T,1,1> keys;
                keys.load(k+(block*C+row*16)*DK+dg*(DK/2)+d*16,DK);
                shell_mma_one<false,true>(dw_tile.frag_at(row,0),keys.frag_at(0,0),adjoint.frag_at(0,d));
            }
        }
        shell_exchange(dw_tile.elems(),Rows::kElemsPerTile,exchange,dg,lane);
        Local local_du;
        STEEL_PRAGMA_UNROLL
        for(short row=0;row<TR;++row) {
            NAXTile<T,1,1> weighted;
            STEEL_PRAGMA_UNROLL
            for(short i=0;i<2;++i) {
                T r=ratio[block*C+row*16+coord.y+i*8];
                STEEL_PRAGMA_UNROLL
                for(short j=0;j<4;++j) {
                    int index=i*4+j;
                    weighted.elems()[index]=T(dw_tile.frag_at(row,0)[index]);
                    local_du.frag_at(row,0)[index]=T(float(r)*float(weighted.elems()[index]));
                }
            }
            if(dg==0) {
                weighted.store(dw+(block*C+row*16)*DV+vb,DV);
                NAXTile<T,1,1> part;
                part.frag_at(0,0)=local_du.frag_at(row,0);
                part.store(du+(block*C+row*16)*DV+vb,DV);
            }
        }
        State before;
        before.load(history+(block*DV+vb)*DK+dg*(DK/2),DK);
        Rows memory;memory.clear();
        STEEL_PRAGMA_UNROLL
        for(short row=0;row<TR;++row) {
            STEEL_PRAGMA_UNROLL
            for(short d=0;d<TD;++d) {
                NAXTile<T,1,1> keys;
                keys.load(kt+(block*C+row*16)*DK+dg*(DK/2)+d*16,DK);
                shell_mma_one<false,true>(memory.frag_at(row,0),keys.frag_at(0,0),before.frag_at(0,d));
            }
        }
        shell_exchange(memory.elems(),Rows::kElemsPerTile,exchange,dg,lane);
        if(dg==0) {
            STEEL_PRAGMA_UNROLL
            for(short row=0;row<TR;++row) {
                NAXTile<T,1,1> base;
                base.load(u0+(block*C+row*16)*DV+vb,DV);
                STEEL_PRAGMA_UNROLL
                for(short i=0;i<8;++i) base.elems()[i]=T(float(base.elems()[i])-float(T(memory.frag_at(row,0)[i])));
                base.store(uc+(block*C+row*16)*DV+vb,DV);
            }
        }
        NAXTile<float,1,TD> hidden,remembered;hidden.clear();remembered.clear();
        STEEL_PRAGMA_UNROLL
        for(short d=0;d<TD;d+=2) {
            STEEL_PRAGMA_UNROLL
            for(short row=0;row<TR;++row) {
                NAXTile<T,1,1> gy;
                NAXTile<T,1,2> qtile,ktile;
                gy.load(dy+(block*C+row*16)*DV+vb,DV);
                qtile.load(q+(block*C+row*16)*DK+dg*(DK/2)+d*16,DK);
                ktile.load(kt+(block*C+row*16)*DK+dg*(DK/2)+d*16,DK);
                BaseNAXFrag::mma(hidden.frag_at(0,d),hidden.frag_at(0,d+1),gy.frag_at(0,0),
                    metal::true_type{},qtile.frag_at(0,0),qtile.frag_at(0,1),metal::false_type{});
                BaseNAXFrag::mma(remembered.frag_at(0,d),remembered.frag_at(0,d+1),local_du.frag_at(row,0),
                    metal::true_type{},ktile.frag_at(0,0),ktile.frag_at(0,1),metal::false_type{});
            }
        }
        STEEL_PRAGMA_UNROLL
        for(short i=0;i<State::kElemsPerTile;++i) {
            T local=T(float(T(hidden.elems()[i]))-float(T(remembered.elems()[i])));
            T decayed=T(float(decay[block])*float(adjoint.elems()[i]));
            adjoint.elems()[i]=T(float(decayed)+float(local));
        }
    }
    adjoint.store(dinitial+(bh*DV+vb)*DK+dg*(DK/2),DK);
