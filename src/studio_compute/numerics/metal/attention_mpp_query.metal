    uint3 gid=threadgroup_position_in_grid;
    uint localidx=thread_index_in_threadgroup;
    ShellParams p=*reinterpret_cast<const device ShellParams*>(parameters);
    int qr=gid.x*M,head=gid.y,kvhead=head/p.gqa_factor;
    const device T* qptr=input_q+gid.z*p.Q_strides[0]+head*p.Q_strides[1];
    const device T* kptr=input_k+gid.z*p.K_strides[0]+kvhead*p.K_strides[1];
    const device T* vptr=input_v+gid.z*p.V_strides[0]+kvhead*p.V_strides[1];
    const device T* doptr=input_dy+gid.z*p.Q_strides[0]+head*p.Q_strides[1];
    const device T* lp=input_statistics+gid.z*p.L_strides[0]+head*p.L_strides[1];
    const device T* dp=input_delta+gid.z*p.L_strides[0]+head*p.L_strides[1];
    auto Q=tensor<device T,dextents<int32_t,2>,tensor_inline>(const_cast<device T*>(qptr),dextents<int32_t,2>(256,p.qL));
    auto K=tensor<device T,dextents<int32_t,2>,tensor_inline>(const_cast<device T*>(kptr),dextents<int32_t,2>(256,p.kL));
    auto V=tensor<device T,dextents<int32_t,2>,tensor_inline>(const_cast<device T*>(vptr),dextents<int32_t,2>(256,p.kL));
    auto dO=tensor<device T,dextents<int32_t,2>,tensor_inline>(const_cast<device T*>(doptr),dextents<int32_t,2>(256,p.qL));
    auto mq=Q.slice(0,qr),mo=dO.slice(0,qr);
    constexpr auto score_desc=matmul2d_descriptor(M,N,256,false,true,true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<score_desc,execution_simdgroups<SG>> score_op;
    constexpr auto out_desc=matmul2d_descriptor(M,256,N,false,false,true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<out_desc,execution_simdgroups<SG>> out_op;
    threadgroup T ds_data[M*N];
    auto DS=tensor<threadgroup T,dextents<int32_t,2>,tensor_inline>(ds_data,dextents<int32_t,2>(N,M));
    using MK=decltype(K.slice(0,0));using MD=decltype(tensor<threadgroup T,dextents<int32_t,2>,tensor_inline>(ds_data,dextents<int32_t,2>(N,M)));
    auto result=out_op.template get_destination_cooperative_tensor<MD,MK,float>();
    #pragma clang loop unroll(full)
    for(ushort i=0;i<result.get_capacity();++i) if(result.is_valid_element(i))result[i]=0;
    int lim=min(p.kL,qr+M+p.qL_off);
    for(int kr=0;kr<lim;kr+=N) {
        auto mk=K.slice(0,kr),mv=V.slice(0,kr);
        using MQ=decltype(Q.slice(0,0));
        auto score=score_op.template get_destination_cooperative_tensor<MQ,MK,float>();
        auto derivative=score_op.template get_destination_cooperative_tensor<MQ,MK,float>();
        #pragma clang loop unroll(full)
        for(ushort i=0;i<score.get_capacity();++i) if(score.is_valid_element(i)) {score[i]=0;derivative[i]=0;}
        for(int i=localidx;i<M*N;i+=SG*32)ds_data[i]=T(0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        score_op.run(mq,mk,score);
        score_op.run(mo,mv,derivative);
        #pragma clang loop unroll(full)
        for(ushort i=0;i<score.get_capacity();++i) {
            if(score.is_valid_element(i)) {
                auto index=score.get_multidimensional_index(i);
                int row=qr+index[1],col=kr+index[0];
                bool valid=row<p.qL && col<p.kL && col<=row+p.qL_off;
                float prob=valid ? fast::exp2(score[i]*0.09016844005556f-float(lp[row*2])-float(lp[row*2+1])) : 0;
                float delta=valid ? float(dp[row*2])+float(dp[row*2+1]) : 0;
                ds_data[index[1]*N+index[0]]=T(prob*(derivative[i]-delta));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        out_op.run(DS,mk,result);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device T* target=output+gid.z*p.Q_strides[0]+head*p.Q_strides[1];
    #pragma clang loop unroll(full)
    for(ushort i=0;i<result.get_capacity();++i)if(result.is_valid_element(i)) {
        auto index=result.get_multidimensional_index(i);
        if(qr+index[1]<p.qL)target[(qr+index[1])*256+index[0]]=T(result[i]*0.0625f);
    }
