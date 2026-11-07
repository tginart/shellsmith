    uint3 gid=threadgroup_position_in_grid;
    uint localidx=thread_index_in_threadgroup;
    ShellParams p=*reinterpret_cast<const device ShellParams*>(parameters);
    int kr=gid.x*M,kvhead=gid.y;
    const device T* kptr=input_k+gid.z*p.K_strides[0]+kvhead*p.K_strides[1];
    const device T* vptr=input_v+gid.z*p.V_strides[0]+kvhead*p.V_strides[1];
    auto K=tensor<device T,dextents<int32_t,2>,tensor_inline>(const_cast<device T*>(kptr),dextents<int32_t,2>(256,p.kL));
    auto V=tensor<device T,dextents<int32_t,2>,tensor_inline>(const_cast<device T*>(vptr),dextents<int32_t,2>(256,p.kL));
    auto mk=K.slice(0,kr),mv=V.slice(0,kr);
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
    auto result_value=out_op.template get_destination_cooperative_tensor<MD,MK,float>();
    #pragma clang loop unroll(full)
    for(ushort i=0;i<result.get_capacity();++i)if(result.is_valid_element(i)){result[i]=0;result_value[i]=0;}
    for(int group=0;group<p.gqa_factor;++group) {
        int head=kvhead*p.gqa_factor+group;
        const device T* qptr=input_q+gid.z*p.Q_strides[0]+head*p.Q_strides[1];
        const device T* doptr=input_dy+gid.z*p.Q_strides[0]+head*p.Q_strides[1];
        const device T* lp=input_statistics+gid.z*p.L_strides[0]+head*p.L_strides[1];
        const device T* dp=input_delta+gid.z*p.L_strides[0]+head*p.L_strides[1];
        auto Q=tensor<device T,dextents<int32_t,2>,tensor_inline>(const_cast<device T*>(qptr),dextents<int32_t,2>(256,p.qL));
        auto dO=tensor<device T,dextents<int32_t,2>,tensor_inline>(const_cast<device T*>(doptr),dextents<int32_t,2>(256,p.qL));
        int first=max(0,(kr-p.qL_off)/N)*N;
        for(int qr=first;qr<p.qL;qr+=N) {
            auto mq=Q.slice(0,qr),mo=dO.slice(0,qr);
            auto score=score_op.template get_destination_cooperative_tensor<MK,MK,float>();
            auto derivative=score_op.template get_destination_cooperative_tensor<MK,MK,float>();
            #pragma clang loop unroll(full)
            for(ushort i=0;i<score.get_capacity();++i)if(score.is_valid_element(i)) {
                score[i]=0;
                derivative[i]=0;
            }
            for(int i=localidx;i<M*N;i+=SG*32)ds_data[i]=T(0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            score_op.run(mk,mq,score);
            score_op.run(mv,mo,derivative);
            #pragma clang loop unroll(full)
            for(ushort i=0;i<score.get_capacity();++i)if(score.is_valid_element(i)) {
                auto index=score.get_multidimensional_index(i);
                int row=qr+index[0],col=kr+index[1];
                bool valid=row<p.qL && col<p.kL && col<=row+p.qL_off;
                float prob=valid ? fast::exp2(score[i]*0.09016844005556f-float(lp[row*2])-float(lp[row*2+1])) : 0;
                float delta=valid ? float(dp[row*2])+float(dp[row*2+1]) : 0;
                score[i]=prob;
                prob*=derivative[i]-delta;
                ds_data[index[1]*N+index[0]]=T(prob);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            out_op.run(DS,mq,result);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(int i=localidx;i<M*N;i+=SG*32)ds_data[i]=T(0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma clang loop unroll(full)
            for(ushort i=0;i<score.get_capacity();++i)if(score.is_valid_element(i)) {
                auto index=score.get_multidimensional_index(i);
                ds_data[index[1]*N+index[0]]=T(score[i]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            out_op.run(DS,mo,result_value);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    device T* target=output_key+gid.z*p.K_strides[0]+kvhead*p.K_strides[1];
    #pragma clang loop unroll(full)
    for(ushort i=0;i<result.get_capacity();++i)if(result.is_valid_element(i)) {
        auto index=result.get_multidimensional_index(i);
        if(kr+index[1]<p.kL) {
            target[(kr+index[1])*256+index[0]]=T(result[i]*0.0625f);
            output_value[gid.z*p.K_strides[0]+kvhead*p.K_strides[1]+(kr+index[1])*256+index[0]]=T(result_value[i]);
        }
    }
