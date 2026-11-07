struct ShellForwardParams {
    int qL,kL,gqa_factor,NQ,NK,qL_rem,kL_rem,qL_off;
    long Q_strides[3],K_strides[3],V_strides[3],O_strides[3],L_strides[3];
};

struct ShellQueryParams {
    int qL,kL,gqa_factor,NQ,NK,qL_rem,kL_rem,qL_off;
    long Q_strides[3],K_strides[3],V_strides[3],O_strides[3],L_strides[3];
    long dO_strides[3],dQ_strides[3],D_strides[3];
};
struct ShellKVParams {
    int qL,kL,gqa_factor,NQ,NK,qL_rem,kL_rem,qL_off;
    long Q_strides[3],K_strides[3],V_strides[3],O_strides[3],L_strides[3];
    long dO_strides[3],dK_strides[3],dV_strides[3],D_strides[3];
};

// Purpose: Reduce four head-dimension partials in a fixed device-local order.
// Input: Register fragments and four SIMD slots. Output: Complete fragments.
METAL_FUNC void shell_exchange_four(thread float* values,short count,
        threadgroup float* scratch,short group,short lane) {
    short base=lane*count;
    STEEL_PRAGMA_UNROLL
    for(short i=0;i<count;++i) scratch[group*32*count+base+i]=values[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_UNROLL
    for(short i=0;i<count;++i) {
        values[i]=(scratch[base+i]+scratch[32*count+base+i])
                 +(scratch[64*count+base+i]+scratch[96*count+base+i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
