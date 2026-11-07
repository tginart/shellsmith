#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
using T=bfloat;
struct ShellParams {
    int qL,kL,gqa_factor,NQ,NK,qL_rem,kL_rem,qL_off;
    long Q_strides[3],K_strides[3],V_strides[3],O_strides[3],L_strides[3];
};

constant constexpr int M=32,N=256,SG=8;
