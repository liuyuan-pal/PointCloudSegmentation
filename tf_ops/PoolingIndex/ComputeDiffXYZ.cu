#include "../TFCudaCommon.h"

__global__
void permutateFeatureKernel(
        float * xyzs,               // [pn1,3]
        float *cxyzs,               // [pn2,3]
        float *dxyzs,               // [pn1,3]
        int *cidxs,                 // [pn1]
        int pn1,
        int pn2
)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    if(pi>=pn1) return;
    int ci=cidxs[pi];
    dxyzs[pi*3+0]=xyzs[pi*3+0]-cxyzs[ci*3+0];
    dxyzs[pi*3+1]=xyzs[pi*3+1]-cxyzs[ci*3+1];
    dxyzs[pi*3+2]=xyzs[pi*3+2]-cxyzs[ci*3+2];
}


void permutateFeatureImpl(
        float * xyzs,               // [pn1,3]
        float *cxyzs,               // [pn2,3]
        float *dxyzs,               // [pn1,3]
        int *cidxs,                 // [pn1]
        int pn1,
        int pn2
)
{
    int block_num = pn1 / 1024;
    if (pn1 % 1024 > 0) block_num++;
    dim3 block_dim(block_num);
    dim3 thread_dim(1024);

    permutateFeatureKernel<<<block_dim,thread_dim>>>(xyzs,cxyzs,dxyzs,cidxs,pn1,pn2);
}