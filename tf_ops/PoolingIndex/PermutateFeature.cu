#include "../TFCudaCommon.h"

__global__
void permutateFeatureKernel(
        float *feats,               // [pn,ps]
        int *idxs,                  // [pn]
        float *permutated_feats,    // [pn,ps]
        int pn,
        int ps
)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    int fi = threadIdx.y + blockIdx.y*blockDim.y;
    if(pi>=pn||fi>=ps)  return;

    permutated_feats[pi*ps+fi]=feats[idxs[pi]*ps+fi];
}

void permutateFeature(
        float *feats,               // [pn,ps]
        int *idxs,                  // [pn]
        float *permutated_feats,    // [pn,ps]
        int pn,
        int ps
)
{
    int tdim0,tdim1,tdim2=1;
    int bdim0,bdim1,bdim2=1;

    tdim1=1024/(tdim2);
    if(ps<tdim1) tdim1=infTwoExp(ps);
    bdim1=ps/tdim1;
    if(ps%tdim1>0) bdim1++;

    tdim0=1024/(tdim1*tdim2);
    if(pn<tdim0) tdim0=infTwoExp(pn);
    bdim0=pn/tdim0;
    if(pn%tdim0>0) bdim0++;

    dim3 block_dim(bdim0,bdim1,bdim2);
    dim3 thread_dim(tdim0,tdim1,tdim2);
    permutateFeatureKernel <<<block_dim,thread_dim>>>(feats,idxs,permutated_feats,pn,ps);
}