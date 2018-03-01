#include "TFNeighborKernel.h"
#include "TFCudaCommon.h"

template<typename FLT_TYPE,typename INT_TYPE>
__global__ void locationWeightFeatSumForwardKernelV2(
        FLT_TYPE *itfeats,              // [pn,n,m,ofn]
        FLT_TYPE *ilw,                  // [pn,n,m]
        INT_TYPE *icidxs,               // [pn,n] or [csum]
        INT_TYPE ofn,
        INT_TYPE m,
        INT_TYPE csum,
        FLT_TYPE *otfeats_sum           // [pn,m,ofn]
)
{
    int ci = threadIdx.x + blockIdx.x*blockDim.x;
    int mi = threadIdx.y + blockIdx.y*blockDim.y;
    int oi = threadIdx.z + blockIdx.z*blockDim.z;
    if(ci>=csum||mi>=m||oi>=ofn) return;

    INT_TYPE pi = icidxs[ci];
    atomicAdd(&otfeats_sum[pi*ofn*m+mi*ofn+oi],
              ilw[ci*m+mi]*itfeats[ci*ofn*m+mi*ofn+oi]);
}


template<typename FLT_TYPE,typename INT_TYPE>
void locWFeatSumForwardGPUV2(
        FLT_TYPE *d_itfeats,              // [pn,n,m,ofn]
        FLT_TYPE *d_ilw,                  // [pn,n,m]
        INT_TYPE *d_icidxs,               // [pn] inidxs_lens[i]=n
        INT_TYPE pn,
        INT_TYPE ofn,
        INT_TYPE m,
        INT_TYPE csum,
        FLT_TYPE *d_otfeats_sum           // [pn,m,ofn]
)
{
    int tdim0,tdim1,tdim2;
    int bdim0,bdim1,bdim2;

    tdim2=64;
    if(ofn<tdim2) tdim2=infTwoExp(ofn);
    bdim2=ofn/tdim2;
    if(ofn%tdim2>0) bdim2++;

    tdim1=1024/(tdim2);
    if(m<tdim1) tdim1=infTwoExp(m);
    bdim1=m/tdim1;
    if(m%tdim1>0) bdim1++;

    tdim0=1024/(tdim1*tdim2);
    if(csum<tdim0) tdim0=infTwoExp(csum);
    bdim0=csum/tdim0;
    if(csum%tdim0>0) bdim0++;

    dim3 block_dim(bdim0,bdim1,bdim2);
    dim3 thread_dim(tdim0,tdim1,tdim2);

//    printf("%d %d %d\n",csum,m,ofn);
//    printf("%d %d %d\n",tdim0,tdim1,tdim2);
//    printf("%d %d %d\n",bdim0,bdim1,bdim2);

    gpuErrchk(cudaMemset(d_otfeats_sum,0,pn*m*ofn*sizeof(FLT_TYPE)))
    locationWeightFeatSumForwardKernelV2<FLT_TYPE,INT_TYPE> <<<block_dim,thread_dim>>>
              (d_itfeats,d_ilw,d_icidxs,ofn,m,csum,d_otfeats_sum);
}


template<typename FLT_TYPE,typename INT_TYPE>
__global__ void locationWeightFeatSumBackwardKernelV2(
        FLT_TYPE *itfeats,              // [pn,n,m,ofn]
        FLT_TYPE *ilw,                  // [pn,n,m]
        FLT_TYPE *dotfeats_sum,         // [pn,m,ofn]
        INT_TYPE *icidxs,               // [csum]
        INT_TYPE ofn,
        INT_TYPE m,
        INT_TYPE csum,
        FLT_TYPE *ditfeats,           // [pn,n,m,ofn]
        FLT_TYPE *dilw                // [pn,n,m]
)
{
    int ci = threadIdx.x + blockIdx.x*blockDim.x;
    int mi = threadIdx.y + blockIdx.y*blockDim.y;
    int oi = threadIdx.z + blockIdx.z*blockDim.z;
    if(ci>=csum||mi>=m||oi>=ofn) return;

    INT_TYPE pi=icidxs[ci];
    FLT_TYPE *dotfeats_sum_p=&dotfeats_sum[pi*ofn*m+mi*ofn+oi];
    FLT_TYPE *ilw_p=&ilw[ci*m+mi];
    FLT_TYPE *itfeats_p=&itfeats[ci*ofn*m+mi*ofn+oi];

    FLT_TYPE *dilw_p=&dilw[ci*m+mi];
    FLT_TYPE *ditfeats_p=&ditfeats[ci*ofn*m+mi*ofn+oi];

    (*ditfeats_p)=(*dotfeats_sum_p)*(*ilw_p);
    atomicAdd(dilw_p,(*dotfeats_sum_p)*(*itfeats_p));
}


template<typename FLT_TYPE,typename INT_TYPE>
void locWFeatSumBackwardGPUV2(
        FLT_TYPE *d_itfeats,              // [pn,n,m,ofn]
        FLT_TYPE *d_ilw,                  // [pn,n,m]
        FLT_TYPE *d_dotfeats_sum,         // [pn,m,ofn]
        INT_TYPE *d_icidxs,                // [csum] inidxs_lens[i]=n
        INT_TYPE pn,
        INT_TYPE ofn,
        INT_TYPE m,
        INT_TYPE csum,
        FLT_TYPE *d_ditfeats,           // [pn,n,m,ofn]
        FLT_TYPE *d_dilw                // [pn,n,m]
)
{
    int tdim0,tdim1,tdim2;
    int bdim0,bdim1,bdim2;

    tdim2=64;
    if(ofn<tdim2) tdim2=infTwoExp(ofn);
    bdim2=ofn/tdim2;
    if(ofn%tdim2>0) bdim2++;

    tdim1=1024/(tdim2);
    if(m<tdim1) tdim1=infTwoExp(m);
    bdim1=m/tdim1;
    if(m%tdim1>0) bdim1++;

    tdim0=1024/(tdim1*tdim2);
    if(csum<tdim0) tdim0=infTwoExp(csum);
    bdim0=csum/tdim0;
    if(csum%tdim0>0) bdim0++;

    dim3 block_dim(bdim0,bdim1,bdim2);
    dim3 thread_dim(tdim0,tdim1,tdim2);

    //gpuErrchk(cudaMemset(d_ditfeats,0,csum*m*ofn*sizeof(FLT_TYPE)))
    gpuErrchk(cudaMemset(d_dilw,0,csum*m*sizeof(FLT_TYPE)))
    locationWeightFeatSumBackwardKernelV2<FLT_TYPE,INT_TYPE> <<<block_dim,thread_dim>>>
           (d_itfeats,d_ilw,d_dotfeats_sum,d_icidxs,ofn,m,csum,d_ditfeats,d_dilw);
}


template void locWFeatSumForwardGPUV2<float,unsigned int>
        (float*, float*, unsigned int*,unsigned int,unsigned int,unsigned int,unsigned int,float*);
template void locWFeatSumBackwardGPUV2<float,unsigned int>
        (float*,float*,float*,unsigned int*,unsigned int,unsigned int,unsigned int,unsigned int,float*,float*);
