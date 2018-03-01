#include "TFNeighborKernel.h"
#include "TFCudaCommon.h"

template<typename FLT_TYPE,typename INT_TYPE>
__global__ void sumFeatGather(
        FLT_TYPE *ifeats,               // [csum,fd]
        INT_TYPE *icidxs,               // [csum]
        INT_TYPE fd,
        INT_TYPE csum,
        FLT_TYPE *ogfeats_sum           // [pn,fd]
)
{
    int ci = threadIdx.x + blockIdx.x*blockDim.x;
    int fi = threadIdx.y + blockIdx.y*blockDim.y;
    if(ci>=csum||fi>=fd) return;

    INT_TYPE pi = icidxs[ci];
    atomicAdd(&ogfeats_sum[pi*fd+fi],ifeats[ci*fd+fi]);
}


template<typename FLT_TYPE,typename INT_TYPE>
void neighborSumFeatGatherGPU(
        FLT_TYPE *d_ifeats,               // [csum,fd]
        INT_TYPE *d_icidxs,               // [csum]
        INT_TYPE pn,
        INT_TYPE fd,
        INT_TYPE csum,
        FLT_TYPE *d_ogfeats_sum           // [pn,fd]
)
{
    int tdim0,tdim1,tdim2=1;
    int bdim0,bdim1,bdim2=1;

    tdim1=1024/(tdim2);
    if(fd<tdim1) tdim1=infTwoExp(fd);
    bdim1=fd/tdim1;
    if(fd%tdim1>0) bdim1++;

    tdim0=1024/(tdim1*tdim2);
    if(csum<tdim0) tdim0=infTwoExp(csum);
    bdim0=csum/tdim0;
    if(csum%tdim0>0) bdim0++;

    dim3 block_dim(bdim0,bdim1,bdim2);
    dim3 thread_dim(tdim0,tdim1,tdim2);

//    printf("%d %d %d\n",csum,m,ofn);
//    printf("%d %d %d\n",tdim0,tdim1,tdim2);
//    printf("%d %d %d\n",bdim0,bdim1,bdim2);

    gpuErrchk(cudaMemset(d_ogfeats_sum,0,pn*fd*sizeof(FLT_TYPE)))
    sumFeatGather<FLT_TYPE,INT_TYPE> <<<block_dim,thread_dim>>>(d_ifeats,d_icidxs,fd,csum,d_ogfeats_sum);
}


template<typename FLT_TYPE,typename INT_TYPE>
__global__ void sumFeatScatter(
        FLT_TYPE *igfeats_sum,          // [pn,fd]
        INT_TYPE *icidxs,               // [csum]
        INT_TYPE fd,
        INT_TYPE csum,
        FLT_TYPE *osfeats               // [csum,fd]
)
{
    int ci = threadIdx.x + blockIdx.x*blockDim.x;
    int fi = threadIdx.y + blockIdx.y*blockDim.y;
    if(ci>=csum||fi>=fd) return;

    INT_TYPE pi=icidxs[ci];
    osfeats[ci*fd+fi]=igfeats_sum[pi*fd+fi];
}


template<typename FLT_TYPE,typename INT_TYPE>
void neighborSumFeatScatterGPU(
        FLT_TYPE *d_igfeats_sum,            // [pn,fd]
        INT_TYPE *d_icidxs,                 // [csum] inidxs_lens[i]=n
        INT_TYPE pn,
        INT_TYPE fd,
        INT_TYPE csum,
        FLT_TYPE *d_osfeats                 // [csum,fd]
)
{
    int tdim0,tdim1,tdim2=1;
    int bdim0,bdim1,bdim2=1;

    tdim1=1024/(tdim2);
    if(fd<tdim1) tdim1=infTwoExp(fd);
    bdim1=fd/tdim1;
    if(fd%tdim1>0) bdim1++;

    tdim0=1024/(tdim1*tdim2);
    if(csum<tdim0) tdim0=infTwoExp(csum);
    bdim0=csum/tdim0;
    if(csum%tdim0>0) bdim0++;

    dim3 block_dim(bdim0,bdim1,bdim2);
    dim3 thread_dim(tdim0,tdim1,tdim2);

    sumFeatScatter<FLT_TYPE,INT_TYPE> <<<block_dim,thread_dim>>>(d_igfeats_sum,d_icidxs,fd,csum,d_osfeats);
}


template void neighborSumFeatGatherGPU<float,unsigned int>
        (float*, unsigned int*,unsigned int,unsigned int,unsigned int,float*);
template void neighborSumFeatScatterGPU<float,unsigned int>
        (float*, unsigned int*,unsigned int,unsigned int,unsigned int,float*);
