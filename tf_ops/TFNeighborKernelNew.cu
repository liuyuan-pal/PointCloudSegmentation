#include "TFNeighborKernel.h"
#include "TFCudaCommon.h"
#include <cfloat>
template<typename FLT_TYPE,typename INT_TYPE>
__global__ void sumFeatGather(
        FLT_TYPE *ifeats,               // [csum,fd]
        INT_TYPE *nidxs_lens,           // [pn]
        INT_TYPE *nidxs_bgs,            // [pn]
        INT_TYPE fd,
        INT_TYPE pn,
        FLT_TYPE *ogfeats_sum           // [pn,fd]
)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    int fi = threadIdx.y + blockIdx.y*blockDim.y;
    if(pi>=pn||fi>=fd) return;
    INT_TYPE nn_bg=nidxs_bgs[pi];
    INT_TYPE nn=nidxs_lens[pi];

    FLT_TYPE *ifeats_p=&ifeats[nn_bg*fd+fi];
    FLT_TYPE *ogfeats_sum_p=&ogfeats_sum[pi*fd+fi];
    for(int ni=0;ni<nn;ni++)
    {
        (*ogfeats_sum_p)+=(*ifeats_p);
        ifeats_p+=fd;
    }
}



template<typename FLT_TYPE,typename INT_TYPE>
void neighborSumFeatGatherGPU(
        FLT_TYPE *d_ifeats,               // [csum,fd]
        INT_TYPE *d_nidxs_lens,           // [pn]
        INT_TYPE *d_nidxs_bgs,            // [pn]
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
    if(pn<tdim0) tdim0=infTwoExp(pn);
    bdim0=pn/tdim0;
    if(pn%tdim0>0) bdim0++;

    dim3 block_dim(bdim0,bdim1,bdim2);
    dim3 thread_dim(tdim0,tdim1,tdim2);

//    printf("%d %d %d\n",csum,m,ofn);
//    printf("%d %d %d\n",tdim0,tdim1,tdim2);
//    printf("%d %d %d\n",bdim0,bdim1,bdim2);

    gpuErrchk(cudaMemset(d_ogfeats_sum,0,pn*fd*sizeof(FLT_TYPE)))
    sumFeatGather<FLT_TYPE,INT_TYPE> <<<block_dim,thread_dim>>>
            (d_ifeats,d_nidxs_lens,d_nidxs_bgs,fd,pn,d_ogfeats_sum);
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



template<typename FLT_TYPE,typename INT_TYPE>
__global__ void maxFeatGather(
        FLT_TYPE *ifeats,               // [pn1,fd]
        INT_TYPE *vlens,                // [pn2]
        INT_TYPE *vlens_bg,             // [pn2]
        INT_TYPE pn2,
        INT_TYPE fd,
        FLT_TYPE *feats_max,            // [pn2,fd]
        INT_TYPE *max_idxs              // [pn2,fd] used in backward
)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    int fi = threadIdx.y + blockIdx.y*blockDim.y;
    if(pi>=pn2||fi>=fd) return;
    INT_TYPE bg=vlens_bg[pi];
    INT_TYPE nn=vlens[pi];

    FLT_TYPE *feats_max_p=&feats_max[pi*fd+fi];
    INT_TYPE *max_idxs_p=&max_idxs[pi*fd+fi];
    FLT_TYPE *ifeats_p=&ifeats[bg*fd+fi];
    (*feats_max_p)=-FLT_MAX;
    for(int ni=0;ni<nn;ni++)
    {
        FLT_TYPE cur_val=*ifeats_p;
        if((*feats_max_p)<cur_val)
        {
            (*feats_max_p)=cur_val;
            (*max_idxs_p)=ni;
        }
        ifeats_p+=fd;
    }
}


template<typename FLT_TYPE,typename INT_TYPE>
__global__ void maxFeatScatter(
        FLT_TYPE *ifeats,               // [pn2,fd]
        INT_TYPE *max_idxs,             // [pn2,fd]
        INT_TYPE *vlens_bg,             // [pn2]
        INT_TYPE pn2,
        INT_TYPE fd,
        FLT_TYPE *sfeats                // [pn1,fd]
)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    int fi = threadIdx.y + blockIdx.y*blockDim.y;
    if(pi>=pn2||fi>=fd) return;


    INT_TYPE wpi=vlens_bg[pi]+max_idxs[pi*fd+fi];
    sfeats[wpi*fd+fi]=ifeats[pi*fd+fi];
}

template<typename FLT_TYPE,typename INT_TYPE>
void neighborMaxFeatGatherGPU(
        FLT_TYPE *d_ifeats,               // [pn1,fd]
        INT_TYPE *d_vlens,                // [pn2]
        INT_TYPE *d_vlens_bgs,            // [pn2]
        INT_TYPE pn2,
        INT_TYPE fd,
        FLT_TYPE *d_ogfeats_sum,          // [pn2,fd]
        INT_TYPE *d_ogmax_idxs            // [pn2,fd]
)
{
    int tdim0,tdim1,tdim2=1;
    int bdim0,bdim1,bdim2=1;

    tdim1=1024/(tdim2);
    if(fd<tdim1) tdim1=infTwoExp(fd);
    bdim1=fd/tdim1;
    if(fd%tdim1>0) bdim1++;

    tdim0=1024/(tdim1*tdim2);
    if(pn2<tdim0) tdim0=infTwoExp(pn2);
    bdim0=pn2/tdim0;
    if(pn2%tdim0>0) bdim0++;

    dim3 block_dim(bdim0,bdim1,bdim2);
    dim3 thread_dim(tdim0,tdim1,tdim2);

    maxFeatGather<FLT_TYPE,INT_TYPE> <<<block_dim,thread_dim>>>
      (d_ifeats,d_vlens,d_vlens_bgs,pn2,fd,d_ogfeats_sum,d_ogmax_idxs);
}


template<typename FLT_TYPE,typename INT_TYPE>
void neighborMaxFeatScatterGPU(
        FLT_TYPE *d_igfeats_sum,            // [pn2,fd]
        INT_TYPE *d_igmax_idxs,             // [pn2,fd]
        INT_TYPE *vlens_bg,                 // [pn2]
        INT_TYPE pn1,
        INT_TYPE pn2,
        INT_TYPE fd,
        FLT_TYPE *d_osfeats                 // [pn1,fd]
)
{

    int tdim0,tdim1,tdim2=1;
    int bdim0,bdim1,bdim2=1;

    tdim1=1024/(tdim2);
    if(fd<tdim1) tdim1=infTwoExp(fd);
    bdim1=fd/tdim1;
    if(fd%tdim1>0) bdim1++;

    tdim0=1024/(tdim1*tdim2);
    if(pn2<tdim0) tdim0=infTwoExp(pn2);
    bdim0=pn2/tdim0;
    if(pn2%tdim0>0) bdim0++;

    dim3 block_dim(bdim0,bdim1,bdim2);
    dim3 thread_dim(tdim0,tdim1,tdim2);

    gpuErrchk(cudaMemset(d_osfeats,0,pn1*fd*sizeof(FLT_TYPE)))
    maxFeatScatter<FLT_TYPE,INT_TYPE> <<<block_dim,thread_dim>>>(d_igfeats_sum,d_igmax_idxs,vlens_bg,pn2,fd,d_osfeats);
}


template void neighborSumFeatGatherGPU<float,unsigned int>
        (float*, unsigned int*,unsigned int*,unsigned int,unsigned int,unsigned int,float*);
template void neighborSumFeatScatterGPU<float,unsigned int>
        (float*, unsigned int*,unsigned int,unsigned int,unsigned int,float*);
template void neighborMaxFeatGatherGPU<float,unsigned int>
        (float*, unsigned int*,unsigned int*,unsigned int,unsigned int,float*,unsigned int*);
template void neighborMaxFeatScatterGPU<float,unsigned int>
        (float*, unsigned int*,unsigned int*,unsigned int,unsigned int,unsigned int,float*);
