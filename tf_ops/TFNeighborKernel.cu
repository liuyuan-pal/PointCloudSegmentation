#include <cstring>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "TFNeighborKernel.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

template<typename FLT_TYPE,typename INT_TYPE>
__global__ void diffFeatScatter(
        FLT_TYPE *ifeats,                   // [pn,ifn]
        INT_TYPE *inidxs,               // [pn,n]
        INT_TYPE *inn_bgs,              // [pn]
        INT_TYPE *inidxs_lens,          // [pn]
        INT_TYPE pn,
        INT_TYPE ifn,
        FLT_TYPE *onfeats                     // [pn*n,ifn]
)
{
    int pi = threadIdx.y + blockIdx.y*blockDim.y;
    if(pi>=pn) return;
    INT_TYPE nn_bg = inn_bgs[pi];
    INT_TYPE nn = inidxs_lens[pi];

    FLT_TYPE* cifeats=&ifeats[pi*ifn];
    INT_TYPE* inidxs_p=&inidxs[nn_bg];
    FLT_TYPE* onfeats_p=&onfeats[nn_bg*ifn];

    for(int ni=0;ni<nn;ni++)
    {
        FLT_TYPE* niloc = &ifeats[(*inidxs_p)*ifn];
        for(int li=0;li<ifn;li++)
            onfeats_p[li]=niloc[li]-cifeats[li];
        inidxs_p++;
        onfeats_p+=ifn;
    }
}

template<typename FLT_TYPE,typename INT_TYPE>
__global__ void featScatter(
        FLT_TYPE *ifeats,                   // [pn,ifn]
        INT_TYPE *inidxs,               // [pn,n]
        INT_TYPE *inn_bgs,              // [pn]
        INT_TYPE *inidxs_lens,          // [pn]
        INT_TYPE pn,
        INT_TYPE ifn,
        FLT_TYPE *onfeats                   // [pn*n,ifn]
)
{
    int pi = threadIdx.y + blockIdx.y*blockDim.y;
    if(pi>=pn) return;
    INT_TYPE nn_bg = inn_bgs[pi];
    INT_TYPE nn = inidxs_lens[pi];

    INT_TYPE *inidxs_p=&inidxs[nn_bg];
    FLT_TYPE *onfeats_p=&onfeats[nn_bg*ifn];
    for(int ni=0;ni<nn;ni++)
    {
        INT_TYPE cur_nidx=(*inidxs_p);
        FLT_TYPE* cifeats=&ifeats[cur_nidx*ifn];
        std::memcpy(onfeats_p,cifeats,ifn*sizeof(FLT_TYPE));
        inidxs_p++;
        onfeats_p+=ifn;
    }
}


// todo gather in the (pn,ifn) level can avoid atomic operation
template<typename FLT_TYPE,typename INT_TYPE>
__global__ void featGather(
        FLT_TYPE *sfeats,              // [csum,ifn]
        INT_TYPE *inidxs,               // [csum]
        INT_TYPE *inn_bgs,              // [pn]
        INT_TYPE *inidxs_lens,          // [pn]
        INT_TYPE pn,
        INT_TYPE ifn,
        FLT_TYPE *ifeats               // [pn,ifn]
)
{
    int pi = threadIdx.y + blockIdx.y*blockDim.y;
    if(pi>=pn) return;
    INT_TYPE nn_bg = inn_bgs[pi];
    INT_TYPE nn = inidxs_lens[pi];

    INT_TYPE* inidxs_p=&(inidxs[nn_bg]);
    FLT_TYPE* sfeats_p=&sfeats[nn_bg*ifn];
    for(int ni=0;ni<nn;ni++)
    {
        FLT_TYPE* cifeats_p=&ifeats[(*inidxs_p)*ifn];
        for(int ii=0;ii<ifn;ii++)
            atomicAdd(&(cifeats_p[ii]),sfeats_p[ii]);

        inidxs_p++;
        sfeats_p+=ifn;
    }
}

template<typename FLT_TYPE,typename INT_TYPE>
__global__ void diffFeatGather(
        FLT_TYPE *sfeats,                  // [csum,ifn]
        INT_TYPE *inidxs,               // [csum]
        INT_TYPE *inn_bgs,              // [pn]
        INT_TYPE *inidxs_lens,          // [pn]
        INT_TYPE pn,
        INT_TYPE ifn,
        FLT_TYPE *ifeats                   // [pn,ifn]
)
{
    int pi = threadIdx.y + blockIdx.y*blockDim.y;
    if(pi>=pn) return;
    unsigned int nn_bg = inn_bgs[pi];
    unsigned int nn = inidxs_lens[pi];

    // current minus gradient
    FLT_TYPE *cifeats_p=&ifeats[pi*ifn];
    FLT_TYPE *sfeats_p=&sfeats[nn_bg*ifn];
    for(unsigned int ni=0;ni<nn;ni++)
    {
        for(unsigned int ii=0;ii<ifn;ii++)
            cifeats_p[ii]+=-sfeats_p[ii];
        sfeats_p+=ifn;
    }
    __syncthreads();

    unsigned int* inidxs_p=&(inidxs[nn_bg]);
    sfeats_p=&sfeats[nn_bg*ifn];
    for(unsigned int ni=0;ni<nn;ni++)
    {
        FLT_TYPE* nifeats=&ifeats[(*inidxs_p)*ifn];
        for(unsigned int ii=0;ii<ifn;ii++)
            atomicAdd(&(nifeats[ii]),sfeats_p[ii]);
        inidxs_p++;
        sfeats_p+=ifn;
    }
}


template<typename FLT_TYPE,typename INT_TYPE>
__global__ void locWSumForwardKernel(
        FLT_TYPE *ilw,                      // pn,n,m
        INT_TYPE *inn_bgs,              // [pn] n=nidxs_lens[i]
        INT_TYPE *inidxs_lens,          // [pn] n=nidxs_lens[i]
        INT_TYPE pn,
        INT_TYPE m,
        FLT_TYPE *olw_sum                   // pn,m
)
{
    int pi = threadIdx.y + blockIdx.y*blockDim.y;
    int mi = threadIdx.x + blockIdx.x*blockDim.x;
    if(pi>=pn||mi>=m) return;

    INT_TYPE nn_bg = inn_bgs[pi];
    INT_TYPE nn = inidxs_lens[pi];
    FLT_TYPE *ilw_p=&ilw[nn_bg*m+mi];
    FLT_TYPE *olw_sum_p=&olw_sum[pi*m+mi];
    for(int ni=0;ni<nn;ni++)
    {
        (*olw_sum_p)+=(*ilw_p);
        ilw_p+=m;
    }
}

template<typename FLT_TYPE,typename INT_TYPE>
__global__ void locWSumBackwardKernel(
        FLT_TYPE *dolw_sum,                 // pn,m
        INT_TYPE *inn_bgs,              // [pn]
        INT_TYPE *inidxs_lens,          // [pn] n=nidxs_lens[i]
        INT_TYPE pn,
        INT_TYPE m,
        FLT_TYPE *dilw                      // [pn,n,m]
)
{
    int pi = threadIdx.y + blockIdx.y*blockDim.y;
    int mi = threadIdx.x + blockIdx.x*blockDim.x;
    if(pi>=pn||mi>=m) return;

    INT_TYPE nn_bg = inn_bgs[pi];
    INT_TYPE nn = inidxs_lens[pi];

    FLT_TYPE *dilw_p=&dilw[nn_bg*m+mi];
    FLT_TYPE *dolw_sum_p=&dolw_sum[pi*m+mi];
    for(int ni=0;ni<nn;ni++)
    {
        (*dilw_p)=(*dolw_sum_p);
        dilw_p+=m;
    }
}

template<typename FLT_TYPE,typename INT_TYPE>
__global__ void locationWeightFeatSumForwardKernel(
        FLT_TYPE *itfeats,              // [pn,n,m,ofn]
        FLT_TYPE *ilw,                  // [pn,n,m]
        INT_TYPE *inn_bgs,          // [pn] inidxs_lens[i]=n
        INT_TYPE *inidxs_lens,      // [pn] inidxs_lens[i]=n
        INT_TYPE pn,
        INT_TYPE ofn,
        INT_TYPE m,
        FLT_TYPE *otfeats_sum           // [pn,m,ofn]
)
{
//    printf("here");
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    int mi = threadIdx.y + blockIdx.y*blockDim.y;
    int oi = threadIdx.z + blockIdx.z*blockDim.z;
//    printf("here %d %d %d\n",mi,oi,pi);
    if(pi>=pn||mi>=m||oi>=ofn) return;

    INT_TYPE nn_bg = inn_bgs[pi];
    INT_TYPE nn = inidxs_lens[pi];

    FLT_TYPE *otfeats_sum_p=&otfeats_sum[pi*ofn*m+mi*ofn+oi];
    FLT_TYPE *ilw_p=&ilw[nn_bg*m+mi];
    FLT_TYPE *itfeats_p=&itfeats[nn_bg*ofn*m+mi*ofn+oi];
    for(int ni=0;ni<nn;ni++)
    {
        (*otfeats_sum_p)+=(*ilw_p)*(*itfeats_p);
        itfeats_p+=ofn*m;
        ilw_p+=m;
    }
}

template<typename FLT_TYPE,typename INT_TYPE>
__global__ void locationWeightFeatSumBackwardKernel(
        FLT_TYPE *itfeats,              // [pn,n,m,ofn]
        FLT_TYPE *ilw,                  // [pn,n,m]
        FLT_TYPE *dotfeats_sum,         // [pn,m,ofn]
        INT_TYPE *inn_bgs,          // [pn] inidxs_lens[i]=n
        INT_TYPE *inidxs_lens,      // [pn] inidxs_lens[i]=n
        INT_TYPE pn,
        INT_TYPE ofn,
        INT_TYPE m,
        FLT_TYPE *ditfeats,           // [pn,n,m,ofn]
        FLT_TYPE *dilw                // [pn,n,m]
)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    int mi = threadIdx.y + blockIdx.y*blockDim.y;
    int oi = threadIdx.z + blockIdx.z*blockDim.z;
    if(pi>=pn||mi>=m||oi>=ofn) return;

    INT_TYPE nn_bg = inn_bgs[pi];
    INT_TYPE nn = inidxs_lens[pi];

    FLT_TYPE *dotfeats_sum_p=&dotfeats_sum[pi*ofn*m+mi*ofn+oi];
    FLT_TYPE *ilw_p=&ilw[nn_bg*m+mi];
    FLT_TYPE *itfeats_p=&itfeats[nn_bg*ofn*m+mi*ofn+oi];

    FLT_TYPE *dilw_p=&dilw[nn_bg*m+mi];
    FLT_TYPE *ditfeats_p=&ditfeats[nn_bg*ofn*m+mi*ofn+oi];

    for(int ni=0;ni<nn;ni++)
    {
        (*ditfeats_p)=(*dotfeats_sum_p)*(*ilw_p);
        atomicAdd(dilw_p,(*dotfeats_sum_p)*(*itfeats_p));

        itfeats_p+=ofn*m;
        ditfeats_p+=ofn*m;
        ilw_p+=m;
        dilw_p+=m;
    }
}

template<typename FLT_TYPE,typename INT_TYPE>
void neighborScatterGPU(
        FLT_TYPE *d_ifeats,          // [pn,ifn]
        INT_TYPE *d_inidxs,      // [pn,n] or [csum]
        INT_TYPE *d_inidxs_lens, // [pn] n=nidxs_lens[i]
        INT_TYPE *d_inn_bgs,     // [pn] n=nidxs_lens[i]
        INT_TYPE pn,
        INT_TYPE ifn,
        FLT_TYPE *d_sfeats,           // [csum,ifn]
        bool use_diff                 // default false
)
{
    int block_num=pn/1024;
    if(pn%1024>0) block_num++;
    dim3 block_dim(1,block_num);
    dim3 thread_dim(1,1024);

    // scatter data to matrix
    if(use_diff)
        diffFeatScatter<FLT_TYPE,INT_TYPE> <<<block_dim,thread_dim>>>(d_ifeats,d_inidxs,d_inn_bgs,d_inidxs_lens,pn,ifn,d_sfeats);
    else
        featScatter<FLT_TYPE,INT_TYPE> <<<block_dim,thread_dim>>>(d_ifeats,d_inidxs,d_inn_bgs,d_inidxs_lens,pn,ifn,d_sfeats);
}


template<typename FLT_TYPE,typename INT_TYPE>
void neighborGatherGPU(
        FLT_TYPE *d_sfeats,           // [pn,ifn]
        INT_TYPE *d_inidxs,       // [pn,n] or [csum]
        INT_TYPE *d_inidxs_lens,  // [pn] n=nidxs_lens[i]
        INT_TYPE *d_inn_bgs,      // [pn] n=nidxs_lens[i]
        INT_TYPE pn,
        INT_TYPE ifn,
        FLT_TYPE *d_ifeats,           // [csum,ifn]
        bool use_diff                 // default false
)
{
    int block_num=pn/1024;
    if(pn%1024>0) block_num++;
    dim3 block_dim(1,block_num);
    dim3 thread_dim(1,1024);

    gpuErrchk(cudaMemset(d_ifeats,0,pn*ifn*sizeof(FLT_TYPE)))
    // scatter data to matrix
    if(use_diff)
        diffFeatGather<FLT_TYPE,INT_TYPE> <<<block_dim,thread_dim>>> (d_sfeats,d_inidxs,d_inn_bgs,d_inidxs_lens,pn,ifn,d_ifeats);
    else
        featGather<FLT_TYPE,INT_TYPE> <<<block_dim,thread_dim>>> (d_sfeats,d_inidxs,d_inn_bgs,d_inidxs_lens,pn,ifn,d_ifeats);
}

inline int infTwoExp(int val)
{
    int cval=val;
    int inf=1;
    while(cval>1)
    {
        cval>>=1;
        inf<<=1;
    }
    return inf;
}

template<typename FLT_TYPE,typename INT_TYPE>
void locWFeatSumForwardGPU(
        FLT_TYPE *d_itfeats,              // [pn,n,m,ofn]
        FLT_TYPE *d_ilw,                  // [pn,n,m]
        INT_TYPE *d_inidxs_lens,      // [pn] inidxs_lens[i]=n
        INT_TYPE *d_inn_bgs,          // [pn] inidxs_lens[i]=n
        INT_TYPE pn,
        INT_TYPE ofn,
        INT_TYPE m,
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
    if(pn<tdim0) tdim0=infTwoExp(pn);
    bdim0=pn/tdim0;
    if(pn%tdim0>0) bdim0++;

    dim3 block_dim(bdim0,bdim1,bdim2);
    dim3 thread_dim(tdim0,tdim1,tdim2);

    gpuErrchk(cudaMemset(d_otfeats_sum,0,pn*m*ofn*sizeof(FLT_TYPE)))
    locationWeightFeatSumForwardKernel<FLT_TYPE,INT_TYPE> <<<block_dim,thread_dim>>>
            (d_itfeats,d_ilw,d_inn_bgs,d_inidxs_lens,pn,ofn,m,d_otfeats_sum);
}

template<typename FLT_TYPE,typename INT_TYPE>
void locWFeatSumBackwardGPU(
        FLT_TYPE *d_itfeats,              // [pn,n,m,ofn]
        FLT_TYPE *d_ilw,                  // [pn,n,m]
        FLT_TYPE *d_dotfeats_sum,         // [pn,m,ofn]
        INT_TYPE *d_inidxs_lens,      // [pn] inidxs_lens[i]=n
        INT_TYPE *d_inn_bgs,          // [pn] inidxs_lens[i]=n
        INT_TYPE csum,
        INT_TYPE pn,
        INT_TYPE ofn,
        INT_TYPE m,
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
    if(pn<tdim0) tdim0=infTwoExp(pn);
    bdim0=pn/tdim0;
    if(pn%tdim0>0) bdim0++;

    dim3 block_dim(bdim0,bdim1,bdim2);
    dim3 thread_dim(tdim0,tdim1,tdim2);

    //gpuErrchk(cudaMemset(d_ditfeats,0,csum*m*ofn*sizeof(FLT_TYPE)))
    gpuErrchk(cudaMemset(d_dilw,0,csum*m*sizeof(FLT_TYPE)))
    locationWeightFeatSumBackwardKernel<FLT_TYPE,INT_TYPE> <<<block_dim,thread_dim>>>
            (d_itfeats,d_ilw,d_dotfeats_sum,d_inn_bgs,d_inidxs_lens,pn,ofn,m,d_ditfeats,d_dilw);
}


template<typename FLT_TYPE,typename INT_TYPE>
void locWSumForwardGPU(
        FLT_TYPE *d_ilw,                      // pn,n,m
        INT_TYPE *d_inidxs_lens,          // [pn] n=nidxs_lens[i]
        INT_TYPE *d_inn_bgs,          // [pn] n=nidxs_lens[i]
        INT_TYPE pn,
        INT_TYPE m,
        FLT_TYPE *d_olw_sum                   // pn,m
)
{

    int block_num=pn/1024;
    if(pn%1024>0) block_num++;
    dim3 block_dim(m,block_num);
    dim3 thread_dim(1,1024);

    // scatter data to matrix
    gpuErrchk(cudaMemset(d_olw_sum,0,pn*m*sizeof(FLT_TYPE)))
    locWSumForwardKernel<FLT_TYPE,INT_TYPE><<<block_dim,thread_dim>>>(d_ilw,d_inn_bgs,d_inidxs_lens,pn,m,d_olw_sum);
}

template<typename FLT_TYPE,typename INT_TYPE>
void locWSumBackwardGPU(
        FLT_TYPE *d_dolw_sum,         // pn,m
        INT_TYPE *d_inidxs_lens,      // [pn] n=nidxs_lens[i]
        INT_TYPE *d_inn_bgs,          // [pn] n=nidxs_lens[i]
        INT_TYPE csum,
        INT_TYPE pn,
        INT_TYPE m,
        FLT_TYPE *d_dilw              // [pn,n,m]
)
{
    int block_num=pn/1024;
    if(pn%1024>0) block_num++;
    dim3 block_dim(m,block_num);
    dim3 thread_dim(1,1024);

    gpuErrchk(cudaMemset(d_dilw,0,csum*m*sizeof(FLT_TYPE)))
    locWSumBackwardKernel<FLT_TYPE, INT_TYPE> <<<block_dim,thread_dim>>>(d_dolw_sum,d_inn_bgs,d_inidxs_lens,pn,m, d_dilw);
}

template void neighborGatherGPU<float,unsigned int>
        (float*,unsigned int*,unsigned int*,unsigned int*,unsigned int,unsigned int,float*,bool);
template void neighborScatterGPU<float,unsigned int>
        (float*,unsigned int*,unsigned int*,unsigned int*,unsigned int,unsigned int,float*,bool);
template void locWFeatSumForwardGPU<float,unsigned int>
        (float*, float*, unsigned int*,unsigned int*,unsigned int,unsigned int,unsigned int,float*);
template void locWFeatSumBackwardGPU<float,unsigned int>
        (float*,float*,float*,unsigned int*,unsigned int*,unsigned int,unsigned int,unsigned int,unsigned int,float*,float*);
template void locWSumForwardGPU<float,unsigned int>
        (float*,unsigned int*,unsigned int*,unsigned int,unsigned int,float*);
template void locWSumBackwardGPU<float,unsigned int>
        (float*,unsigned int*,unsigned int*,unsigned int,unsigned int,unsigned int,float*);
