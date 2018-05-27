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
    if(bdim0==0||tdim0==0) return;

    sumFeatScatter<FLT_TYPE,INT_TYPE> <<<block_dim,thread_dim>>>(d_igfeats_sum,d_icidxs,fd,csum,d_osfeats);
    gpuErrchk(cudaGetLastError())
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
    if(nn==0) return;

    FLT_TYPE *feats_max_p=&feats_max[pi*fd+fi];
    INT_TYPE *max_idxs_p=&max_idxs[pi*fd+fi];
    FLT_TYPE *ifeats_p=&ifeats[bg*fd+fi];

    (*feats_max_p)=*ifeats_p;
    (*max_idxs_p)=0;
    ifeats_p+=fd;
    for(int ni=1;ni<nn;ni++)
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
        INT_TYPE *vlens,                // [pn2]
        INT_TYPE pn2,
        INT_TYPE fd,
        FLT_TYPE *sfeats                // [pn1,fd]
)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    int fi = threadIdx.y + blockIdx.y*blockDim.y;
    if(pi>=pn2||fi>=fd) return;
    if(vlens[pi]==0) return;

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

    gpuErrchk(cudaMemset(d_ogfeats_sum,0,pn2*fd*sizeof(FLT_TYPE)))
    maxFeatGather<FLT_TYPE,INT_TYPE> <<<block_dim,thread_dim>>>
      (d_ifeats,d_vlens,d_vlens_bgs,pn2,fd,d_ogfeats_sum,d_ogmax_idxs);
    gpuErrchk(cudaGetLastError())
}


template<typename FLT_TYPE,typename INT_TYPE>
void neighborMaxFeatScatterGPU(
        FLT_TYPE *d_igfeats_sum,            // [pn2,fd]
        INT_TYPE *d_igmax_idxs,             // [pn2,fd]
        INT_TYPE *vlens_bg,                 // [pn2]
        INT_TYPE *vlens,                    // [pn2]
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
    maxFeatScatter<FLT_TYPE,INT_TYPE> <<<block_dim,thread_dim>>>(d_igfeats_sum,d_igmax_idxs,vlens_bg,vlens,pn2,fd,d_osfeats);
    gpuErrchk(cudaGetLastError())
}


template<typename FLT_TYPE,typename INT_TYPE>
__global__ void concatNonCenterFeatScatter(
        FLT_TYPE *ifeats,                   // [pn,ifn]
        INT_TYPE *inidxs,                   // [pn,n]
        INT_TYPE *inidxs_lens,              // [pn]
        INT_TYPE *inn_bgs,                  // [pn]
        INT_TYPE pn,
        INT_TYPE ifn,
        FLT_TYPE *onfeats                   // [pn*(n-1),2*ifn]
)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    if(pi>=pn) return;
    INT_TYPE nn_bg = inn_bgs[pi];
    INT_TYPE nn = inidxs_lens[pi];

//    printf("nn_bg %d nn %d pi %d ifn %d\n",nn_bg,nn,pi,ifn);

    FLT_TYPE* cifeats=&ifeats[pi*ifn];
    INT_TYPE* inidxs_p=&inidxs[nn_bg];
    FLT_TYPE* onfeats_p=&onfeats[(nn_bg-pi)*ifn*2];

    for(int ni=0;ni<nn;ni++)
    {
        int idx=(*inidxs_p);
        inidxs_p++;
        if(idx==pi) continue;
        FLT_TYPE* niloc = &ifeats[idx*ifn];
        for(int li=0;li<ifn;li++)
        {
            onfeats_p[li]=cifeats[li];
            onfeats_p[li+ifn]=niloc[li];
        }
        onfeats_p+=ifn*2;
    }
}


template<typename FLT_TYPE,typename INT_TYPE>
__global__ void concatNonCenterFeatGather(
        FLT_TYPE *sfeats,                  // [pn*(n-1),2*ifn]
        INT_TYPE *inidxs,                  // [n]
        INT_TYPE *inidxs_lens,             // [pn]
        INT_TYPE *inn_bgs,                 // [pn]
        INT_TYPE pn,
        INT_TYPE ifn,
        FLT_TYPE *ifeats                   // [pn,ifn]
)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    if(pi>=pn) return;
    unsigned int nn_bg = inn_bgs[pi];
    unsigned int nn = inidxs_lens[pi];

    // current minus gradient
    FLT_TYPE *cifeats_p=&ifeats[pi*ifn];
    FLT_TYPE *sfeats_p=&sfeats[(nn_bg-pi)*ifn*2];
    for(unsigned int ni=0;ni<nn-1;ni++)
    {
        for(unsigned int ii=0;ii<ifn;ii++)
            cifeats_p[ii]+=sfeats_p[ii];
        sfeats_p+=ifn*2;
    }
    __syncthreads();

    unsigned int* inidxs_p=&(inidxs[nn_bg]);
    sfeats_p=&sfeats[(nn_bg-pi)*ifn*2];
    for(unsigned int ni=0;ni<nn;ni++)
    {
        int idx=(*inidxs_p);
        inidxs_p++;
        if(idx==pi) continue;
        FLT_TYPE* nifeats=&ifeats[idx*ifn];
        for(unsigned int ii=0;ii<ifn;ii++)
            atomicAdd(&(nifeats[ii]),sfeats_p[ii+ifn]);
        sfeats_p+=ifn*2;
    }
}

template<typename FLT_TYPE,typename INT_TYPE>
void concatNonCenterFeatScatterGPU(
        FLT_TYPE *d_ifeats,                   // [pn,ifn]
        INT_TYPE *d_inidxs,                   // [pn,n]
        INT_TYPE *d_inidxs_lens,              // [pn]
        INT_TYPE *d_inn_bgs,                  // [pn]
        INT_TYPE pn,
        INT_TYPE ifn,
        FLT_TYPE *d_onfeats                   // [pn*(n-1),2*ifn]
)
{

//    printf("here\n");
    int block_num=pn/1024;
    if(pn%1024>0) block_num++;
    dim3 block_dim(block_num);
    dim3 thread_dim(1024);
//    printf("here\n");
    concatNonCenterFeatScatter<FLT_TYPE,INT_TYPE> <<<block_dim,thread_dim>>>
            (d_ifeats,d_inidxs,d_inidxs_lens,d_inn_bgs,pn,ifn,d_onfeats);
    gpuErrchk(cudaGetLastError())
//    printf("forward end\n");
}

template<typename FLT_TYPE,typename INT_TYPE>
void concatNonCenterFeatGatherGPU(
        FLT_TYPE *d_sfeats,                  // [pn*(n-1),2*ifn]
        INT_TYPE *d_inidxs,                  // [n]
        INT_TYPE *d_inidxs_lens,             // [pn]
        INT_TYPE *d_inn_bgs,                 // [pn]
        INT_TYPE pn,
        INT_TYPE ifn,
        FLT_TYPE *d_ifeats                   // [pn,ifn]
)
{
//    printf("here\n");
    int block_num=pn/1024;
    if(pn%1024>0) block_num++;
    dim3 block_dim(block_num);
    dim3 thread_dim(1024);
//    printf("here\n");
    gpuErrchk(cudaMemset(d_ifeats,0,pn*ifn*sizeof(float)))
    concatNonCenterFeatGather<FLT_TYPE,INT_TYPE> <<<block_dim,thread_dim>>>
               (d_sfeats,d_inidxs,d_inidxs_lens,d_inn_bgs,pn,ifn,d_ifeats);
    gpuErrchk(cudaGetLastError())
//    printf("backward end\n");
}

template<typename INT_TYPE>
__global__ void eliminateCenter(
        INT_TYPE *inidxs,                   // [n]
        INT_TYPE *inidxs_lens,              // [pn]
        INT_TYPE *inidxs_bgs,               // [pn]
        INT_TYPE pn,
        INT_TYPE *onidxs,                   // [n-pn]
        INT_TYPE *onidxs_lens,              // [pn]
        INT_TYPE *onidxs_bgs,               // [pn]
        INT_TYPE *ocidxs                    // [n-pn]

)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    if(pi>=pn) return;

    INT_TYPE nn_bg=inidxs_bgs[pi];
    onidxs_bgs[pi]=nn_bg-pi;

    INT_TYPE nn_size=inidxs_lens[pi];
    onidxs_lens[pi]=nn_size-1;

    INT_TYPE* onidxs_p=&onidxs[nn_bg-pi];
    INT_TYPE* inidxs_p=&inidxs[nn_bg];
    INT_TYPE* ocidxs_p=&ocidxs[nn_bg-pi];
    for(int ni=0;ni<nn_size;ni++)
    {
        if(inidxs_p[ni]==pi) continue;
        (*onidxs_p)=inidxs_p[ni];
        (*ocidxs_p)=pi;
        onidxs_p++;
        ocidxs_p++;
    }
}

template<typename INT_TYPE>
void eliminateCenterGPU(
        INT_TYPE *inidxs,                   // [n]
        INT_TYPE *inidxs_lens,              // [pn]
        INT_TYPE *inidxs_bgs,               // [pn]
        INT_TYPE pn,
        INT_TYPE *onidxs,                   // [n-pn]
        INT_TYPE *onidxs_lens,              // [pn]
        INT_TYPE *onidxs_bgs,               // [pn]
        INT_TYPE *ocidxs                    // [n-pn]

)
{
    int block_num=pn/1024;
    if(pn%1024>0) block_num++;
    dim3 block_dim(block_num);
    dim3 thread_dim(1024);
    eliminateCenter<INT_TYPE> <<<block_dim,thread_dim>>>
              (inidxs,inidxs_lens,inidxs_bgs,pn,onidxs,onidxs_lens,onidxs_bgs,ocidxs);
    gpuErrchk(cudaGetLastError())
}

template void neighborSumFeatGatherGPU<float,unsigned int>
        (float*, unsigned int*,unsigned int*,unsigned int,unsigned int,unsigned int,float*);
template void neighborSumFeatScatterGPU<float,unsigned int>
        (float*, unsigned int*,unsigned int,unsigned int,unsigned int,float*);
template void neighborMaxFeatGatherGPU<float,unsigned int>
        (float*, unsigned int*,unsigned int*,unsigned int,unsigned int,float*,unsigned int*);
template void neighborMaxFeatScatterGPU<float,unsigned int>
        (float*, unsigned int*,unsigned int*,unsigned int*,unsigned int,unsigned int,unsigned int,float*);

template void concatNonCenterFeatScatterGPU<float,unsigned int>
        (float*, unsigned int*,unsigned int*,unsigned int*,unsigned int,unsigned int,float*);
template void concatNonCenterFeatGatherGPU<float,unsigned int>
        (float*, unsigned int*,unsigned int*,unsigned int*,unsigned int,unsigned int,float*);

template void eliminateCenterGPU<unsigned int>
        (unsigned int*,unsigned int*,unsigned int*,unsigned int,unsigned int*,unsigned int*,unsigned int*,unsigned int*);
