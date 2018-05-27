#include "../TFCudaCommon.h"
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/fill.h>

__global__
void computeNeighborIdxsFixedKernel(
        float * xyzs,               // [pn,3]
        int *idxs,                  // [en]
        int *cens,                  // [en]
        float squared_nn_size,
        int fixed_size,
        int pn
)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    if(pi>=pn) return;

    float cx=xyzs[pi*3+0];
    float cy=xyzs[pi*3+1];
    float cz=xyzs[pi*3+2];
    int* cur_idxs=&idxs[pi*fixed_size];
    int* cur_cens=&cens[pi*fixed_size];
    int counts=0;
    for(int i=0;i<pn&&counts<fixed_size;i++)
    {
        float tx=xyzs[i*3+0];
        float ty=xyzs[i*3+1];
        float tz=xyzs[i*3+2];
        if((tx-cx)*(tx-cx)+(ty-cy)*(ty-cy)+(tz-cz)*(tz-cz)<squared_nn_size)
        {
            *cur_idxs=i;
            cur_idxs++;
            *cur_cens=pi;
            cur_cens++;
            counts++;
        }
    }

    int nn_idx;
    if(counts==0)
        nn_idx=pi;
    else
        nn_idx=*(cur_idxs-1);
    for(int i=counts;i<fixed_size;i++)
    {
        *cur_idxs=nn_idx;
        *cur_cens=pi;
        cur_cens++;
        cur_idxs++;
    }
}
__global__
void computeNeighborIdxsFixedRangeKernel(
        float * xyzs,               // [pn,3]
        int *idxs,                  // [en]
        int *cens,                  // [en]
        float squared_min_nn_size,
        float squared_max_nn_size,
        int fixed_size,
        int pn
)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    if(pi>=pn) return;

    float cx=xyzs[pi*3+0];
    float cy=xyzs[pi*3+1];
    float cz=xyzs[pi*3+2];
    int* cur_idxs=&idxs[pi*fixed_size];
    int* cur_cens=&cens[pi*fixed_size];
    int counts=0;
    for(int i=0;i<pn&&counts<fixed_size;i++)
    {
        float tx=xyzs[i*3+0];
        float ty=xyzs[i*3+1];
        float tz=xyzs[i*3+2];
        float sq_dist=(tx-cx)*(tx-cx)+(ty-cy)*(ty-cy)+(tz-cz)*(tz-cz);
        if(sq_dist<squared_max_nn_size&&sq_dist>squared_min_nn_size)
        {
            *cur_idxs=i;
            cur_idxs++;
            *cur_cens=pi;
            cur_cens++;
            counts++;
        }
    }

    int nn_idx;
    if(counts==0)
        nn_idx=pi;
    else
        nn_idx=*(cur_idxs-1);
    for(int i=counts;i<fixed_size;i++)
    {
        *cur_idxs=nn_idx;
        cur_idxs++;
        *cur_cens=pi;
        cur_cens++;

    }
}


void searchNeighborhoodFixedImpl(
        float * xyzs,               // [pn,3]
        int *idxs,                  // [en]
        int *cens,                  // [en]
        float squared_nn_size,
        int fixed_size,
        int pn
)
{
    int block_num = pn / 1024;
    if (pn % 1024 > 0) block_num++;
    dim3 block_dim(block_num);
    dim3 thread_dim(1024);
    computeNeighborIdxsFixedKernel<<<block_dim,thread_dim>>>(xyzs,idxs,cens,squared_nn_size,fixed_size,pn);
    gpuErrchk(cudaGetLastError())
}

void searchNeighborhoodFixedRangeImpl(
        float * xyzs,               // [pn,3]
        int *idxs,                  // [en]
        int *cens,                  // [en]
        float squared_min_nn_size,
        float squared_max_nn_size,
        int fixed_size,
        int pn
)
{
    int block_num = pn / 1024;
    if (pn % 1024 > 0) block_num++;
    dim3 block_dim(block_num);
    dim3 thread_dim(1024);
    computeNeighborIdxsFixedRangeKernel<<<block_dim,thread_dim>>>(xyzs,idxs,cens,squared_min_nn_size,squared_max_nn_size,fixed_size,pn);
    gpuErrchk(cudaGetLastError())
}

void computeNeighborhoodStats(
        int *lens,                  // [en]
        int *begs,                  // [en]
        int fixed_size,
        int pn
)
{
    thrust::device_ptr<int> lens_ptr(lens);
    thrust::device_ptr<int> begs_ptr(begs);

    thrust::fill(lens_ptr,lens_ptr+pn,fixed_size);
    thrust::exclusive_scan(lens_ptr,lens_ptr+pn,begs_ptr);
}