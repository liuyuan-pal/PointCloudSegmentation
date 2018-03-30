#include "../TFCudaCommon.h"
#include <thrust/device_vector.h>
#include <thrust/scan.h>

__global__
void countNeighborNumKernel(
        float * xyzs,               // [pn,3]
        int *lens,                  // [pn]
        float squared_nn_size,
        int pn
)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    if(pi>=pn) return;
    float cx=xyzs[pi*3+0];
    float cy=xyzs[pi*3+1];
    float cz=xyzs[pi*3+2];
    int count=0;
    for(int i=0;i<pn;i++)
    {
        float tx=xyzs[i*3+0];
        float ty=xyzs[i*3+1];
        float tz=xyzs[i*3+2];
        if((tx-cx)*(tx-cx)+(ty-cy)*(ty-cy)+(tz-cz)*(tz-cz)<squared_nn_size)
            count++;
    }
    lens[pi]=count;
}

__global__
void computeNeighborIdxsKernel(
        float * xyzs,               // [pn,3]
        int *begs,                  // [pn]
        int *idxs,                  // [en]
        float squared_nn_size,
        int pn
)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    if(pi>=pn) return;

    float cx=xyzs[pi*3+0];
    float cy=xyzs[pi*3+1];
    float cz=xyzs[pi*3+2];
    int* cur_idxs=&idxs[begs[pi]];
    for(int i=0;i<pn;i++)
    {
        float tx=xyzs[i*3+0];
        float ty=xyzs[i*3+1];
        float tz=xyzs[i*3+2];
        if((tx-cx)*(tx-cx)+(ty-cy)*(ty-cy)+(tz-cz)*(tz-cz)<squared_nn_size)
        {
            *cur_idxs=i;
            cur_idxs++;
        }
    }
}


int searchNeighborhoodCountImpl(
        float * xyzs,               // [pn,3]
        int *lens,                  // [pn]
        int *begs,                  // [pn]
        float squared_nn_size,
        int pn
)
{
    int block_num = pn / 1024;
    if (pn % 1024 > 0) block_num++;
    dim3 block_dim(block_num);
    dim3 thread_dim(1024);
    countNeighborNumKernel<<<block_dim,thread_dim>>>(xyzs,lens,squared_nn_size,pn);

    thrust::device_ptr<int> len_ptr(lens);
    thrust::device_ptr<int> beg_ptr(begs);
    thrust::exclusive_scan(len_ptr,len_ptr+pn,beg_ptr);

    return *(beg_ptr+pn-1)+*(len_ptr+pn-1);
}

void searchNeighborhoodImpl(
        float * xyzs,               // [pn,3]
        int *idxs,                  // [pn]
        int *begs,                  // [pn]
        float squared_nn_size,
        int pn
)
{
    int block_num = pn / 1024;
    if (pn % 1024 > 0) block_num++;
    dim3 block_dim(block_num);
    dim3 thread_dim(1024);
    computeNeighborIdxsKernel<<<block_dim,thread_dim>>>(xyzs,begs,idxs,squared_nn_size,pn);
}

__global__
void countNeighborNumWithBinsKernel(
        float * xyzs,               // [pn,3]
        int *lens,                  // [pn]
        int *bin_idxs,              // [pn,3]
        int bin_thresh,
        float squared_nn_size,
        int pn
)
{
    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    if(pi>=pn) return;
    float cx=xyzs[pi*3+0];
    float cy=xyzs[pi*3+1];
    float cz=xyzs[pi*3+2];

    int cbx=bin_idxs[pi*3+0];
    int cby=bin_idxs[pi*3+1];
    int cbz=bin_idxs[pi*3+2];

    int count=0;
    for(int i=0;i<pn;i++)
    {
        int tbx=bin_idxs[pi*3+0];
        int tby=bin_idxs[pi*3+1];
        int tbz=bin_idxs[pi*3+2];
        if(abs(tby-cby)>bin_thresh) continue;
        if(abs(tbx-cbx)>bin_thresh) continue;
        if(abs(tbz-cbz)>bin_thresh) continue;


        float tx=xyzs[i*3+0];
        float ty=xyzs[i*3+1];
        float tz=xyzs[i*3+2];
        if((tx-cx)*(tx-cx)+(ty-cy)*(ty-cy)+(tz-cz)*(tz-cz)<squared_nn_size)
            count++;
    }
    lens[pi]=count;
}

__global__
void computeNeighborIdxsWithBinsKernel(
        float * xyzs,               // [pn,3]
        int *begs,                  // [pn]
        int *idxs,                  // [en]
        int *bin_idxs,              // [pn,3]
        int bin_thresh,
        float squared_nn_size,
        int pn
)
{

    int pi = threadIdx.x + blockIdx.x*blockDim.x;
    if(pi>=pn) return;

    float cx=xyzs[pi*3+0];
    float cy=xyzs[pi*3+1];
    float cz=xyzs[pi*3+2];

    int cbx=bin_idxs[pi*3+0];
    int cby=bin_idxs[pi*3+1];
    int cbz=bin_idxs[pi*3+2];

    int* cur_idxs=&idxs[begs[pi]];
    for(int i=0;i<pn;i++)
    {
        int tbx=bin_idxs[pi*3+0];
        if(abs(tbx-cbx)>bin_thresh) continue;
        int tby=bin_idxs[pi*3+1];
        if(abs(tby-cby)>bin_thresh) continue;
        int tbz=bin_idxs[pi*3+2];
        if(abs(tbz-cbz)>bin_thresh) continue;

        float tx=xyzs[i*3+0];
        float ty=xyzs[i*3+1];
        float tz=xyzs[i*3+2];
        if((tx-cx)*(tx-cx)+(ty-cy)*(ty-cy)+(tz-cz)*(tz-cz)<squared_nn_size)
        {
            *cur_idxs=i;
            cur_idxs++;
        }
    }
}


int searchNeighborhoodCountWithBinsImpl(
        float * xyzs,               // [pn,3]
        int *lens,                  // [pn]
        int *begs,                  // [pn]
        int *bin_idxs,              // [pn,3]
        int bin_thresh,
        float squared_nn_size,
        int pn
)
{
    int block_num = pn / 1024;
    if (pn % 1024 > 0) block_num++;
    dim3 block_dim(block_num);
    dim3 thread_dim(1024);
    countNeighborNumWithBinsKernel<<<block_dim,thread_dim>>>(xyzs,lens,bin_idxs,bin_thresh,squared_nn_size,pn);

    thrust::device_ptr<int> len_ptr(lens);
    thrust::device_ptr<int> beg_ptr(begs);
    thrust::exclusive_scan(len_ptr,len_ptr+pn,beg_ptr);

    return *(beg_ptr+pn-1)+*(len_ptr+pn-1);
}

void searchNeighborhoodWithBinsImpl(
        float * xyzs,               // [pn,3]
        int *idxs,                  // [pn]
        int *begs,                  // [pn]
        int *bin_idxs,              // [pn,3]
        int bin_thresh,
        float squared_nn_size,
        int pn
)
{
    int block_num = pn / 1024;
    if (pn % 1024 > 0) block_num++;
    dim3 block_dim(block_num);
    dim3 thread_dim(1024);
    computeNeighborIdxsWithBinsKernel<<<block_dim,thread_dim>>>(xyzs,begs,idxs,bin_idxs,bin_thresh,squared_nn_size,pn);
}