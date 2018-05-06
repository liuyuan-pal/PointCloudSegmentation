#include "../TFCudaCommon.h"
#include <thrust/device_vector.h>


__global__
void computeVoxelLabelKernel(
        int *point_label,
        int *vlens,
        int *vbegs,
        int *voxel_label,
        int *label_count,
        int vn,
        int cn
)
{
    int vi = threadIdx.x + blockIdx.x * blockDim.x;
    if (vi >= vn)
        return;

    int nn=vlens[vi];
    int *point_label_ptr=&point_label[vbegs[vi]];
    int *label_count_ptr=&label_count[vi];
    for (int ni=0;ni<nn;ni++)
        label_count_ptr[point_label_ptr[ni]]++;

    int max_label_count=0;
    int max_label=-1;
    for(int ci=0;ci<cn;ci++)
    {
        if(label_count_ptr[ci]>max_label_count)
        {
            max_label=ci;
            max_label_count=label_count_ptr[ci];
        }
    }

    voxel_label[vi]=max_label;
}

void computeVoxelLabelImpl(
        int *point_label,
        int *vlens,
        int *vbegs,
        int *voxel_label,
        int vn,
        int cn
)
{
    thrust::device_vector<int> label_count(vn*cn);
    thrust::fill(label_count.begin(),label_count.end(),0);


    int block_num = vn / 1024;
    if (vn % 1024 > 0) block_num++;
    dim3 block_dim(block_num);
    dim3 thread_dim(1024);
    computeVoxelLabelKernel<<<block_dim,thread_dim>>>(
         point_label,vlens,vbegs,voxel_label,thrust::raw_pointer_cast(label_count.data()),vn,cn);
    gpuErrchk(cudaGetLastError())
}
