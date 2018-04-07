//
// Created by pal on 18-3-1.
//

#ifndef POINTUTIL_TFCUDACOMMON_H
#define POINTUTIL_TFCUDACOMMON_H

#include <cstring>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

inline int infTwoExp(int val)
{
    int inf=1;
    while(val>inf) inf<<=1;
    return inf;
}

#endif //POINTUTIL_TFCUDACOMMON_H
