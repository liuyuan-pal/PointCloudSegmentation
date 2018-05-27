#!/usr/bin/env bash

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

/usr/local/cuda/bin/nvcc PoolingIndex/ComputeDiffXYZ.cu -o build/ComputeDiffXYZ.cu.o -c -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
/usr/local/cuda/bin/nvcc PoolingIndex/ComputeVoxelIdx.cu -o build/ComputeVoxelIdx.cu.o -c -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
/usr/local/cuda/bin/nvcc PoolingIndex/ComputeVoxelLabel.cu -o build/ComputeVoxelLabel.cu.o -c -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
/usr/local/cuda/bin/nvcc PoolingIndex/PermutateFeature.cu -o build/PermutateFeature.cu.o -c -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
/usr/local/cuda/bin/nvcc PoolingIndex/SearchNeighborhood.cu -o build/SearchNeighborhood.cu.o -c -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
/usr/local/cuda/bin/nvcc PoolingIndex/ComputePermutationInfo.cu -o build/ComputePermutationInfo.cu.o -c -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
/usr/local/cuda/bin/nvcc PoolingIndex/ComputeRepermutationInfo.cu -o build/ComputeRepermutationInfo.cu.o -c -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
/usr/local/cuda/bin/nvcc PoolingIndex/SearchNeighborhoodFixed.cu -o build/SearchNeighborhoodFixed.cu.o -c -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2


cu_lib="build/ComputeDiffXYZ.cu.o
        build/ComputeVoxelIdx.cu.o
        build/PermutateFeature.cu.o
        build/SearchNeighborhood.cu.o
        build/ComputePermutationInfo.cu.o
        build/ComputeRepermutationInfo.cu.o
        build/SearchNeighborhoodFixed.cu.o
        "

cc_file="PoolingIndex/ComputeDiffXYZ.cc
        PoolingIndex/ComputeVoxelIdx.cc
        PoolingIndex/PermutateFeature.cc
        PoolingIndex/SearchNeighborhood.cc
        PoolingIndex/ComputePermutationInfo.cpp
        PoolingIndex/ComputePermutationInfo.cc
        PoolingIndex/ComputeRepermutationInfo.cc
        PoolingIndex/SearchNeighborhoodFixed.cc
        "

g++ -std=c++11 -shared ${cu_lib} ${cc_file} -o build/PoolingOps.so \
         -fPIC -I$TF_INC -I/home/liuyuan/lib/include -I$TF_INC/external/nsync/public \
         -L$TF_LIB -ltensorflow_framework \
         -L/usr/local/cuda/lib64/libcudart.so \
         -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -pthread