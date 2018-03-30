#!/usr/bin/env bash

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

/usr/local/cuda/bin/nvcc PoolingIndex/ComputeDiffXYZ.cu -o build/ComputeDiffXYZ.cu.o -c -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
/usr/local/cuda/bin/nvcc PoolingIndex/ComputeVoxelIdx.cu -o build/ComputeVoxelIdx.cu.o -c -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
/usr/local/cuda/bin/nvcc PoolingIndex/PermutateFeature.cu -o build/PermutateFeature.cu.o -c -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
/usr/local/cuda/bin/nvcc PoolingIndex/SearchNeighborhood.cu -o build/SearchNeighborhood.cu.o -c -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2


cu_lib="build/ComputeDiffXYZ.cu.o
        build/ComputeVoxelIdx.cu.o
        build/PermutateFeature.cu.o
        build/SearchNeighborhood.cu.o"

cc_file="ComputeDiffXYZ.cc
        ComputeVoxelIdx.cc
        PermutateFeature.cc
        SearchNeighborhood.cc
        ComputePermutationInfo.cpp
        ComputePermutationInfo.cc
        "

g++ -std=c++11 -shared ${cu_lib} ${cc_file} -o build/PoolingOps.so \
         -fPIC -I$TF_INC -I$TF_INC/external/nsync/public \
         -L$TF_LIB -ltensorflow_framework \
         -D_GLIBCXX_USE_CXX11_ABI=0 -O2