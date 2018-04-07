#!/usr/bin/env bash
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

/usr/local/cuda/bin/nvcc ComputePermutationInfo.cu -o ComputePermutationInfo.cu.o -c -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2
/usr/local/cuda/bin/nvcc ComputeRepermutationInfo.cu -o ComputeRepermutationInfo.cu.o -c -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

g++ -std=c++11 ComputeRepermutationInfo.cc ComputeRepermutationInfo.cu.o ComputePermutationInfo.cu.o \
            -shared -o ComputeRepermutationInfoOp.so \
            -fPIC -I$TF_INC -I$TF_INC/external/nsync/public \
            -L$TF_LIB -ltensorflow_framework \
            -D_GLIBCXX_USE_CXX11_ABI=0 -O2