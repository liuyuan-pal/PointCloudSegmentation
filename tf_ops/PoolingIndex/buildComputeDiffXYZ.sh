#!/usr/bin/env bash
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

/usr/local/cuda/bin/nvcc ComputeDiffXYZ.cu -o ComputeDiffXYZ.cu.o -c -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

g++ -std=c++11 -shared ComputeDiffXYZ.cu.o \
         ComputeDiffXYZ.cc -o ComputeDiffXYZOp.so \
         -fPIC -I$TF_INC -I$TF_INC/external/nsync/public \
         -L$TF_LIB -ltensorflow_framework \
         -D_GLIBCXX_USE_CXX11_ABI=0 -O2