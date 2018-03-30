
project(tf_ops C CXX)
include_directories(/usr/local/lib/python2.7/dist-packages/tensorflow/include)
set(SOURCE_FILES ComputeVoxelIdx.cu ComputeVoxelIdx.cc ComputePermutationInfo.cc
        ComputePermutationInfo.cpp PermutateFeature.cu PermutateFeature.cc ComputeDiffXYZ.cu ComputeDiffXYZ.cc
        SearchNeighborhood.cu SearchNeighborhood.cc)
add_library(pooling_index SHARED ${SOURCE_FILES})