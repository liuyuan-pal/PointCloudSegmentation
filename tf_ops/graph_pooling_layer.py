import tensorflow as tf
import os
path = os.path.split(os.path.realpath(__file__))[0]
pooling_ops=tf.load_op_library(path+'/build/PoolingOps.so')
import sys
sys.path.append(path)
from graph_conv_layer import neighbor_ops

from tensorflow.python.framework import ops

@ops.RegisterGradient("PermutateFeature")
def _neighbor_scatter_gradient(op,dpemutated_feature):
    print 'here'
    dfeats=pooling_ops.neighbor_gather(dpemutated_feature, op.inputs[1])
    return [dfeats,None,None]


def search_neighborhood(xyzs,radius):
    idxs,lens,begs,cens=pooling_ops.search_neighborhood_brute_force(xyzs,squared_nn_size=radius*radius)
    return idxs,lens,begs,cens


def points_pooling(xyzs,feats,labels,voxel_size=0.2,block_size=3.0):
    # voxel idxs
    voxel_idxs=pooling_ops.compute_voxel_index(xyzs,voxel_len=voxel_size,block_size=block_size)

    # permutation info
    origin2permutation_idxs,permutation2origin_idxs,voxel_idxs_lens,voxel_idxs_begs,voxel_idxs_cens\
        =pooling_ops.compute_permutation_info(voxel_idxs)

    # permutated features
    permutated_pts=pooling_ops.permutate_feature(xyzs,origin2permutation_idxs,permutation2origin_idxs)
    permutated_feats=pooling_ops.permutate_feature(feats,origin2permutation_idxs,permutation2origin_idxs)

    labels=tf.expand_dims(labels,axis=1)
    permutated_labels=pooling_ops.permutate_feature(labels,origin2permutation_idxs,permutation2origin_idxs)
    permutated_labels=tf.squeeze(permutated_labels,axis=1)
    permutated_labels=tf.cast(permutated_labels,tf.int64)

    # compute center coordinates
    center_pts=neighbor_ops.neighbor_sum_feat_gather(permutated_pts,voxel_idxs_cens,voxel_idxs_lens,voxel_idxs_begs) # [vn,3]
    center_pts=center_pts/tf.expand_dims(tf.cast(voxel_idxs_lens,tf.float32),axis=1)

    # diff between center and points inside
    diff_pts = pooling_ops.compute_diff_xyz(permutated_pts, center_pts, voxel_idxs_cens)

    return permutated_pts,center_pts,diff_pts,permutated_feats,\
           permutated_labels,voxel_idxs_lens,voxel_idxs_begs,voxel_idxs_cens


def class_pooling(xyzs,feats,classes,labels,voxel_size=0.2,block_size=3.0):
    # voxel idxs
    voxel_idxs=pooling_ops.compute_voxel_index(xyzs,voxel_len=voxel_size,block_size=block_size)

    # permutation info
    origin2permutation_idxs,permutation2origin_idxs,voxel_idxs_lens,voxel_idxs_begs,voxel_idxs_cens\
        =pooling_ops.compute_permutation_info_with_class(voxel_idxs,classes)

    # permutated features
    permutated_pts=pooling_ops.permutate_feature(xyzs,origin2permutation_idxs,permutation2origin_idxs)
    permutated_feats=pooling_ops.permutate_feature(feats,origin2permutation_idxs,permutation2origin_idxs)

    labels=tf.expand_dims(labels,axis=1)
    permutated_labels=pooling_ops.permutate_feature(labels,origin2permutation_idxs,permutation2origin_idxs)
    permutated_labels=tf.squeeze(permutated_labels,axis=1)
    permutated_labels=tf.cast(permutated_labels,tf.int64)

    # compute center coordinates
    center_pts=neighbor_ops.neighbor_sum_feat_gather(permutated_pts,voxel_idxs_cens,voxel_idxs_lens,voxel_idxs_begs) # [vn,3]
    center_pts=center_pts/tf.expand_dims(tf.cast(voxel_idxs_lens,tf.float32),axis=1)

    # diff between center and points inside
    diff_pts = pooling_ops.compute_diff_xyz(permutated_pts, center_pts, voxel_idxs_cens)

    return permutated_pts,center_pts,diff_pts,permutated_feats,\
           permutated_labels,voxel_idxs_lens,voxel_idxs_begs,voxel_idxs_cens

