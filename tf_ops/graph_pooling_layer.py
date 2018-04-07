import tensorflow as tf
import os
path = os.path.split(os.path.realpath(__file__))[0]
pooling_ops=tf.load_op_library(path+'/build/PoolingOps.so')
import sys
sys.path.append(path)
from graph_conv_layer import neighbor_ops

from tensorflow.python.framework import ops

@ops.RegisterGradient("PermutateFeature")
def _permutate_feature_gradient(op,dpemutated_feature):
    dfeats=pooling_ops.permutate_feature_backward(dpemutated_feature, op.inputs[2])
    return [dfeats,None,None]

@ops.RegisterGradient("ComputeDiffXyz")
def _compute_diff_xyz_gradient(op,ddxyz):
    return [None,None,None]

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

def points_pooling_two_layers(xyzs,feats,labels,voxel_size1,voxel_size2,block_size=3.0):
    labels=tf.expand_dims(tf.cast(labels,tf.int32),axis=1)

    # permutation 1
    vidxs1 = pooling_ops.compute_voxel_index(xyzs, voxel_len=voxel_size1, block_size=block_size)
    o2p_idxs1, p2o_idxs1, vlens1, vbegs1, vcens1 = pooling_ops.compute_permutation_info(vidxs1)

    pts1 = pooling_ops.permutate_feature(xyzs, o2p_idxs1, p2o_idxs1)     # pn1
    feats = pooling_ops.permutate_feature(feats, o2p_idxs1, p2o_idxs1)
    labels = pooling_ops.permutate_feature(labels, o2p_idxs1, p2o_idxs1)

    # compute center 1
    pts2 = neighbor_ops.neighbor_sum_feat_gather(pts1, vcens1, vlens1, vbegs1)  # pn2
    pts2 = pts2 / tf.expand_dims(tf.cast(vlens1, tf.float32), axis=1)
    dpts1 = pooling_ops.compute_diff_xyz(pts1, pts2, vcens1)

    # permutation 2
    vidxs2 = pooling_ops.compute_voxel_index(pts2, voxel_len=voxel_size2, block_size=block_size)
    o2p_idxs2, p2o_idxs2, vlens2, vbegs2, vcens2 = pooling_ops.compute_permutation_info(vidxs2)
    pts2 = pooling_ops.permutate_feature(pts2, o2p_idxs2, p2o_idxs2)  # pn2

    # compute center 2
    pts3 = neighbor_ops.neighbor_sum_feat_gather(pts2, vcens2, vlens2, vbegs2)  # pn2
    pts3 = pts3 / tf.expand_dims(tf.cast(vlens2, tf.float32), axis=1)
    dpts2 = pooling_ops.compute_diff_xyz(pts2, pts3, vcens2)

    # repermutate 1
    reper_o2p_idxs1, reper_p2o_idxs1, vlens1, vbegs1, vcens1 = \
        pooling_ops.compute_repermutation_info(o2p_idxs2, vlens1, vbegs1, vcens1)
    pts1 = pooling_ops.permutate_feature(pts1, reper_o2p_idxs1, reper_p2o_idxs1)
    dpts1 = pooling_ops.permutate_feature(dpts1, reper_o2p_idxs1, reper_p2o_idxs1)
    feats = pooling_ops.permutate_feature(feats, reper_o2p_idxs1, reper_p2o_idxs1)
    labels = pooling_ops.permutate_feature(labels, reper_o2p_idxs1, reper_p2o_idxs1)

    labels=tf.cast(tf.squeeze(labels,axis=1),tf.int64)

    return [pts1,pts2,pts3],[dpts1,dpts2],feats,labels,[vlens1,vlens2],[vbegs1,vbegs2],[vcens1,vcens2]


def points_pooling_two_layers_tmp(xyzs,feats,labels,voxel_size1,voxel_size2,block_size=3.0):
    labels=tf.expand_dims(tf.cast(labels,tf.int32),axis=1)

    # permutation 1
    vidxs1 = pooling_ops.compute_voxel_index(xyzs, voxel_len=voxel_size1, block_size=block_size)
    o2p_idxs1, p2o_idxs1, vlens1, vbegs1, vcens1 = pooling_ops.compute_permutation_info(vidxs1)

    pts1 = pooling_ops.permutate_feature(xyzs, o2p_idxs1, p2o_idxs1)     # pn1
    feats = pooling_ops.permutate_feature(feats, o2p_idxs1, p2o_idxs1)
    labels = pooling_ops.permutate_feature(labels, o2p_idxs1, p2o_idxs1)

    # compute center 1
    pts2 = neighbor_ops.neighbor_sum_feat_gather(pts1, vcens1, vlens1, vbegs1)  # pn2
    pts2 = pts2 / tf.expand_dims(tf.cast(vlens1, tf.float32), axis=1)
    dpts1 = pooling_ops.compute_diff_xyz(pts1, pts2, vcens1)

    # permutation 2
    vidxs2 = pooling_ops.compute_voxel_index(pts2, voxel_len=voxel_size2, block_size=block_size)
    o2p_idxs2, p2o_idxs2, vlens2, vbegs2, vcens2 = pooling_ops.compute_permutation_info(vidxs2)
    pts2 = pooling_ops.permutate_feature(pts2, o2p_idxs2, p2o_idxs2)  # pn2

    # compute center 2
    pts3 = neighbor_ops.neighbor_sum_feat_gather(pts2, vcens2, vlens2, vbegs2)  # pn2
    pts3 = pts3 / tf.expand_dims(tf.cast(vlens2, tf.float32), axis=1)
    dpts2 = pooling_ops.compute_diff_xyz(pts2, pts3, vcens2)

    # repermutate 1
    reper_o2p_idxs1, reper_p2o_idxs1, vlens1, vbegs1, vcens1 = \
        pooling_ops.compute_repermutation_info(o2p_idxs2, vlens1, vbegs1, vcens1)
    pts1 = pooling_ops.permutate_feature(pts1, reper_o2p_idxs1, reper_p2o_idxs1)
    dpts1 = pooling_ops.permutate_feature(dpts1, reper_o2p_idxs1, reper_p2o_idxs1)
    feats = pooling_ops.permutate_feature(feats, reper_o2p_idxs1, reper_p2o_idxs1)
    labels = pooling_ops.permutate_feature(labels, reper_o2p_idxs1, reper_p2o_idxs1)

    labels=tf.cast(tf.squeeze(labels,axis=1),tf.int64)

    return [pts1,pts2,pts3],[dpts1,dpts2],feats,labels,[vlens1,vlens2],[vbegs1,vbegs2],[vcens1,vcens2], vidxs1, vidxs2

def class_pooling(xyzs,feats,classes,labels,voxel_size=0.2,block_size=3.0):
    # voxel idxs
    voxel_idxs=pooling_ops.compute_voxel_index(xyzs,voxel_len=voxel_size,block_size=block_size)

    # permutation info
    classes=tf.cast(classes,tf.int32)
    origin2permutation_idxs,permutation2origin_idxs,voxel_idxs_lens,voxel_idxs_begs,voxel_idxs_cens\
        =pooling_ops.compute_permutation_info_with_class(voxel_idxs,classes)

    # permutated features
    permutated_pts=pooling_ops.permutate_feature(xyzs,origin2permutation_idxs,permutation2origin_idxs)
    permutated_feats=pooling_ops.permutate_feature(feats,origin2permutation_idxs,permutation2origin_idxs)

    labels=tf.expand_dims(labels,axis=1)
    labels=tf.cast(labels,tf.int32)
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

def permutate_feats(feats,o2p,p2o):
    return pooling_ops.permutate_feature(feats,o2p,p2o)

