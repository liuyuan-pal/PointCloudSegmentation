import tensorflow as tf
import os
path = os.path.split(os.path.realpath(__file__))[0]
pooling_ops=tf.load_op_library(path+'/build/PoolingOps.so')
import sys
sys.path.append(path)
from graph_conv_layer import *

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

def search_neighborhood_range(xyzs,min_radius,max_radius):
    idxs,lens,begs,cens=pooling_ops.search_neighborhood_brute_force_range(xyzs,squared_min_nn_size=min_radius*min_radius,
                                                                          squared_max_nn_size=max_radius*max_radius)
    return idxs,lens,begs,cens

def search_neighborhood_fixed(xyzs,radius,fixed_size=10):
    idxs,lens,begs,cens=pooling_ops.search_neighborhood_fixed_brute_force(xyzs,squared_nn_size=radius*radius,fixed_size=fixed_size)
    return idxs,lens,begs,cens

def search_neighborhood_fixed_range(xyzs,min_radius,max_radius,fixed_size):
    idxs,lens,begs,cens=pooling_ops.search_neighborhood_fixed_brute_force_range(
        xyzs,squared_min_nn_size=min_radius*min_radius,squared_max_nn_size=max_radius*max_radius,fixed_size=fixed_size)
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


def average_downsample(xyzs, feats, ds_size=0.2, min_coordinate=3.0):
    # voxel idxs
    voxel_idxs=pooling_ops.compute_voxel_index(xyzs, voxel_len=ds_size, block_size=min_coordinate * 2)

    # permutation info
    origin2permutation_idxs,permutation2origin_idxs,voxel_idxs_lens,voxel_idxs_begs,voxel_idxs_cens\
        =pooling_ops.compute_permutation_info(voxel_idxs)

    # permutated features
    permutated_pts=pooling_ops.permutate_feature(xyzs,origin2permutation_idxs,permutation2origin_idxs)
    permutated_feats=pooling_ops.permutate_feature(feats,origin2permutation_idxs,permutation2origin_idxs)

    # compute center coordinates
    center_pts=neighbor_ops.neighbor_sum_feat_gather(permutated_pts,voxel_idxs_cens,voxel_idxs_lens,voxel_idxs_begs) # [vn,3]
    center_pts=center_pts/tf.expand_dims(tf.cast(voxel_idxs_lens,tf.float32),axis=1)
    center_feats=neighbor_ops.neighbor_sum_feat_gather(permutated_feats,voxel_idxs_cens,voxel_idxs_lens,voxel_idxs_begs) # [vn,f]
    center_feats=center_feats/tf.expand_dims(tf.cast(voxel_idxs_lens,tf.float32),axis=1)

    return center_pts,center_feats

def context_points_pooling(xyzs,feats,ctx_idxs,voxel_size=0.2,block_size=3.0):
    # voxel idxs
    voxel_idxs=pooling_ops.compute_voxel_index(xyzs,voxel_len=voxel_size,block_size=block_size)

    # permutation info
    origin2permutation_idxs,permutation2origin_idxs,voxel_idxs_lens,voxel_idxs_begs,voxel_idxs_cens\
        =pooling_ops.compute_permutation_info(voxel_idxs)

    # permutated features
    permutated_pts=pooling_ops.permutate_feature(xyzs,origin2permutation_idxs,permutation2origin_idxs)
    permutated_feats=pooling_ops.permutate_feature(feats,origin2permutation_idxs,permutation2origin_idxs)
    ctx_idxs=tf.gather(permutation2origin_idxs,ctx_idxs)

    # compute center coordinates
    center_pts=neighbor_ops.neighbor_sum_feat_gather(permutated_pts,voxel_idxs_cens,voxel_idxs_lens,voxel_idxs_begs) # [vn,3]
    center_pts=center_pts/tf.expand_dims(tf.cast(voxel_idxs_lens,tf.float32),axis=1)

    # diff between center and points inside
    diff_pts = pooling_ops.compute_diff_xyz(permutated_pts, center_pts, voxel_idxs_cens)

    return permutated_pts,center_pts,diff_pts,permutated_feats,\
           voxel_idxs_lens,voxel_idxs_begs,voxel_idxs_cens,ctx_idxs

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


def context_points_pooling_two_layers(xyzs,feats,labels,ctx_idxs,voxel_size1,voxel_size2,block_size=3.0):
    labels=tf.expand_dims(tf.cast(labels,tf.int32),axis=1)
    ctx_idxs=tf.expand_dims(ctx_idxs,axis=1)

    # permutation 1
    vidxs1 = pooling_ops.compute_voxel_index(xyzs, voxel_len=voxel_size1, block_size=block_size)
    o2p_idxs1, p2o_idxs1, vlens1, vbegs1, vcens1 = pooling_ops.compute_permutation_info(vidxs1)

    pts1 = pooling_ops.permutate_feature(xyzs, o2p_idxs1, p2o_idxs1)     # pn1
    feats = pooling_ops.permutate_feature(feats, o2p_idxs1, p2o_idxs1)
    labels = pooling_ops.permutate_feature(labels, o2p_idxs1, p2o_idxs1)
    ctx_idxs = pooling_ops.permutate_feature(ctx_idxs, o2p_idxs1, p2o_idxs1)

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
    ctx_idxs = pooling_ops.permutate_feature(ctx_idxs, reper_o2p_idxs1, reper_p2o_idxs1)

    labels=tf.cast(tf.squeeze(labels,axis=1),tf.int64)
    ctx_idxs=tf.squeeze(ctx_idxs,axis=1)

    return [pts1,pts2,pts3],[dpts1,dpts2],feats,labels,[vlens1,vlens2],[vbegs1,vbegs2],[vcens1,vcens2],ctx_idxs


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



def points_pooling_three_layers(xyzs,feats,labels,voxel_size1,voxel_size2,voxel_size3,block_size=3.0):
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

    # permutation_3
    vidxs3 = pooling_ops.compute_voxel_index(pts3, voxel_len=voxel_size3, block_size=block_size)
    o2p_idxs3, p2o_idxs3, vlens3, vbegs3, vcens3 = pooling_ops.compute_permutation_info(vidxs3)
    pts3 = pooling_ops.permutate_feature(pts3, o2p_idxs3, p2o_idxs3)  # pn2

    # compute center 3
    pts4 = neighbor_ops.neighbor_sum_feat_gather(pts3, vcens3, vlens3, vbegs3)  # pn2
    pts4 = pts4 / tf.expand_dims(tf.cast(vlens3, tf.float32), axis=1)
    dpts3 = pooling_ops.compute_diff_xyz(pts3, pts4, vcens3)

    # repermutate 2
    reper_o2p_idxs2, reper_p2o_idxs2, vlens2, vbegs2, vcens2 = \
        pooling_ops.compute_repermutation_info(o2p_idxs3, vlens2, vbegs2, vcens2)
    pts2 = pooling_ops.permutate_feature(pts2, reper_o2p_idxs2, reper_p2o_idxs2)
    dpts2 = pooling_ops.permutate_feature(dpts2, reper_o2p_idxs2, reper_p2o_idxs2)

    # repermutate 1
    reper_o2p_idxs1, reper_p2o_idxs1, vlens1, vbegs1, vcens1 = \
        pooling_ops.compute_repermutation_info(reper_o2p_idxs2, vlens1, vbegs1, vcens1)
    pts1 = pooling_ops.permutate_feature(pts1, reper_o2p_idxs1, reper_p2o_idxs1)
    dpts1 = pooling_ops.permutate_feature(dpts1, reper_o2p_idxs1, reper_p2o_idxs1)
    feats = pooling_ops.permutate_feature(feats, reper_o2p_idxs1, reper_p2o_idxs1)
    labels = pooling_ops.permutate_feature(labels, reper_o2p_idxs1, reper_p2o_idxs1)

    labels=tf.cast(tf.squeeze(labels,axis=1),tf.int64)

    return [pts1,pts2,pts3,pts4],[dpts1,dpts2,dpts3],feats,labels,\
           [vlens1,vlens2,vlens3],[vbegs1,vbegs2,vbegs3],[vcens1,vcens2,vcens3]

def permutate_feats(feats,o2p,p2o):
    return pooling_ops.permutate_feature(feats,o2p,p2o)

def voxel_downsample(xyzs,feats,labels,voxel_size,block_size):
    # voxel idxs
    voxel_idxs=pooling_ops.compute_voxel_index(xyzs,voxel_len=voxel_size,block_size=block_size)

    # permutation info
    o2p_idxs,p2o_idxs,vlens,vidxs,vcens=pooling_ops.compute_permutation_info(voxel_idxs)

    # permutated features
    xyzs=pooling_ops.permutate_feature(xyzs,o2p_idxs,p2o_idxs)
    feats=pooling_ops.permutate_feature(feats,o2p_idxs,p2o_idxs)

    labels=tf.expand_dims(labels,axis=1)
    labels=pooling_ops.permutate_feature(labels,o2p_idxs,p2o_idxs)
    labels=tf.squeeze(labels,axis=1)

    # compute center coordinates
    ds_xyzs=neighbor_ops.neighbor_sum_feat_gather(xyzs,vcens,vlens,vidxs) # [vn,3]
    ds_xyzs=ds_xyzs/tf.expand_dims(tf.cast(vlens,tf.float32),axis=1)

    # diff between center and points inside
    dxyzs = pooling_ops.compute_diff_xyz(xyzs, ds_xyzs, vcens)

    return permutated_pts,center_pts,diff_pts,permutated_feats,\
           permutated_labels,vlens,vidxs,vcens


