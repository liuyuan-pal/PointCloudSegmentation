import tensorflow as tf
import sys
from tensorflow.python.framework import ops
import os
# get the file path
neighbor_ops=tf.load_op_library(os.path.split(os.path.realpath(__file__))[0]+"/build/libTFNeighborBackwardOps.so")

@ops.RegisterGradient("NeighborScatter")
def _neighbor_scatter_gradient(op,dsfeats):
    use_diff=op.get_attr('use_diff')
    difeats=neighbor_ops.neighbor_gather(dsfeats, op.inputs[1], op.inputs[2], op.inputs[3], use_diff=use_diff)
    return [difeats,None,None,None]


@ops.RegisterGradient("LocationWeightFeatSum")
def _location_weight_feat_sum_gradient(op,dtfeats_sum):
    dlw,dfeats=neighbor_ops.location_weight_feat_sum_backward(op.inputs[0], op.inputs[1], dtfeats_sum, op.inputs[2], op.inputs[3])
    return [dlw,dfeats,None,None]


@ops.RegisterGradient("LocationWeightFeatSumV2")
def _location_weight_feat_sum_v2_gradient(op,dtfeats_sum):
    dlw,dfeats=neighbor_ops.location_weight_feat_sum_backward_v2(op.inputs[0], op.inputs[1], dtfeats_sum, op.inputs[2])
    return [dlw,dfeats,None,None]


@ops.RegisterGradient("LocationWeightSum")
def _location_weight_feat_sum_gradient(op,dlw_sum):
    dlw=neighbor_ops.location_weight_sum_backward(op.inputs[0], dlw_sum, op.inputs[1], op.inputs[2])
    return [dlw,None,None]
