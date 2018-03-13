import tensorflow as tf
import os
path = os.path.split(os.path.realpath(__file__))[0]
neighbor_ops=tf.load_op_library(path+'/build/libTFNeighborOps.so')
import sys
sys.path.append(path)

from tensorflow.python.framework import ops

@ops.RegisterGradient("NeighborScatter")
def _neighbor_scatter_gradient(op,dsfeats):
    use_diff=op.get_attr('use_diff')
    difeats=neighbor_ops.neighbor_gather(dsfeats, op.inputs[1], op.inputs[2], op.inputs[3], use_diff=use_diff)
    return [difeats,None,None,None]


@ops.RegisterGradient("LocationWeightFeatSum")
def _location_weight_feat_sum_gradient(op,dtfeats_sum):
    dlw,dfeats=neighbor_ops.location_weight_feat_sum_backward(op.inputs[0], op.inputs[1], dtfeats_sum, op.inputs[2], op.inputs[3])
    return [dlw,dfeats,None,None]


@ops.RegisterGradient("NeighborSumFeatGather")
def _neighbor_sum_feat_gather_gradient(op, dgfeats):
    difeats=neighbor_ops.neighbor_sum_feat_scatter(dgfeats, op.inputs[1], op.inputs[2], op.inputs[3])
    return [difeats,None,None,None]


@ops.RegisterGradient("NeighborSumFeatScatter")
def _neighbor_sum_feat_scatter_gradient(op,dsfeats):
    difeats=neighbor_ops.neighbor_sum_feat_gather(dsfeats, op.inputs[1], op.inputs[2], op.inputs[3])
    return [difeats,None,None,None]


@ops.RegisterGradient("LocationWeightSum")
def _location_weight_feat_sum_gradient(op,dlw_sum):
    dlw=neighbor_ops.location_weight_sum_backward(op.inputs[0], dlw_sum, op.inputs[1], op.inputs[2])
    return [dlw,None,None]


@ops.RegisterGradient("NeighborMaxFeatGather")
def _neighbor_max_feat_gather_gradient(op,dgfeats,dmax_idxs):
    difeats=neighbor_ops.neighbor_max_feat_scatter(dgfeats,op.inputs[0],op.outputs[1],op.inputs[2])
    return [difeats,None,None]


@ops.RegisterGradient("NeighborConcatNonCenterScatter")
def _neighbor_concat_non_center_scatter_gradient(op,dsfeats):
    difeats=neighbor_ops.neighbor_concat_non_center_gather(dsfeats, op.inputs[1], op.inputs[2], op.inputs[3])
    return [difeats,None,None,None]


def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        # tf.add_to_collection(tf.GraphKeys.TRAINABLE_VARIABLES,var)
    return var


def compute_pfeats(lw,lw_sum,tfeats,nidxs_lens,nidxs_bgs,cidxs):
    '''

    :param lw:          [csum,m]
    :param lw_sum:      [pn,m]
    :param tfeats:      [csum,m,ofn]
    :param nidxs_lens:  [pn]
    :param nidxs_bgs:   [pn]
    :param cidxs:       [csum]
    :return: pfeats [pn,ofn]
    '''

    lw_exp=tf.expand_dims(lw,axis=2)                                        #[csum,m,1]
    wtfeats=lw_exp*tfeats                                                   #[csum,m,ofn]
    wtfeats=tf.reshape(wtfeats,[tf.shape(wtfeats)[0],-1])                   #[csum,m*ofn]
    tfeats_sum=neighbor_ops.neighbor_sum_feat_gather(wtfeats, cidxs, nidxs_lens, nidxs_bgs)
    tfeats_sum=tf.reshape(tfeats_sum,[-1,tf.shape(tfeats)[1],tf.shape(tfeats)[2]])
    # tfeats_sum=neighbor_ops.location_weight_feat_sum(lw,tfeats,nidxs_lens,nidxs_bgs)

    inv_lw_sum=1.0/(lw_sum+1e-6)
    inv_lw_sum_exp=tf.expand_dims(inv_lw_sum,axis=2)
    pfeats=tfeats_sum*inv_lw_sum_exp
    pfeats=tf.reduce_sum(pfeats,axis=1)
    return pfeats


def graph_conv_xyz_feats_impl(xyz,feats,cidxs,nidxs,nidxs_lens,nidxs_bgs,pw,
                              compute_lw=False,lw=None,lw_sum=None,pmiu=None):

    sxyz=neighbor_ops.neighbor_scatter(xyz,nidxs,nidxs_lens,nidxs_bgs,use_diff=True)
    sfeats=neighbor_ops.neighbor_scatter(feats,nidxs,nidxs_lens,nidxs_bgs,use_diff=False)
    cfeats=tf.concat([sxyz,sfeats],axis=1)          # [csum,3+ifn]
    pw_reshape=tf.reshape(pw,[tf.shape(pw)[0],-1])  # [ifn,m,ofn] -> [3+ifn,m*ofn]
    tfeats=tf.matmul(cfeats,pw_reshape)             # [csum,m*ofn]
    tfeats=tf.reshape(tfeats,[-1,tf.shape(pw)[1],tf.shape(pw)[2]]) # [csum,m,ofn]

    if compute_lw:
        assert lw is None and lw_sum is None and pmiu is not None
        lw=tf.exp(tf.matmul(sxyz,pmiu))                                     # [csum,m]
        lw_sum=neighbor_ops.location_weight_sum(lw,nidxs_lens,nidxs_bgs)    # [pn,m]
    else:
        assert lw is not None and lw_sum is not None

    pfeats=compute_pfeats(lw,lw_sum,tfeats,nidxs_lens,nidxs_bgs,cidxs)   # [pn,ofn]

    return pfeats,lw,lw_sum


def graph_conv_xyz_impl(xyz,cidxs,nidxs,nidxs_lens,nidxs_bgs,pw,
                        compute_lw=False,lw=None,lw_sum=None,pmiu=None):

    sxyz=neighbor_ops.neighbor_scatter(xyz,nidxs,nidxs_lens,nidxs_bgs,use_diff=True)
    pw_reshape=tf.reshape(pw,[tf.shape(pw)[0],-1])
    tfeats=tf.matmul(sxyz,pw_reshape)             # [csum,3]
    tfeats=tf.reshape(tfeats,[-1,tf.shape(pw)[1],tf.shape(pw)[2]])

    if compute_lw:
        assert lw is None and lw_sum is None and pmiu is not None
        lw=tf.exp(tf.matmul(sxyz,pmiu))           # [csum,m]
        lw_sum=neighbor_ops.location_weight_sum(lw,nidxs_lens,nidxs_bgs)
    else:
        assert lw is not None and lw_sum is not None

    pfeats=compute_pfeats(lw,lw_sum,tfeats,nidxs_lens,nidxs_bgs,cidxs)

    return pfeats,lw,lw_sum


def graph_conv_feats_impl(feats,cidxs,nidxs,nidxs_lens,nidxs_bgs,pw,lw,lw_sum):

    pw_reshape=tf.reshape(pw,[tf.shape(pw)[0],-1])  # [ifn,m*ofn]
    tfeats=tf.matmul(feats,pw_reshape)              # [pn,m*ofn]
    tfeats=neighbor_ops.neighbor_scatter(tfeats,nidxs,nidxs_lens,nidxs_bgs,use_diff=False)
    tfeats=tf.reshape(tfeats,[-1,tf.shape(pw)[1],tf.shape(pw)[2]]) # [csum,m,ofn]

    pfeats=compute_pfeats(lw,lw_sum,tfeats,nidxs_lens,nidxs_bgs,cidxs)

    return pfeats


def graph_conv_xyz_feats(xyz, feats, cidxs, nidxs, nidxs_lens, nidxs_bgs, name, ifn, m, ofn,
                         compute_lw=False, lw=None, lw_sum=None, pmiu=None,
                         use_bias=True,activation_fn=tf.nn.relu,
                         initializer=tf.contrib.layers.xavier_initializer(),reuse=None):

    with tf.variable_scope(name,reuse=reuse):
        pw=_variable_on_cpu('pw', [ifn,m,ofn], initializer)
        if compute_lw and pmiu is None:
                pmiu=_variable_on_cpu('pmiu', [3,m], initializer)

        if use_bias:
            bias=_variable_on_cpu('bias',[ofn],tf.zeros_initializer())

    with tf.name_scope(name):
        pfeats,lw_,lw_sum_=graph_conv_xyz_feats_impl(xyz,feats,cidxs,nidxs,nidxs_lens,nidxs_bgs,pw,compute_lw,lw,lw_sum,pmiu)

        if use_bias:
            pfeats=tf.add(pfeats,tf.expand_dims(bias,axis=0))

        if activation_fn is not None:
            pfeats=activation_fn(pfeats)

        if compute_lw:
            lw=lw_
            lw_sum=lw_sum_
            return pfeats,lw,lw_sum
        else:
            return pfeats


def graph_conv_xyz(xyz, cidxs, nidxs, nidxs_lens, nidxs_bgs, name, ifn, m, ofn,
                   compute_lw=False, lw=None, lw_sum=None, pmiu=None,
                   use_bias=True,activation_fn=tf.nn.relu,
                   initializer=tf.contrib.layers.xavier_initializer(),reuse=None):
    with tf.variable_scope(name,reuse=reuse):
        pw=_variable_on_cpu('pw', [ifn,m,ofn], initializer)
        if compute_lw and pmiu is None:
                pmiu=_variable_on_cpu('pmiu', [3,m], initializer)

        if use_bias:
            bias=_variable_on_cpu('bias',[ofn],tf.zeros_initializer())

    with tf.name_scope(name):
        pfeats,lw_,lw_sum_=graph_conv_xyz_impl(xyz,cidxs,nidxs,nidxs_lens,nidxs_bgs,pw,compute_lw,lw,lw_sum,pmiu)

        if use_bias:
            pfeats=tf.add(pfeats,tf.expand_dims(bias,axis=0))

        if activation_fn is not None:
            pfeats=activation_fn(pfeats)
            pfeats=tf.reshape(pfeats,[-1,ofn])

        if compute_lw:
            lw=lw_
            lw_sum=lw_sum_
            return pfeats,lw,lw_sum
        else:
            return pfeats


def graph_conv_feats(feats, cidxs, nidxs, nidxs_lens, nidxs_bgs, name, ifn, m, ofn, lw, lw_sum,
                     use_bias=True,activation_fn=tf.nn.relu,
                     initializer=tf.contrib.layers.xavier_initializer(),
                     reuse=None):

    with tf.variable_scope(name,reuse=reuse):
        pw=_variable_on_cpu('pw', [ifn,m,ofn], initializer)
        if use_bias:
            bias=_variable_on_cpu('bias',[ofn],tf.zeros_initializer())

    with tf.name_scope(name):
        pfeats=graph_conv_feats_impl(feats,cidxs,nidxs,nidxs_lens,nidxs_bgs,pw,lw,lw_sum)

        if use_bias:
            pfeats=tf.add(pfeats,tf.expand_dims(bias,axis=0))

        if activation_fn is not None:
            pfeats=activation_fn(pfeats)

    return pfeats


def graph_pool(feats,vlens,vlens_bgs):
    pool_feats,max_idxs=neighbor_ops.neighbor_max_feat_gather(feats,vlens,vlens_bgs)
    return pool_feats


def graph_unpool(feats,vlens,vlens_bgs,cidxs):
    unpool_feats=neighbor_ops.neighbor_sum_feat_scatter(feats,cidxs,vlens,vlens_bgs)
    return unpool_feats


def graph_concat_non_center_scatter(ifeats,nidxs,nidxs_lens,nidxs_bgs):
    return neighbor_ops.neighbor_concat_non_center_scatter(ifeats,nidxs,nidxs_lens,nidxs_bgs)