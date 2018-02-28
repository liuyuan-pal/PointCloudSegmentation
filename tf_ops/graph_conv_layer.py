import tensorflow as tf
import tensorflow.contrib.framework as framework
import os
path = os.path.split(os.path.realpath(__file__))[0]
neighbor_ops=tf.load_op_library(path+'/build/libTFNeighborForwardOps.so')
import sys
sys.path.append(path)
import tf_ops_backward


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


def compute_pfeats(lw,tfeats,nidxs_lens,nidxs_bgs):
    '''

    :param lw:          [pn,n,m] csum=pn*n
    :param tfeats:      [pn,n,m,ofn]
    :param nidxs_lens:  [pn]
    :param nidxs_bgs:   [pn]
    :return:
    '''
    lw_sum=neighbor_ops.location_weight_sum(lw,nidxs_lens,nidxs_bgs)
    tfeats_sum=neighbor_ops.location_weight_feat_sum(lw,tfeats,nidxs_lens,nidxs_bgs)

    lw_sum_exp=tf.tile(tf.expand_dims(lw_sum,axis=2),[1,1,tf.shape(tfeats_sum)[2]]) # [pn,m,ofn]
    pfeats=tf.div(tfeats_sum,lw_sum_exp)
    pfeats=tf.reduce_sum(pfeats,axis=1)
    return pfeats


def graph_conv_xyz_feats_impl(xyz,feats,nidxs,nidxs_lens,nidxs_bgs,pw,lw=None,pmiu=None):
    '''

    :param xyz:         [pn,3]
    :param feats:       [pn,ifn]
    :param nidxs:       [csum]
    :param nidxs_lens:  [pn]
    :param nidxs_bgs:   [pn]
    :param pw:          [ifn+3,m,ofn]
    :param lw:          [csum,m]
    :param pmiu:        [3,m]
    :return:
    '''
    sxyz=neighbor_ops.neighbor_scatter(xyz,nidxs,nidxs_lens,nidxs_bgs,use_diff=True)
    sfeats=neighbor_ops.neighbor_scatter(feats,nidxs,nidxs_lens,nidxs_bgs,use_diff=False)
    cfeats=tf.concat([sxyz,sfeats],axis=1)  # [csum,3+ifn]
    pw_reshape=tf.reshape(pw,[tf.shape(pw)[0],-1])
    tfeats=tf.matmul(cfeats,pw_reshape)     # [csum,m*ofn]
    tfeats=tf.reshape(tfeats,[-1,tf.shape(pw)[1],tf.shape(pw)[2]])

    if lw is None:
        lw=tf.exp(tf.matmul(sxyz,pmiu))           # [csum,3]

    pfeats=compute_pfeats(lw,tfeats,nidxs_lens,nidxs_bgs)

    return pfeats,lw


def graph_conv_xyz_impl(xyz,nidxs,nidxs_lens,nidxs_bgs,pw,lw=None,pmiu=None):
    '''

    :param xyz:         [pn,3]
    :param nidxs:       [csum]
    :param nidxs_lens:  [pn]
    :param nidxs_bgs:   [pn]
    :param pw:          [3,m,ofn]
    :param lw:          [csum,m]
    :param pmiu:        [3,m]
    :return:
    '''
    sxyz=neighbor_ops.neighbor_scatter(xyz,nidxs,nidxs_lens,nidxs_bgs,use_diff=True)
    pw_reshape=tf.reshape(pw,[tf.shape(pw)[0],-1])
    tfeats=tf.matmul(sxyz,pw_reshape)             # [csum,3]
    tfeats=tf.reshape(tfeats,[-1,tf.shape(pw)[1],tf.shape(pw)[2]])

    if lw is None:
        lw=tf.exp(tf.matmul(sxyz,pmiu))           # [csum,m]

    pfeats=compute_pfeats(lw,tfeats,nidxs_lens,nidxs_bgs)

    return pfeats,lw


def graph_conv_feats_impl(feats,nidxs,nidxs_lens,nidxs_bgs,pw,lw):
    '''

    :param feats:       [pn,ifn]
    :param nidxs:       [csum]
    :param nidxs_lens:  [pn]
    :param nidxs_bgs:   [pn]
    :param pw:          [ifn,m,ofn]
    :param lw:          [csum,m]
    :return:    [pn,ofn]
    '''
    pw_reshape=tf.reshape(pw,[tf.shape(pw)[0],-1])  # [ifn,m*ofn]
    tfeats=tf.matmul(feats,pw_reshape)              # [pn,m*ofn]
    tfeats=neighbor_ops.neighbor_scatter(tfeats,nidxs,nidxs_lens,nidxs_bgs,use_diff=False)
    tfeats=tf.reshape(tfeats,[-1,tf.shape(pw)[1],tf.shape(pw)[2]]) # [csum,m,ofn]
    pfeats=compute_pfeats(lw,tfeats,nidxs_lens,nidxs_bgs)

    return pfeats


def graph_conv_xyz_feats(xyz, feats, nidxs, nidxs_lens, nidxs_bgs, name, ifn, m, ofn,
                         lw=None, pmiu=None,use_bias=True,activation_fn=tf.nn.relu,
                         initializer=tf.contrib.layers.xavier_initializer(),reuse=None):
    '''

    :param xyz:
    :param feats:
    :param nidxs:
    :param nidxs_lens:
    :param nidxs_bgs:
    :param name:
    :param ifn:
    :param m:
    :param ofn:
    :param lw:
    :param pmiu:
    :param use_bias:
    :param activation_fn:
    :param initializer:
    :param reuse:
    :return:
    '''
    with tf.variable_scope(name,reuse=reuse):
        pw=_variable_on_cpu('pw', [ifn,m,ofn], initializer)
        if lw is None and pmiu is None:
                pmiu=_variable_on_cpu('pmiu', [3,m], initializer)

        if use_bias:
            bias=_variable_on_cpu('bias',[ofn],tf.zeros_initializer())

    pfeats,lw_=graph_conv_xyz_feats_impl(xyz, feats, nidxs, nidxs_lens, nidxs_bgs,pw,lw,pmiu)

    if use_bias:
        pfeats=tf.add(pfeats,tf.expand_dims(bias,axis=0))

    if activation_fn is not None:
        pfeats=activation_fn(pfeats)

    if lw is None:
        lw=lw_
        return pfeats,lw
    else:
        return pfeats


def graph_conv_xyz(xyz, nidxs, nidxs_lens, nidxs_bgs, name, ifn, m, ofn,
                 lw=None, pmiu=None,use_bias=True,activation_fn=tf.nn.relu,
                 initializer=tf.contrib.layers.xavier_initializer(),reuse=None):
    '''

    :param xyz:
    :param nidxs:
    :param nidxs_lens:
    :param nidxs_bgs:
    :param name:
    :param ifn:
    :param m:
    :param ofn:
    :param lw:
    :param pmiu:
    :param use_bias:
    :param activation_fn:
    :param initializer:
    :param reuse:
    :return:
    '''
    with tf.variable_scope(name,reuse=reuse):
        pw=_variable_on_cpu('pw', [ifn,m,ofn], initializer)
        if lw is None and pmiu is None:
                pmiu=_variable_on_cpu('pmiu', [3,m], initializer)

        if use_bias:
            bias=_variable_on_cpu('bias',[ofn],tf.zeros_initializer())

    pfeats,lw_=graph_conv_xyz_impl(xyz, nidxs, nidxs_lens, nidxs_bgs, pw, lw, pmiu)

    if use_bias:
        pfeats=tf.add(pfeats,tf.expand_dims(bias,axis=0))

    if activation_fn is not None:
        pfeats=activation_fn(pfeats)

    if lw is None:
        lw=lw_
        return pfeats,lw
    else:
        return pfeats

def graph_conv_feats(feats, nidxs, nidxs_lens, nidxs_bgs, name, ifn, m, ofn,
                     lw,use_bias=True,activation_fn=tf.nn.relu,
                     initializer=tf.contrib.layers.xavier_initializer(),
                     reuse=None):

    with tf.variable_scope(name,reuse=reuse):
        pw=_variable_on_cpu('pw', [ifn,m,ofn], initializer)
        if use_bias:
            bias=_variable_on_cpu('bias',[ofn],tf.zeros_initializer())

    pfeats=graph_conv_feats_impl(feats, nidxs, nidxs_lens, nidxs_bgs,pw, lw)

    if use_bias:
        pfeats=tf.add(pfeats,tf.expand_dims(bias,axis=0))

    if activation_fn is not None:
        pfeats=activation_fn(pfeats)

    return pfeats

def graph_conv_encoder(xyz, feats, nidxs, nidxs_lens, nidxs_bgs, m, pmiu=None, reuse=False, final_dim=512):
    '''
    :param xyz:     [pn,3] xyz
    :param feats:   [pn,f] rgb
    :param nidxs:
    :param nidxs_lens:
    :param nidxs_bgs:
    :param reuse:
    :param trainable:
    :param final_dim:
    :return:
    '''
    with tf.name_scope('encoder'):
        with framework.arg_scope([tf.contrib.layers.fully_connected],activation_fn=tf.nn.relu,reuse=reuse):

            gc1,lw=graph_conv_xyz_feats(xyz,feats,nidxs,nidxs_lens,nidxs_bgs,'gc1',6,m,32,pmiu=pmiu,reuse=reuse)
            gc1=tf.concat([gc1,feats],axis=1)

            gc2=graph_conv_feats(gc1,nidxs,nidxs_lens,nidxs_bgs,'gc2',35,m,64,lw,reuse=reuse)
            gc2=tf.concat([gc2,feats],axis=1)

            gc3=graph_conv_feats(gc2,nidxs,nidxs_lens,nidxs_bgs,'gc3',67,m,64,lw,reuse=reuse)
            gc3=tf.concat([gc3,feats],axis=1)

            gc4=graph_conv_feats(gc3,nidxs,nidxs_lens,nidxs_bgs,'gc4',67,m,64,lw,reuse=reuse)
            gc4=tf.concat([gc4,feats],axis=1)

            gc4=tf.reshape(gc4,[-1,67])
            fc1=tf.contrib.layers.fully_connected(gc4,num_outputs=128,scope='fc1')
            fc2=tf.contrib.layers.fully_connected(fc1,num_outputs=final_dim,scope='fc2',activation_fn=None)

        codewords = tf.reduce_max(fc2, axis=0)

    return codewords, fc2