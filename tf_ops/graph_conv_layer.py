import tensorflow as tf
import os
path = os.path.split(os.path.realpath(__file__))[0]
neighbor_ops=tf.load_op_library(path+'/build/libTFNeighborOps.so')
import sys
sys.path.append(path)
from generate_pmiu import generate_pmiu

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


# @ops.RegisterGradient("NeighborConcatNonCenterScatter")
# def _neighbor_concat_non_center_scatter_gradient(op,dsfeats):
#     difeats=neighbor_ops.neighbor_concat_non_center_gather(dsfeats, op.inputs[1], op.inputs[2], op.inputs[3])
#     return [difeats,None,None,None]


def _variable_on_cpu(name, shape, initializer, use_fp16=False,init_val=None):
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
        if init_val is not None:
            var_init=tf.Variable(init_val,False,name='{}_init_val'.format(name),dtype=dtype, expected_shape=init_val.shape)
            var = tf.get_variable(name, initializer=var_init.initialized_value(), dtype=dtype)
        else:
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        # tf.add_to_collection(tf.GraphKeys.TRAINABLE_VARIABLES,var)
    return var


def compute_pfeats(lw,lw_sum,tfeats,nidxs_lens,nidxs_bgs,cidxs,no_sum=False):
    '''

    :param lw:          [csum,m]
    :param lw_sum:      [pn,m]
    :param tfeats:      [csum,m,ofn]
    :param nidxs_lens:  [pn]
    :param nidxs_bgs:   [pn]
    :param cidxs:       [csum]
    :param no_sum:
    :return: pfeats [pn,ofn] or [pn,n,ofn] if no sum
    '''

    lw_exp=tf.expand_dims(lw,axis=2)                                        #[csum,m,1]
    wtfeats=lw_exp*tfeats                                                   #[csum,m,ofn]
    wtfeats=tf.reshape(wtfeats,[tf.shape(wtfeats)[0],-1])                   #[csum,m*ofn]
    tfeats_sum=neighbor_ops.neighbor_sum_feat_gather(wtfeats, cidxs, nidxs_lens, nidxs_bgs)
    tfeats_sum=tf.reshape(tfeats_sum,[-1,tf.shape(tfeats)[1],tf.shape(tfeats)[2]])
    # tfeats_sum=neighbor_ops.location_weight_feat_sum(lw,tfeats,nidxs_lens,nidxs_bgs)

    inv_lw_sum=1.0/(lw_sum+1e-6)        # [pn,m]
    inv_lw_sum_exp=tf.expand_dims(inv_lw_sum,axis=2)    # [pn,m,1]
    pfeats=tfeats_sum*inv_lw_sum_exp                    # [pn,m,ofn]
    if not no_sum:
        pfeats=tf.reduce_sum(pfeats,axis=1)
    else:
        pfeats=tf.reshape(pfeats,[tf.shape(pfeats)[0],-1])

    return pfeats


def graph_conv_xyz_feats_impl(xyz,feats,cidxs,nidxs,nidxs_lens,nidxs_bgs,pw,
                              compute_lw=False,lw=None,lw_sum=None,pmiu=None,no_sum=False):

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

    pfeats=compute_pfeats(lw,lw_sum,tfeats,nidxs_lens,nidxs_bgs,cidxs,no_sum)   # [pn,ofn]

    return pfeats,lw,lw_sum


def graph_conv_xyz_impl(xyz,cidxs,nidxs,nidxs_lens,nidxs_bgs,pw,
                        compute_lw=False,lw=None,lw_sum=None,pmiu=None,no_sum=False):

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

    pfeats=compute_pfeats(lw,lw_sum,tfeats,nidxs_lens,nidxs_bgs,cidxs,no_sum)

    return pfeats,lw,lw_sum


def graph_conv_feats_impl(feats,cidxs,nidxs,nidxs_lens,nidxs_bgs,pw,lw,lw_sum,no_sum=False):

    pw_reshape=tf.reshape(pw,[tf.shape(pw)[0],-1])  # [ifn,m*ofn]
    tfeats=tf.matmul(feats,pw_reshape)              # [pn,m*ofn]
    tfeats=neighbor_ops.neighbor_scatter(tfeats,nidxs,nidxs_lens,nidxs_bgs,use_diff=False)
    tfeats=tf.reshape(tfeats,[-1,tf.shape(pw)[1],tf.shape(pw)[2]]) # [csum,m,ofn]

    pfeats=compute_pfeats(lw,lw_sum,tfeats,nidxs_lens,nidxs_bgs,cidxs,no_sum)

    return pfeats


def graph_diff_conv_feats_impl(feats,cidxs,nidxs,nidxs_lens,nidxs_bgs,pw,lw,lw_sum,no_sum=False):
    pw_reshape=tf.reshape(pw,[tf.shape(pw)[0],-1])  # [ifn,m*ofn]
    tfeats=neighbor_ops.neighbor_scatter(feats,nidxs,nidxs_lens,nidxs_bgs,use_diff=True)
    tfeats=tf.matmul(tfeats,pw_reshape)              # [pn,m*ofn]
    tfeats=tf.reshape(tfeats,[-1,tf.shape(pw)[1],tf.shape(pw)[2]]) # [csum,m,ofn]
    pfeats=compute_pfeats(lw,lw_sum,tfeats,nidxs_lens,nidxs_bgs,cidxs,no_sum)

    return pfeats

def graph_conv_xyz_feats(xyz, feats, cidxs, nidxs, nidxs_lens, nidxs_bgs, name, ifn, m, ofn,no_sum=False,
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
        pfeats,lw_,lw_sum_=graph_conv_xyz_feats_impl(xyz,feats,cidxs,nidxs,nidxs_lens,nidxs_bgs,pw,compute_lw,lw,lw_sum,pmiu,no_sum)

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


def graph_conv_xyz(xyz, cidxs, nidxs, nidxs_lens, nidxs_bgs, name, ifn, m, ofn,no_sum=False,
                   compute_lw=False, lw=None, lw_sum=None, pmiu=None,
                   use_bias=True,activation_fn=tf.nn.relu,
                   initializer=tf.contrib.layers.xavier_initializer(),reuse=None):
    with tf.variable_scope(name,reuse=reuse):
        pw=_variable_on_cpu('pw', [ifn,m,ofn], initializer)
        if compute_lw and pmiu is None:
            pmiu_init=generate_pmiu(m)
            pmiu=_variable_on_cpu('pmiu', [3,m], initializer=None, init_val=pmiu_init)

        if use_bias:
            bdim=ofn if not no_sum else ofn*m
            bias=_variable_on_cpu('bias',[bdim],tf.zeros_initializer())

    with tf.name_scope(name):
        pfeats,lw_,lw_sum_=graph_conv_xyz_impl(xyz,cidxs,nidxs,nidxs_lens,nidxs_bgs,pw,compute_lw,lw,lw_sum,pmiu,no_sum)

        if use_bias:
            pfeats=tf.add(pfeats,tf.expand_dims(bias,axis=0))

        if activation_fn is not None:
            pfeats=activation_fn(pfeats)
            odim = ofn if not no_sum else ofn * m
            pfeats=tf.reshape(pfeats,[-1,odim])

        if compute_lw:
            lw=lw_
            lw_sum=lw_sum_
            return pfeats,lw,lw_sum
        else:
            return pfeats


def graph_conv_feats(feats, cidxs, nidxs, nidxs_lens, nidxs_bgs, name, ifn, m, ofn, lw, lw_sum,no_sum=False,
                     use_bias=True,activation_fn=tf.nn.relu,
                     initializer=tf.contrib.layers.xavier_initializer(),
                     reuse=None):

    with tf.variable_scope(name,reuse=reuse):
        pw=_variable_on_cpu('pw', [ifn,m,ofn], initializer)
        if use_bias:
            bdim=ofn if not no_sum else ofn*m
            bias=_variable_on_cpu('bias',[bdim],tf.zeros_initializer())

    with tf.name_scope(name):
        pfeats=graph_conv_feats_impl(feats,cidxs,nidxs,nidxs_lens,nidxs_bgs,pw,lw,lw_sum,no_sum)

        if use_bias:
            pfeats=tf.add(pfeats,tf.expand_dims(bias,axis=0))

        if activation_fn is not None:
            pfeats=activation_fn(pfeats)

    return pfeats


def graph_diff_conv_feats(feats, cidxs, nidxs, nidxs_lens, nidxs_bgs, name, ifn, m, ofn, lw, lw_sum,no_sum=False,
                          use_bias=True,activation_fn=tf.nn.relu,
                          initializer=tf.contrib.layers.xavier_initializer(),
                          reuse=None):

    with tf.variable_scope(name,reuse=reuse):
        pw=_variable_on_cpu('pw', [ifn,m,ofn], initializer)
        if use_bias:
            bdim=ofn if not no_sum else ofn*m
            bias=_variable_on_cpu('bias',[bdim],tf.zeros_initializer())

    with tf.name_scope(name):
        pfeats=graph_diff_conv_feats_impl(feats,cidxs,nidxs,nidxs_lens,nidxs_bgs,pw,lw,lw_sum,no_sum)

        if use_bias:
            pfeats=tf.add(pfeats,tf.expand_dims(bias,axis=0))

        if activation_fn is not None:
            pfeats=activation_fn(pfeats)

    return pfeats


def graph_pool(feats,vlens,vlens_bgs):
    pool_feats,max_idxs=neighbor_ops.neighbor_max_feat_gather(feats,vlens,vlens_bgs)
    return pool_feats


def graph_pool_for_test(feats,vlens,vlens_bgs):
    pool_feats,max_idxs=neighbor_ops.neighbor_max_feat_gather(feats,vlens,vlens_bgs)
    return pool_feats,max_idxs

def graph_avg_pool(feats,vlens,vbegs,vcens):
    feats=neighbor_ops.neighbor_sum_feat_gather(feats,vcens,vlens,vbegs)
    feats/=tf.expand_dims(tf.cast(vlens,tf.float32),axis=1)
    return feats


def graph_unpool(feats,vlens,vlens_bgs,cidxs):
    unpool_feats=neighbor_ops.neighbor_sum_feat_scatter(feats,cidxs,vlens,vlens_bgs)
    return unpool_feats


def graph_concat_non_center_scatter(ifeats,nidxs,nidxs_lens,nidxs_bgs):
    nc_nidxs,nc_nidxs_lens,nc_nidxs_bgs,nc_cidxs=neighbor_ops.eliminate_center(nidxs,nidxs_lens,nidxs_bgs)
    sfeats1=neighbor_ops.neighbor_sum_feat_scatter(ifeats,nc_cidxs,nc_nidxs_lens,nc_nidxs_bgs)
    sfeats2=neighbor_ops.neighbor_scatter(ifeats,nc_nidxs,nc_nidxs_lens,nc_nidxs_bgs,use_diff=False)
    sfeats=tf.concat([sfeats1,sfeats2],axis=1)
    return sfeats

def graph_eliminate_center(nidxs,nidxs_lens,nidxs_bgs):
    nc_nidxs,nc_nidxs_lens,nc_nidxs_bgs,nc_cidxs=neighbor_ops.eliminate_center(nidxs,nidxs_lens,nidxs_bgs)
    return nc_nidxs,nc_nidxs_lens,nc_nidxs_bgs,nc_cidxs

def graph_neighbor_scatter(ifeats,nidxs,nidxs_lens,nidxs_bgs):
    return neighbor_ops.neighbor_scatter(ifeats,nidxs,nidxs_lens,nidxs_bgs,use_diff=False)

def graph_neighbor_sum(ifeats,nidxs_lens,nidxs_bgs,cidxs):
    return neighbor_ops.neighbor_sum_feat_gather(ifeats,cidxs,nidxs_lens,nidxs_bgs)

def graph_learn_pmiu(ifeats, m, scope, nidxs, nidxs_lens, nidxs_bgs):
    with tf.variable_scope(scope):
        pmiu_init=generate_pmiu(m)
        pmiu=_variable_on_cpu('pmiu', [3,m], None, init_val=pmiu_init)

    with tf.name_scope(scope):
        sfeats = neighbor_ops.neighbor_scatter(ifeats, nidxs, nidxs_lens, nidxs_bgs, use_diff=True)
        lw = tf.exp(tf.matmul(sfeats, pmiu))                    # [csum,m]
        lw_sum = neighbor_ops.location_weight_sum(lw, nidxs_lens, nidxs_bgs)
    return lw,lw_sum



def compute_tfeats_v2(lw,lw_sum,sfeats,pw,pbias,nidxs_lens,nidxs_bgs,cidxs):
    '''
    :param lw:          [en,m]
    :param lw_sum:      [pn,m]
    :param sfeats:      [en,ifn]
    :param pw:          [m*ifn,ofn]
    :param pbias:       [ofn]
    :param nidxs_lens:  [pn]
    :param nidxs_bgs:   [pn]
    :param cidxs:       [en]
    :param no_sum:
    :return: ofeats [pn,ofn]
    '''

    lw_exp=tf.expand_dims(lw,axis=2)                                                         #[en,m,1]
    sfeats_exp=tf.expand_dims(sfeats,axis=1)                                                 #[en,1,ifn]
    wsfeats=lw_exp*sfeats_exp                                                                #[en,m,ifn]
    wsfeats_shape=tf.shape(wsfeats)
    wsfeats=tf.reshape(wsfeats,[wsfeats_shape[0],wsfeats_shape[1]*wsfeats_shape[2]])         #[en,m*ifn]
    wsfeats_sum=neighbor_ops.neighbor_sum_feat_gather(wsfeats, cidxs, nidxs_lens, nidxs_bgs) #[pn,m*ifn]
    wsfeats_sum=tf.reshape(wsfeats_sum,[-1,wsfeats_shape[1],wsfeats_shape[2]])               #[pn,m,ifn]

    inv_lw_sum=1.0/(lw_sum+1e-6)                        # [pn,m]
    inv_lw_sum_exp=tf.expand_dims(inv_lw_sum,axis=2)    # [pn,m,1]
    wfeats=wsfeats_sum*inv_lw_sum_exp                   # [pn,m,ifn]
    wfeats=tf.reshape(wfeats,[-1,wsfeats_shape[1]*wsfeats_shape[2]])
    tfeats=tf.matmul(wfeats,pw)+pbias

    return tfeats


def graph_conv_xyz_v2_impl(xyz,cidxs,nidxs,nidxs_lens,nidxs_bgs,pw,pbias,scale_val=1.0,
                           compute_lw=False,lw=None,lw_sum=None,pmiu=None):
    '''

    :param xyz:         [pn,3]
    :param cidxs:       [en]
    :param nidxs:       [en]
    :param nidxs_lens:  [pn]
    :param nidxs_bgs:   [pn]
    :param pw:          [ifn*m,ofn]
    :param pbias:       [ofn]
    :param compute_lw:
    :param lw:          [en,m]
    :param lw_sum:      [pn,m]
    :param pmiu:        [3,m]
    :return:
    '''

    sxyz=neighbor_ops.neighbor_scatter(xyz,nidxs,nidxs_lens,nidxs_bgs,use_diff=True) # [en,ifn]

    if compute_lw:
        assert lw is None and lw_sum is None and pmiu is not None
        lw=tf.exp(tf.matmul(sxyz*scale_val,pmiu))           # [csum,m]
        lw_sum=neighbor_ops.location_weight_sum(lw,nidxs_lens,nidxs_bgs)
    else:
        assert lw is not None and lw_sum is not None

    tfeats=compute_tfeats_v2(lw,lw_sum,sxyz,pw,pbias,nidxs_lens,nidxs_bgs,cidxs)

    return tfeats,lw,lw_sum


def graph_conv_xyz_v2(xyz, cidxs, nidxs, nidxs_lens, nidxs_bgs, name, ifn, m, ofn, scale_val=1.0,
                      compute_lw=False, lw=None, lw_sum=None, pmiu=None,activation_fn=tf.nn.relu,
                      initializer=tf.contrib.layers.xavier_initializer(),reuse=None):

    with tf.variable_scope(name,reuse=reuse):
        pw =_variable_on_cpu('pw', [ifn*m,ofn], initializer)
        pb =_variable_on_cpu('bias',[ofn],tf.zeros_initializer())
        if compute_lw and pmiu is None:
            pmiu_init=generate_pmiu(m)
            pmiu=_variable_on_cpu('pmiu', [3,m], initializer=None, init_val=pmiu_init)

    with tf.name_scope(name):
        pfeats,lw_,lw_sum_=graph_conv_xyz_v2_impl(xyz,cidxs,nidxs,nidxs_lens,nidxs_bgs,pw,pb,scale_val,compute_lw,lw,lw_sum,pmiu)

        if activation_fn is not None:
            pfeats=activation_fn(pfeats)
            pfeats=tf.reshape(pfeats,[-1,ofn])

        if compute_lw:
            lw=lw_
            lw_sum=lw_sum_
            return pfeats,lw,lw_sum
        else:
            return pfeats


def graph_conv_feats_v2_impl(feats,cidxs,nidxs,nidxs_lens,nidxs_bgs,pw,pbias,lw,lw_sum):
    '''

    :param xyz:         [pn,ifn]
    :param cidxs:       [en]
    :param nidxs:       [en]
    :param nidxs_lens:  [pn]
    :param nidxs_bgs:   [pn]
    :param pw:          [ifn*m,ofn]
    :param pbias:       [ofn]
    :param lw:          [en,m]
    :param lw_sum:      [pn,m]
    :return:
    '''

    sfeats=neighbor_ops.neighbor_scatter(feats,nidxs,nidxs_lens,nidxs_bgs,use_diff=False) # [en,ifn]
    tfeats=compute_tfeats_v2(lw,lw_sum,sfeats,pw,pbias,nidxs_lens,nidxs_bgs,cidxs)

    return tfeats

def graph_conv_feats_v2(feats, cidxs, nidxs, nidxs_lens, nidxs_bgs, name, ifn, m, ofn, lw, lw_sum,
                        activation_fn=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer(),reuse=None):

    with tf.variable_scope(name,reuse=reuse):
        pw =_variable_on_cpu('pw', [ifn*m,ofn], initializer)
        pb =_variable_on_cpu('bias',[ofn],tf.zeros_initializer())

    with tf.name_scope(name):
        pfeats=graph_conv_feats_v2_impl(feats,cidxs,nidxs,nidxs_lens,nidxs_bgs,pw,pb,lw,lw_sum)

        if activation_fn is not None:
            pfeats=activation_fn(pfeats)
            pfeats=tf.reshape(pfeats,[-1,ofn])

        return pfeats

##########################################################################################
def edge_weighted_trans(feats, wlw, nidxs_lens, nidxs_bgs, cidxs, model='sum'):
    '''

    :param feats:      [en,m,ifn]
    :param wlw:         [en,m,1]
    :param nidxs_lens:  [pn]
    :param nidxs_bgs:   [pn]
    :param cidxs:       [en]
    :param no_sum:
    :return: ofeats [pn,ofn]
    '''

    feats=wlw*feats                                                                     #[en,m,ifn]
    feats_shape=tf.shape(feats)
    feats=tf.reshape(feats,[feats_shape[0],feats_shape[1]*feats_shape[2]])              #[en,m*ifn]
    feats=neighbor_ops.neighbor_sum_feat_gather(feats, cidxs, nidxs_lens, nidxs_bgs)    #[pn,m*ifn]
    if model=='sum':
        feats=tf.reshape(feats,[-1,feats_shape[1],feats_shape[2]])                          #[pn,m,ifn]
        feats=tf.reduce_sum(feats,axis=1)                                                   #[pn,ifn]

    return feats


def compute_wlw(xyz,nidxs,nidxs_lens,nidxs_bgs,cidxs,pmiu,scale_val,name='weighted_lw'):
    with tf.name_scope(name):
        sxyz=neighbor_ops.neighbor_scatter(xyz,nidxs,nidxs_lens,nidxs_bgs,use_diff=True) # [en,ifn]

        lw=tf.exp(tf.matmul(sxyz*scale_val,pmiu))                                        # [en,m]
        lw_sum=neighbor_ops.location_weight_sum(lw,nidxs_lens,nidxs_bgs)
        lw_sum=1.0/(lw_sum+1e-6)
        lw_sum=neighbor_ops.neighbor_sum_feat_scatter(lw_sum,cidxs,nidxs_lens,nidxs_bgs)

    return tf.expand_dims(lw_sum*lw,axis=2)


def graph_conv_xyz_sum(xyz,wlw,m,ofn,nidxs,nidxs_lens,nidxs_bgs,cidxs,name='xyz_sum',reuse=None,activation_fn=tf.nn.relu):
    with tf.name_scope(name):

        feats=neighbor_ops.neighbor_scatter(xyz,nidxs,nidxs_lens,nidxs_bgs,use_diff=True) # [en,ifn]
        feats=tf.contrib.layers.fully_connected(feats, num_outputs=m*ofn, scope='{}_embed'.format(name),
                                                activation_fn=None, reuse=reuse) # [en,m*ofn]

        feats=tf.reshape(feats,[-1,m,ofn])# [en,m,ofn]
        feats=edge_weighted_trans(feats, wlw, nidxs_lens, nidxs_bgs, cidxs)   # [pn,ofn]

        if activation_fn is not None:
            feats=activation_fn(feats)

    return feats


def graph_conv_feats_sum(feats,wlw,m,ofn,nidxs,nidxs_lens,nidxs_bgs,cidxs,name='feats_sum',reuse=None,activation_fn=tf.nn.relu):
    with tf.name_scope(name):

        feats=tf.contrib.layers.fully_connected(feats, num_outputs=m*ofn, scope='{}_embed'.format(name),
                                                activation_fn=None, reuse=reuse)
        feats=neighbor_ops.neighbor_scatter(feats,nidxs,nidxs_lens,nidxs_bgs,use_diff=False) # [en,ifn]

        feats = tf.reshape(feats, [-1, m, ofn])
        feats=edge_weighted_trans(feats, wlw, nidxs_lens, nidxs_bgs, cidxs)

        if activation_fn is not None:
            feats=activation_fn(feats)

    return feats


def graph_conv_xyz_concat(xyz,wlw,m,ofn,nidxs,nidxs_lens,nidxs_bgs,cidxs,name='xyz_sum',reuse=None,activation_fn=tf.nn.relu):
    with tf.name_scope(name):
        feats=neighbor_ops.neighbor_scatter(xyz,nidxs,nidxs_lens,nidxs_bgs,use_diff=True) # [en,ifn]
        feats=tf.expand_dims(feats,axis=1)                                                # [en,1,ifn]
        feats=edge_weighted_trans(feats, wlw, nidxs_lens, nidxs_bgs, cidxs, 'concat')     # [pn,m*ifn]
        feats=tf.reshape(feats,[-1,m*3])
        feats=tf.contrib.layers.fully_connected(feats, num_outputs=ofn, scope='{}_embed'.format(name),
                                                activation_fn=activation_fn, reuse=reuse) # [pn,ofn]

    return feats


def graph_conv_feats_concat(feats,wlw,ifn,m,ofn,nidxs,nidxs_lens,nidxs_bgs,cidxs,name='feats_sum',reuse=None,activation_fn=tf.nn.relu):
    with tf.name_scope(name):
        feats=neighbor_ops.neighbor_scatter(feats,nidxs,nidxs_lens,nidxs_bgs,use_diff=False) # [en,ifn]
        feats=tf.expand_dims(feats,axis=1)
        feats=edge_weighted_trans(feats, wlw, nidxs_lens, nidxs_bgs, cidxs, 'concat')        # [pn,m*ifn]
        feats=tf.reshape(feats,[-1,m*ifn])
        feats=tf.contrib.layers.fully_connected(feats, num_outputs=ofn, scope='{}_embed'.format(name),
                                                activation_fn=activation_fn, reuse=reuse)

    return feats

def trainable_pmiu(m,scope,is_training=True):
    with tf.variable_scope(scope):
        if is_training:
            pmiu_init=generate_pmiu(m)
            pmiu=_variable_on_cpu('pmiu', [3,m], None, init_val=pmiu_init)
        else:
            pmiu=_variable_on_cpu('pmiu', [3,m], tf.contrib.layers.xavier_initializer())

    return pmiu


def compute_diff_feats_wlw(feats, m, fc_dims, nidxs, nidxs_lens, nidxs_bgs, cidxs, name='weighted_lw',reuse=None):
    with tf.name_scope(name):
        feats=neighbor_ops.neighbor_scatter(feats, nidxs, nidxs_lens, nidxs_bgs, use_diff=True) # [en,ifn]

        for idx,fd in enumerate(fc_dims):
            feats=tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='{}_fc_{}'.format(name,idx),
                                                    activation_fn=tf.nn.relu, reuse=reuse)
        lw = tf.contrib.layers.fully_connected(feats, num_outputs=m, scope='{}_fc_weights'.format(name,),
                                               activation_fn=None, reuse=reuse)
        lw=tf.clip_by_value(lw,-10,10)
        lw=tf.exp(lw)

        lw_sum=neighbor_ops.location_weight_sum(lw,nidxs_lens,nidxs_bgs)
        lw_sum=1.0/(lw_sum+1e-6)
        lw_sum=neighbor_ops.neighbor_sum_feat_scatter(lw_sum,cidxs,nidxs_lens,nidxs_bgs)

    return tf.expand_dims(lw_sum*lw,axis=2)


def graph_conv_edge(sxyzs,feats,ifn,fc_dims,ofn,nidxs,nidxs_lens,nidxs_bgs,cidxs,name='feats_sum',reuse=None,activation_fn=tf.nn.relu):
    with tf.name_scope(name):
        sfeats = neighbor_ops.neighbor_scatter(feats, nidxs, nidxs_lens, nidxs_bgs, use_diff=True)  # [en,ifn]
        sfeats = tf.concat([sfeats,sxyzs],axis=1)

        for idx,fd in enumerate(fc_dims):
            cfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=fd, scope='{}_fc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            sfeats=tf.concat([cfeats,sfeats],axis=1)

        sfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=ifn*ofn, scope='{}_fc_ew'.format(name),
                                                 activation_fn=None, reuse=reuse)

        ew=tf.reshape(sfeats,[-1,ifn,ofn])
        feats=neighbor_ops.neighbor_scatter(feats, nidxs, nidxs_lens, nidxs_bgs, use_diff=False)      # [en,ifn]
        feats=tf.expand_dims(feats,axis=1)
        feats=tf.squeeze(tf.matmul(feats,ew),axis=1)                                                  # [en,ofn]
        weights_inv=tf.expand_dims(1.0/tf.cast(nidxs_lens,tf.float32),axis=1)                         # [pn]
        feats=weights_inv*neighbor_ops.neighbor_sum_feat_gather(feats, cidxs, nidxs_lens, nidxs_bgs)  # [pn,ofn]

        with tf.variable_scope(name,reuse=reuse):
            bias=_variable_on_cpu('{}_bias'.format(name),[ofn],tf.zeros_initializer())

        feats=tf.nn.bias_add(feats,bias)

        if activation_fn is not None:
            feats=activation_fn(feats)

        return feats

def graph_conv_edge_xyz(sxyzs,ifn,fc_dims,ofn,nidxs,nidxs_lens,nidxs_bgs,cidxs,name='feats_sum',reuse=None,activation_fn=tf.nn.relu):
    with tf.name_scope(name):
        sfeats=sxyzs
        for idx,fd in enumerate(fc_dims):
            cfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=fd, scope='{}_fc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            sfeats=tf.concat([cfeats,sfeats],axis=1)

        sfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=ifn*ofn, scope='{}_fc_ew'.format(name),
                                                 activation_fn=None, reuse=reuse)

        ew=tf.reshape(sfeats,[-1,ifn,ofn])
        sxyzs=tf.expand_dims(sxyzs,axis=1)
        feats=tf.squeeze(tf.matmul(sxyzs,ew),axis=1)                                                  # [en,ofn]
        weights_inv=tf.expand_dims(1.0/tf.cast(nidxs_lens,tf.float32),axis=1)                         # [pn]
        feats=weights_inv*neighbor_ops.neighbor_sum_feat_gather(feats, cidxs, nidxs_lens, nidxs_bgs)  # [pn,ofn]

        with tf.variable_scope(name,reuse=reuse):
            bias=_variable_on_cpu('{}_bias'.format(name),[ofn],tf.zeros_initializer())

        feats=tf.nn.bias_add(feats,bias)

        if activation_fn is not None:
            feats=activation_fn(feats)

        return feats

def graph_conv_edge_xyz_v2(sxyzs,ifn,fc_dims,ofn,nidxs,nidxs_lens,nidxs_bgs,cidxs,name='feats_sum',reuse=None,activation_fn=tf.nn.relu):
    with tf.name_scope(name):
        sfeats=sxyzs
        dim_sum=3
        for idx,fd in enumerate(fc_dims):
            cfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=fd, scope='{}_fc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            sfeats=tf.concat([cfeats,sfeats],axis=1)
            dim_sum+=fd

        ew=tf.contrib.layers.fully_connected(sfeats, num_outputs=dim_sum*ofn, scope='{}_fc_ew'.format(name),
                                             activation_fn=None, reuse=reuse)

        ew=tf.reshape(ew,[-1,dim_sum,ofn])
        sfeats=tf.expand_dims(sfeats,axis=1)
        feats=tf.squeeze(tf.matmul(sfeats,ew),axis=1)                                                 # [en,ofn]

        eps=1e-3
        weights_inv=tf.expand_dims((1.0+eps)/(tf.cast(nidxs_lens,tf.float32)+eps),axis=1)                         # [pn]
        feats=weights_inv*neighbor_ops.neighbor_sum_feat_gather(feats, cidxs, nidxs_lens, nidxs_bgs)  # [pn,ofn]

        with tf.variable_scope(name,reuse=reuse):
            bias=_variable_on_cpu('{}_bias'.format(name),[ofn],tf.zeros_initializer())

        feats=tf.nn.bias_add(feats,bias)

        if activation_fn is not None:
            feats=activation_fn(feats)

        return feats


def graph_conv_edge_simp(sxyzs, feats, ifn, ifc_dims, ofc_dims, ofn, nidxs, nidxs_lens, nidxs_bgs, cidxs, name, reuse=None):
    with tf.name_scope(name):
        # sfeats = graph_concat_scatter(feats, nidxs, nidxs_lens, nidxs_bgs, cidxs)  # [en,2*ifn]
        sfeats = neighbor_ops.neighbor_scatter(feats, nidxs, nidxs_lens, nidxs_bgs, use_diff=True)  # [en,ifn]
        sfeats = tf.concat([sfeats,sxyzs],axis=1)

        for idx,fd in enumerate(ifc_dims):
            cfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=fd, scope='{}_ifc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            sfeats=tf.concat([cfeats,sfeats],axis=1)

        ew=tf.contrib.layers.fully_connected(sfeats, num_outputs=ifn, scope='{}_fc_ew'.format(name),
                                             activation_fn=None, reuse=reuse)

        feats=neighbor_ops.neighbor_scatter(feats, nidxs, nidxs_lens, nidxs_bgs, use_diff=False)      # [en,ifn]
        feats=ew*feats
        for idx,fd in enumerate(ofc_dims):
            cfeats=tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='{}_ofc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            feats=tf.concat([cfeats,feats],axis=1)

        eps=1e-3
        weights_inv=tf.expand_dims((1.0+eps)/(tf.cast(nidxs_lens,tf.float32)+eps),axis=1)                         # [pn]
        feats=weights_inv*neighbor_ops.neighbor_sum_feat_gather(feats, cidxs, nidxs_lens, nidxs_bgs)  # [pn,ofn]
        feats=tf.contrib.layers.fully_connected(feats, num_outputs=ofn, scope='{}_fc_out'.format(name),
                                                activation_fn=tf.nn.relu, reuse=reuse)


        return feats


def graph_conv_edge_xyz_simp(sxyzs, ifn, ifc_dims, ofc_dims, ofn, nidxs, nidxs_lens, nidxs_bgs, cidxs, name, reuse=None):
    with tf.name_scope(name):
        sfeats=sxyzs
        dim_sum=3
        for idx,fd in enumerate(ifc_dims):
            cfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=fd, scope='{}_ifc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            sfeats=tf.concat([cfeats,sfeats],axis=1)
            dim_sum+=fd

        ew=tf.contrib.layers.fully_connected(sfeats, num_outputs=dim_sum, scope='{}_fc_ew'.format(name),
                                             activation_fn=None, reuse=reuse)

        feats=ew*sfeats                                                                               # [en,ifn]
        # we need to embed the edge-conditioned feature to avoid signal mixed up
        for idx,fd in enumerate(ofc_dims):
            cfeats=tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='{}_ofc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            feats=tf.concat([cfeats,feats],axis=1)

        eps=1e-3
        weights_inv=tf.expand_dims((1.0+eps)/(tf.cast(nidxs_lens,tf.float32)+eps),axis=1)                         # [pn]
        feats=weights_inv*neighbor_ops.neighbor_sum_feat_gather(feats, cidxs, nidxs_lens, nidxs_bgs)  # [pn,ofn]

        feats=tf.contrib.layers.fully_connected(feats, num_outputs=ofn, scope='{}_fc_out'.format(name),
                                                activation_fn=tf.nn.relu, reuse=reuse)

        return feats


def graph_conv_edge_simp_v2(sxyzs, feats, ifn, ifc_dims, ofc_dims, ofn, nidxs, nidxs_lens, nidxs_bgs, cidxs, name, reuse=None):
    with tf.name_scope(name):
        # sfeats = graph_concat_scatter(feats, nidxs, nidxs_lens, nidxs_bgs, cidxs)  # [en,2*ifn]
        sfeats = neighbor_ops.neighbor_scatter(feats, nidxs, nidxs_lens, nidxs_bgs, use_diff=True)  # [en,ifn]
        sfeats = tf.concat([sfeats,sxyzs],axis=1)

        for idx,fd in enumerate(ifc_dims):
            cfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=fd, scope='{}_ifc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            sfeats=tf.concat([cfeats,sfeats],axis=1)

        ew=tf.contrib.layers.fully_connected(sfeats, num_outputs=ifn, scope='{}_fc_ew'.format(name),
                                             activation_fn=None, reuse=reuse)

        feats=neighbor_ops.neighbor_scatter(feats, nidxs, nidxs_lens, nidxs_bgs, use_diff=False)      # [en,ifn]
        feats=ew*feats
        for idx,fd in enumerate(ofc_dims):
            cfeats=tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='{}_ofc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            feats=tf.concat([cfeats,feats],axis=1)

        eps=1e-3
        weights_inv=tf.expand_dims((1.0+eps)/(tf.cast(nidxs_lens,tf.float32)+eps),axis=1)                         # [pn]
        feats=weights_inv*neighbor_ops.neighbor_sum_feat_gather(feats, cidxs, nidxs_lens, nidxs_bgs)  # [pn,ofn]
        feats=tf.contrib.layers.fully_connected(feats, num_outputs=ofn, scope='{}_fc_out'.format(name),
                                                activation_fn=tf.nn.relu, reuse=reuse)


        return feats


def graph_conv_edge_xyz_simp_v2(sxyzs, ifn, ifc_dims, ofc_dims, ofn, nidxs, nidxs_lens, nidxs_bgs, cidxs, name, reuse=None):
    with tf.name_scope(name):
        sfeats=sxyzs
        dim_sum=3
        for idx,fd in enumerate(ifc_dims):
            cfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=fd, scope='{}_ifc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            sfeats=tf.concat([cfeats,sfeats],axis=1)
            dim_sum+=fd

        ew=tf.contrib.layers.fully_connected(sfeats, num_outputs=dim_sum, scope='{}_fc_ew'.format(name),
                                             activation_fn=None, reuse=reuse)

        feats=ew*sfeats                                                                               # [en,ifn]
        # we need to embed the edge-conditioned feature to avoid signal mixed up
        for idx,fd in enumerate(ofc_dims):
            cfeats=tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='{}_ofc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            feats=tf.concat([cfeats,feats],axis=1)

        weights_inv=tf.expand_dims(1.0/tf.cast(nidxs_lens,tf.float32),axis=1)                         # [pn]
        feats=weights_inv*neighbor_ops.neighbor_sum_feat_gather(feats, cidxs, nidxs_lens, nidxs_bgs)  # [pn,ofn]

        feats=tf.contrib.layers.fully_connected(feats, num_outputs=ofn, scope='{}_fc_out'.format(name),
                                                activation_fn=tf.nn.relu, reuse=reuse)

        # with tf.variable_scope(name,reuse=reuse):
        #     bias=_variable_on_cpu('{}_bias'.format(name),[ofn],tf.zeros_initializer())
        # feats=tf.nn.bias_add(feats,bias)
        # if activation_fn is not None:
        #     feats=activation_fn(feats)

        return feats

def graph_concat_scatter(feats, nidxs, nlens, nbegs, ncens):
    scatter_feats1 = graph_unpool(feats, nlens, nbegs, ncens)
    scatter_feats2 = graph_neighbor_scatter(feats, nidxs, nlens, nbegs)
    scatter_feats = tf.concat([scatter_feats1, scatter_feats2], axis=1)
    return scatter_feats




def graph_conv_edge_simp_test(sxyzs, feats, ifn, ifc_dims, ofc_dims, ofn, nidxs, nidxs_lens, nidxs_bgs, cidxs, name, reuse=None):
    with tf.name_scope(name):
        # sfeats = neighbor_ops.neighbor_scatter(feats, nidxs, nidxs_lens, nidxs_bgs, use_diff=True)  # [en,ifn]
        # sfeats = tf.concat([sfeats,sxyzs],axis=1)
        #
        # for idx,fd in enumerate(ifc_dims):
        #     cfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=fd, scope='{}_ifc_{}'.format(name,idx),
        #                                              activation_fn=tf.nn.relu, reuse=reuse)
        #     sfeats=tf.concat([cfeats,sfeats],axis=1)
        #
        # ew=tf.contrib.layers.fully_connected(sfeats, num_outputs=ifn, scope='{}_fc_ew'.format(name),
        #                                      activation_fn=None, reuse=reuse)

        feats=neighbor_ops.neighbor_scatter(feats, nidxs, nidxs_lens, nidxs_bgs, use_diff=False)      # [en,ifn]
        feats=tf.concat([feats,sxyzs],axis=1)

        # feats=ew*feats
        for idx,fd in enumerate(ofc_dims):
            cfeats=tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='{}_ofc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            feats=tf.concat([cfeats,feats],axis=1)

        eps=1e-3
        weights_inv=tf.expand_dims((1.0+eps)/(tf.cast(nidxs_lens,tf.float32)+eps),axis=1)                         # [pn]
        feats=weights_inv*neighbor_ops.neighbor_sum_feat_gather(feats, cidxs, nidxs_lens, nidxs_bgs)  # [pn,ofn]
        feats=tf.contrib.layers.fully_connected(feats, num_outputs=ofn, scope='{}_fc_out'.format(name),
                                                activation_fn=tf.nn.relu, reuse=reuse)


        return feats


def graph_conv_edge_xyz_simp_test(sxyzs, ifn, ifc_dims, ofc_dims, ofn, nidxs, nidxs_lens, nidxs_bgs, cidxs, name, reuse=None):
    with tf.name_scope(name):
        # sfeats=sxyzs
        # dim_sum=3
        # for idx,fd in enumerate(ifc_dims):
        #     cfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=fd, scope='{}_ifc_{}'.format(name,idx),
        #                                              activation_fn=tf.nn.relu, reuse=reuse)
        #     sfeats=tf.concat([cfeats,sfeats],axis=1)
        #     dim_sum+=fd
        #
        # ew=tf.contrib.layers.fully_connected(sfeats, num_outputs=dim_sum, scope='{}_fc_ew'.format(name),
        #                                      activation_fn=None, reuse=reuse)

        # feats=ew*sfeats                                                                               # [en,ifn]
        feats=sxyzs
        # we need to embed the edge-conditioned feature to avoid signal mixed up
        for idx,fd in enumerate(ofc_dims):
            cfeats=tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='{}_ofc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            feats=tf.concat([cfeats,feats],axis=1)

        eps=1e-3
        weights_inv=tf.expand_dims((1.0+eps)/(tf.cast(nidxs_lens,tf.float32)+eps),axis=1)                         # [pn]
        feats=weights_inv*neighbor_ops.neighbor_sum_feat_gather(feats, cidxs, nidxs_lens, nidxs_bgs)  # [pn,ofn]

        feats=tf.contrib.layers.fully_connected(feats, num_outputs=ofn, scope='{}_fc_out'.format(name),
                                                activation_fn=tf.nn.relu, reuse=reuse)

        return feats