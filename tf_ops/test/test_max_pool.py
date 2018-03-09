import sys
sys.path.append('..')
import libPointUtil
import numpy as np
from draw_util import *
from data_util import *
import graph_conv_layer
import tensorflow as tf
from np_ops import *

def permutation(feats_list,idxs):
    for i in xrange(len(feats_list)):
        feats_list[i]=feats_list[i][idxs]
    return feats_list


def build_hierarchy(xyzs,feats_list,vs1,vs2):

    ###########################

    cxyz1=np.ascontiguousarray(xyzs)
    sidxs1,vlens1=libPointUtil.sortVoxelGPU(cxyz1,vs1)

    cxyz1=cxyz1[sidxs1,:]
    cxyz1=np.ascontiguousarray(cxyz1)
    dxyz1,cxyz2=libPointUtil.computeCenterDiffCPU(cxyz1,vlens1)

    feats_list=permutation(feats_list,sidxs1)

    ############################

    cxyz2=np.ascontiguousarray(cxyz2)
    sidxs2,vlens2=libPointUtil.sortVoxelGPU(cxyz2,vs2)

    cxyz2=cxyz2[sidxs2,:]
    cxyz2=np.ascontiguousarray(cxyz2)
    dxyz2,cxyz3=libPointUtil.computeCenterDiffCPU(cxyz2,vlens2)

    sidxs1,vlens1=libPointUtil.adjustPointsMemoryCPU(vlens1,sidxs2,cxyz1.shape[0])
    dxyz1,cxyz1=permutation([dxyz1,cxyz1],sidxs1)
    feats_list=permutation(feats_list,sidxs1)

    return cxyz1,dxyz1,vlens1,cxyz2,dxyz2,vlens2,cxyz3,feats_list


def eval_max_pool(ifeats,vlens,vlens_bgs,dpfeats):
    ifeats_pl=tf.placeholder(tf.float32,[None,None])
    vlens_pl=tf.placeholder(tf.int32,[None])
    vlens_bgs_pl=tf.placeholder(tf.int32,[None])

    pfeats=graph_conv_layer.graph_pool(ifeats_pl,vlens_pl,vlens_bgs_pl)
    difeats=tf.gradients(pfeats,ifeats_pl,dpfeats)[0]

    with tf.Session() as sess:
        pfeats_val,difeats_val=sess.run(
            [pfeats,difeats],feed_dict={
                ifeats_pl:ifeats,
                vlens_pl:vlens,
                vlens_bgs_pl:vlens_bgs
            }
        )

    return pfeats_val,difeats_val


def eval_unpool(ifeats,vlens,vlens_bgs,cidxs,dupfeats):
    ifeats_pl=tf.placeholder(tf.float32,[None,None])
    vlens_pl=tf.placeholder(tf.int32,[None])
    vlens_bgs_pl=tf.placeholder(tf.int32,[None])
    cidxs_pl=tf.placeholder(tf.int32,[None])

    upfeats=graph_conv_layer.graph_unpool(ifeats_pl,vlens_pl,vlens_bgs_pl,cidxs_pl)
    difeats=tf.gradients(upfeats,ifeats_pl,dupfeats)[0]

    with tf.Session() as sess:
        upfeats_val,difeats_val=sess.run(
            [upfeats,difeats],feed_dict={
                ifeats_pl:ifeats,
                vlens_pl:vlens,
                vlens_bgs_pl:vlens_bgs,
                cidxs_pl:cidxs
            }
        )

    return upfeats_val,difeats_val

def test_max_pool():
    pn2=30
    dim=4
    vlens=np.random.randint(3,5,[pn2])
    vlens_bgs=compute_nidxs_bgs(vlens)

    pn1=vlens_bgs[-1]+vlens[-1]
    ifeats=np.random.uniform(-1.0,1.0,[pn1,dim])
    pfeats,max_idxs=np_pool_forward(ifeats,vlens,vlens_bgs)
    dpfeats=np.random.uniform(-1.0,1.0,pfeats.shape)
    difeats=np_pool_backward(dpfeats,pfeats,ifeats,vlens,vlens_bgs)

    ifeats=np.ascontiguousarray(ifeats,np.float32)
    dpfeats=np.ascontiguousarray(dpfeats,np.float32)
    vlens=np.ascontiguousarray(vlens,np.int32)
    vlens_bgs=np.ascontiguousarray(vlens_bgs,np.int32)
    pfeats_tf,difeats_tf=eval_max_pool(ifeats,vlens,vlens_bgs,dpfeats)

    print 'forward diff {}'.format(np.max(pfeats_tf-pfeats))
    print 'backward diff {}'.format(np.max(np.abs(difeats_tf-difeats)))
    print np.mean(difeats_tf)


def test_unpool():
    vlens=np.random.randint(5,10,[50])
    vlens_bgs=compute_nidxs_bgs(vlens)
    cidxs=compute_cidxs(vlens)

    pn1=vlens_bgs[-1]+vlens[-1]
    pn2=50
    ifeats=np.random.uniform(-1.0,1.0,[pn2,16])
    upfeats=np_unpool_forward(ifeats,vlens,vlens_bgs,cidxs)
    dupfeats=np.random.uniform(-1.0,1.0,upfeats.shape)
    difeats=np_unpool_backward(dupfeats,upfeats,ifeats,vlens,vlens_bgs,cidxs)

    ifeats=np.ascontiguousarray(ifeats,np.float32)
    dupfeats=np.ascontiguousarray(dupfeats,np.float32)
    vlens=np.ascontiguousarray(vlens,np.int32)
    vlens_bgs=np.ascontiguousarray(vlens_bgs,np.int32)
    upfeats_tf,difeats_tf=eval_unpool(ifeats,vlens,vlens_bgs,cidxs,dupfeats)

    print 'forward diff {}'.format(np.max(upfeats_tf-upfeats))
    print 'backward diff {}'.format(np.max(difeats_tf-difeats))
    print np.mean(upfeats_tf)
    print np.mean(difeats_tf)

if __name__=="__main__":
    test_max_pool()