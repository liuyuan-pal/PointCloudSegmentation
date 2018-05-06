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


def eval_max_pool(ifeats,vlens,vlens_bgs,dpfeats,sess):
    ifeats_pl=tf.placeholder(tf.float32,[None,None])
    vlens_pl=tf.placeholder(tf.int32,[None])
    vlens_bgs_pl=tf.placeholder(tf.int32,[None])

    pfeats,max_idxs=graph_conv_layer.graph_pool_for_test(ifeats_pl,vlens_pl,vlens_bgs_pl)
    difeats=tf.gradients(pfeats,ifeats_pl,dpfeats)[0]

    pfeats_val,difeats_val,idxs_val=sess.run(
        [pfeats,difeats,max_idxs],feed_dict={
            ifeats_pl:ifeats,
            vlens_pl:vlens,
            vlens_bgs_pl:vlens_bgs
        }
    )

    return pfeats_val,difeats_val,idxs_val

def test_max_pool_single(pn2,dim,sess):
    vlens=np.random.randint(3,5,[pn2])
    vlens_bgs=compute_nidxs_bgs(vlens)

    pn1=vlens_bgs[-1]+vlens[-1]
    ifeats=np.random.uniform(-1.0,1.0,[pn1,dim])
    pfeats,idxs=np_pool_forward(ifeats,vlens,vlens_bgs)
    dpfeats=np.random.uniform(-1.0,1.0,pfeats.shape)
    difeats=np_pool_backward(dpfeats,pfeats,ifeats,vlens,vlens_bgs)

    ifeats=np.ascontiguousarray(ifeats,np.float32)
    dpfeats=np.ascontiguousarray(dpfeats,np.float32)
    vlens=np.ascontiguousarray(vlens,np.int32)
    vlens_bgs=np.ascontiguousarray(vlens_bgs,np.int32)
    pfeats_tf,difeats_tf,idxs_tf=eval_max_pool(ifeats,vlens,vlens_bgs,dpfeats,sess)

    diff_abs=np.abs(pfeats_tf-pfeats)
    # print np.mean(diff_abs),np.max(diff_abs)
    if np.mean(diff_abs) > 1e-5 or np.max(diff_abs) > 1e-4:
        print 'forward error!'
        print pn2,dim
        exit(0)

    idxs_diff=np.sum(idxs-idxs_tf)
    if idxs_diff>0:
        xs,ys=np.nonzero(np.not_equal(idxs,idxs_tf))
        error=False
        for k in xrange(len(xs)):
            diff_val=ifeats[vlens_bgs[xs[k]]+idxs[xs[k],ys[k]],ys[k]]-ifeats[vlens_bgs[xs[k]]+idxs_tf[xs[k],ys[k]],ys[k]]
            if abs(diff_val)>1e-5:
                error=True
                break

        if error:
            print 'idxs error!'
            print pn2,dim
            exit(0)
        else:
            return

    diff_abs=np.abs(difeats_tf-difeats)
    # print np.mean(diff_abs),np.max(diff_abs)
    if np.mean(diff_abs) > 1e-5 or np.max(diff_abs) > 1e-4:
        print 'backward error!'
        print pn2,dim
        exit(0)

def test_max_pool():
    sess=tf.Session()
    for i in xrange(100):
        pn=np.random.randint(30,1030)
        fd=np.random.randint(30,1030)
        test_max_pool_single(pn,fd,sess)

    for i in xrange(100):
        pn=np.random.randint(1010,1050)
        fd=np.random.randint(300,500)
        test_max_pool_single(pn,fd,sess)

    for i in xrange(100):
        pn=np.random.randint(300,500)
        fd=np.random.randint(1010,1050)
        test_max_pool_single(pn,fd,sess)

    for i in xrange(10):
        pn=np.random.randint(2048,4096)
        fd=np.random.randint(50,100)
        test_max_pool_single(pn,fd,sess)

def eval_unpool(ifeats,vlens,vlens_bgs,cidxs,dupfeats,sess):
    ifeats_pl=tf.placeholder(tf.float32,[None,None])
    vlens_pl=tf.placeholder(tf.int32,[None])
    vlens_bgs_pl=tf.placeholder(tf.int32,[None])
    cidxs_pl=tf.placeholder(tf.int32,[None])

    upfeats=graph_conv_layer.graph_unpool(ifeats_pl,vlens_pl,vlens_bgs_pl,cidxs_pl)
    difeats=tf.gradients(upfeats,ifeats_pl,dupfeats)[0]

    upfeats_val,difeats_val=sess.run(
        [upfeats,difeats],feed_dict={
            ifeats_pl:ifeats,
            vlens_pl:vlens,
            vlens_bgs_pl:vlens_bgs,
            cidxs_pl:cidxs
        }
    )

    return upfeats_val,difeats_val

def test_unpool_single(pn2, dim, sess):
    vlens=np.random.randint(0,10,[pn2])
    vlens_bgs=compute_nidxs_bgs(vlens)
    cidxs=compute_cidxs(vlens)

    pn1=vlens_bgs[-1]+vlens[-1]
    ifeats=np.random.uniform(-1.0, 1.0, [pn2, dim])
    upfeats=np_unpool_forward(ifeats,vlens,vlens_bgs,cidxs)
    dupfeats=np.random.uniform(-1.0,1.0,upfeats.shape)
    difeats=np_unpool_backward(dupfeats,upfeats,ifeats,vlens,vlens_bgs,cidxs)

    ifeats=np.ascontiguousarray(ifeats,np.float32)
    dupfeats=np.ascontiguousarray(dupfeats,np.float32)
    vlens=np.ascontiguousarray(vlens,np.int32)
    vlens_bgs=np.ascontiguousarray(vlens_bgs,np.int32)
    upfeats_tf,difeats_tf=eval_unpool(ifeats,vlens,vlens_bgs,cidxs,dupfeats,sess)

    diff_abs=np.abs(upfeats_tf-upfeats)
    if np.mean(diff_abs) > 1e-5 or np.max(diff_abs) > 1e-4:
        print 'error!'
        print pn2,dim
        exit(0)

    diff_abs=np.abs(difeats_tf-difeats)
    if np.mean(diff_abs) > 1e-5 or np.max(diff_abs) > 1e-4:
        print 'error!'
        print pn2,dim
        exit(0)

if __name__=="__main__":
    test_max_pool()