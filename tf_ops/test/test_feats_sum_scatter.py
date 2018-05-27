import numpy as np
from data_util import compute_nidxs_bgs,compute_cidxs
from np_ops import eval_numerical_gradient_array
import sys
sys.path.append('..')
from graph_conv_layer import neighbor_ops
import tensorflow as tf


def sum_scatter_forward_np(feats,lens,begs,cens):
    pn,fd=feats.shape
    en=cens.shape[0]
    sfeats=np.empty([en,fd],np.float64)
    for i in xrange(en):
        sfeats[i,:]=feats[cens[i],:]

    return sfeats

def sum_scatter_backward_np(feats,dsfeats,lens,begs,cens):
    pn,fd=feats.shape
    en=cens.shape[0]
    dfeats=np.zeros_like(feats,np.float64)

    for i in xrange(pn):
        bg=begs[i]
        for j in xrange(lens[i]):
            dfeats[i,:]+=dsfeats[bg+j,:]

    return dfeats


def eval_val(feats,dsfeats,lens,begs,cens,sess):
    feats_pl=tf.placeholder(tf.float32,[None,None])
    cens_pl=tf.placeholder(tf.int32,[None])
    begs_pl=tf.placeholder(tf.int32,[None])
    lens_pl=tf.placeholder(tf.int32,[None])
    dsfeats_pl=tf.placeholder(tf.float32,[None,None])

    sfeats=neighbor_ops.neighbor_sum_feat_scatter(feats_pl,cens_pl,lens_pl,begs_pl)
    dfeats=tf.gradients(sfeats,feats_pl,dsfeats_pl)[0]

    sfeats_val,dfeats_val=sess.run([sfeats,dfeats],feed_dict={
        feats_pl:feats,
        cens_pl:cens,
        begs_pl:begs,
        lens_pl:lens,
        dsfeats_pl:dsfeats,
    })

    return sfeats_val,dfeats_val

def test_single(pn,fd,sess):
    lens = np.random.randint(0, 5, [pn])
    begs = compute_nidxs_bgs(lens)
    cens = compute_cidxs(lens)
    en = np.sum(lens)

    feats = np.random.uniform(-1, 1, [pn, fd])
    dsfeats = np.random.uniform(-1, 1, [en, fd])

    sfeats = sum_scatter_forward_np(feats, lens, begs, cens)
    dfeats = sum_scatter_backward_np(feats, dsfeats, lens, begs, cens)
    sfeats_val, dfeats_val = eval_val(feats, dsfeats, lens, begs, cens, sess)

    diff_abs = np.abs(dfeats_val - dfeats)
    # print 'diff {} {}'.format(np.mean(diff_abs),np.max(diff_abs))
    if np.mean(diff_abs) > 1e-5 or np.max(diff_abs) > 1e-4:
        print 'error!'
        print pn, fd
        exit(0)
    diff_abs = np.abs(sfeats_val - sfeats)
    # print 'diff {} {}'.format(np.mean(diff_abs),np.max(diff_abs))
    if np.mean(diff_abs) > 1e-5 or np.max(diff_abs) > 1e-4:
        print 'error!'
        print pn, fd
        exit(0)

def test():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess=tf.Session(config=config)

    for _ in xrange(100):
        pn=np.random.randint(30,1030)
        fd=np.random.randint(30,1030)
        test_single(pn,fd,sess)

    for _ in xrange(100):
        pn=np.random.randint(1020,1030)
        fd=np.random.randint(100,200)
        test_single(pn,fd,sess)

    for _ in xrange(100):
        pn=np.random.randint(100,200)
        fd=np.random.randint(1020,1030)
        test_single(pn,fd,sess)



if __name__=="__main__":
    test()

