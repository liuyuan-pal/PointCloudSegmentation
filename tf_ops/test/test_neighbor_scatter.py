import numpy as np
from np_ops import compute_nidxs_bgs,compute_cidxs,eval_numerical_gradient_array
import sys
sys.path.append('..')
from graph_conv_layer import neighbor_ops
import tensorflow as tf

def copy_scatter_np(feats,nidxs,nlens,nbegs,ncens):
    pn,fd=feats.shape
    en=nidxs.shape[0]
    sfeats=np.empty([en,fd],np.float64)

    for i in xrange(pn):
        bg=nbegs[i]
        ln=nlens[i]

        for j in xrange(ln):
            sfeats[bg+j,:]=feats[nidxs[bg+j],:]

    return sfeats

def copy_gather_np(feats,dsfeats,nidxs,nlens,nbegs,ncens):
    pn,fd=feats.shape
    en=nidxs.shape[0]
    dfeats=np.zeros_like(feats)

    for i in xrange(pn):
        bg=nbegs[i]
        ln=nlens[i]

        for j in xrange(ln):
            dfeats[nidxs[bg + j], :]+=dsfeats[bg+j,:]

    return dfeats

def diff_scatter_np(feats,nidxs,nlens,nbegs,ncens):
    pn,fd=feats.shape
    en=nidxs.shape[0]
    sfeats=np.empty([en,fd],np.float64)

    for i in xrange(pn):
        bg=nbegs[i]
        ln=nlens[i]

        for j in xrange(ln):
            sfeats[bg+j,:]=feats[nidxs[bg+j],:]-feats[i,:]

    return sfeats

def diff_gather_np(feats,dsfeats,nidxs,nlens,nbegs,ncens):
    pn,fd=feats.shape
    en=nidxs.shape[0]
    dfeats=np.zeros_like(feats)

    for i in xrange(pn):
        bg=nbegs[i]
        ln=nlens[i]

        for j in xrange(ln):
            dfeats[nidxs[bg + j], :]+=dsfeats[bg+j,:]
            dfeats[i,:]-=dsfeats[bg+j,:]

    return dfeats

def test_copy_np(pn,fd):
    feats=np.random.uniform(-1,1,[pn,fd])
    nidxs=[]
    for i in xrange(pn):
        idxs=list(np.random.choice(pn,5,False))
        if i not in idxs: idxs.append(i)
        nidxs.append(idxs)

    nlens=np.asarray([len(idxs) for idxs in nidxs])
    nbegs=compute_nidxs_bgs(nlens)
    ncens=compute_cidxs(nlens)
    nidxs=np.concatenate(nidxs,axis=0)

    sfeats=copy_scatter_np(feats,nidxs,nlens,nbegs,ncens)
    dsfeats=np.random.uniform(-1,1,sfeats.shape)
    dfeats=copy_gather_np(feats,dsfeats,nidxs,nlens,nbegs,ncens)

    fn=lambda feats: copy_scatter_np(feats,nidxs,nlens,nbegs,ncens)
    dfeats_num=eval_numerical_gradient_array(fn,feats,dsfeats)

    abs_val=np.abs(dfeats-dfeats_num)
    print np.mean(abs_val),np.max(abs_val)

def test_diff_np(pn,fd):
    feats=np.random.uniform(-1,1,[pn,fd])
    nidxs=[]
    for i in xrange(pn):
        idxs=list(np.random.choice(pn,5,False))
        if i not in idxs: idxs.append(i)
        nidxs.append(idxs)

    nlens=np.asarray([len(idxs) for idxs in nidxs])
    nbegs=compute_nidxs_bgs(nlens)
    ncens=compute_cidxs(nlens)
    nidxs=np.concatenate(nidxs,axis=0)

    sfeats=diff_scatter_np(feats,nidxs,nlens,nbegs,ncens)
    dsfeats=np.random.uniform(-1,1,sfeats.shape)
    dfeats=diff_gather_np(feats,dsfeats,nidxs,nlens,nbegs,ncens)

    fn=lambda feats: diff_scatter_np(feats,nidxs,nlens,nbegs,ncens)
    dfeats_num=eval_numerical_gradient_array(fn,feats,dsfeats)

    abs_val=np.abs(dfeats-dfeats_num)
    print np.mean(abs_val),np.max(abs_val)



def eval_tf_val(feats,dsfeats,nidxs,nlens,nbegs,sess):
    nidxs_pl=tf.placeholder(tf.int32,[None])
    nlens_pl=tf.placeholder(tf.int32,[None])
    nbegs_pl=tf.placeholder(tf.int32,[None])
    feats_pl=tf.placeholder(tf.float32,[None,None])
    dsfeats_pl=tf.placeholder(tf.float32,[None,None])

    sfeats_op=neighbor_ops.neighbor_scatter(feats_pl, nidxs_pl, nlens_pl, nbegs_pl, use_diff=False)
    dfeats_op=tf.gradients(sfeats_op,feats_pl,dsfeats_pl)[0]

    sfeats,dfeats=sess.run([sfeats_op,dfeats_op],feed_dict={
        nidxs_pl:nidxs,
        nlens_pl:nlens,
        nbegs_pl:nbegs,
        feats_pl:feats,
        dsfeats_pl:dsfeats,
    })
    return sfeats,dfeats

def eval_tf_concat_val(feats,dsfeats,nidxs,nlens,nbegs,sess):
    nidxs_pl=tf.placeholder(tf.int32,[None])
    nlens_pl=tf.placeholder(tf.int32,[None])
    nbegs_pl=tf.placeholder(tf.int32,[None])
    feats_pl=tf.placeholder(tf.float32,[None,None])
    dsfeats_pl=tf.placeholder(tf.float32,[None,None])

    sfeats_op=neighbor_ops.neighbor_scatter(feats_pl, nidxs_pl, nlens_pl, nbegs_pl, use_diff=False)

    dfeats_op=tf.gradients(sfeats_op,feats_pl,dsfeats_pl)[0]

    sfeats,dfeats=sess.run([sfeats_op,dfeats_op],feed_dict={
        nidxs_pl:nidxs,
        nlens_pl:nlens,
        nbegs_pl:nbegs,
        feats_pl:feats,
        dsfeats_pl:dsfeats,
    })
    return sfeats,dfeats

def eval_tf_diff_val(feats,dsfeats,nidxs,nlens,nbegs,sess):
    nidxs_pl=tf.placeholder(tf.int32,[None])
    nlens_pl=tf.placeholder(tf.int32,[None])
    nbegs_pl=tf.placeholder(tf.int32,[None])
    feats_pl=tf.placeholder(tf.float32,[None,None])
    dsfeats_pl=tf.placeholder(tf.float32,[None,None])

    sfeats_op=neighbor_ops.neighbor_scatter(feats_pl, nidxs_pl, nlens_pl, nbegs_pl, use_diff=True)
    dfeats_op=tf.gradients(sfeats_op,feats_pl,dsfeats_pl)[0]

    sfeats,dfeats=sess.run([sfeats_op,dfeats_op],feed_dict={
        nidxs_pl:nidxs,
        nlens_pl:nlens,
        nbegs_pl:nbegs,
        feats_pl:feats,
        dsfeats_pl:dsfeats,
    })
    return sfeats,dfeats

def test_copy_single(pn,fd,sess):
    feats=np.random.uniform(-1,1,[pn,fd])
    nidxs=[]
    nlens=[]
    for i in xrange(pn):
        num=np.random.randint(0,10)
        nlens.append(num)
        if num>0:
            idxs=np.random.choice(pn,num,False)
            nidxs.append(idxs)

    nlens=np.asarray(nlens)
    nbegs=compute_nidxs_bgs(nlens)
    ncens=compute_cidxs(nlens)
    nidxs=np.concatenate(nidxs,axis=0)

    sfeats=copy_scatter_np(feats,nidxs,nlens,nbegs,ncens)
    dsfeats=np.random.uniform(-1,1,sfeats.shape)
    dfeats=copy_gather_np(feats,dsfeats,nidxs,nlens,nbegs,ncens)

    sfeats_tf,dfeats_tf=eval_tf_val(feats,dsfeats,nidxs,nlens,nbegs,sess)

    abs_val=np.abs(dfeats_tf-dfeats)
    if np.mean(abs_val) > 1e-5 or np.max(abs_val) > 1e-4:
        print 'error!'
        print pn, fd
        exit(0)
    abs_val=np.abs(sfeats_tf-sfeats)
    if np.mean(abs_val) > 1e-5 or np.max(abs_val) > 1e-4:
        print 'error!'
        print pn, fd
        exit(0)

def test_diff_single(pn,fd,sess):
    feats=np.random.uniform(-1,1,[pn,fd])
    ifeats=np.copy(feats)
    nidxs=[]
    nlens=[]
    for i in xrange(pn):
        num=np.random.randint(0,10)
        nlens.append(num)
        if num>0:
            idxs=np.random.choice(pn,num,False)
            nidxs.append(idxs)

    nlens=np.asarray(nlens)
    nbegs=compute_nidxs_bgs(nlens)
    ncens=compute_cidxs(nlens)
    nidxs=np.concatenate(nidxs,axis=0)

    sfeats=diff_scatter_np(feats,nidxs,nlens,nbegs,ncens)
    dsfeats=np.random.uniform(-1,1,sfeats.shape)
    idsfeats=np.copy(dsfeats)
    dfeats=diff_gather_np(feats,dsfeats,nidxs,nlens,nbegs,ncens)
    sfeats_tf,dfeats_tf=eval_tf_diff_val(ifeats,idsfeats,nidxs,nlens,nbegs,sess)

    abs_val=np.abs(dfeats_tf-dfeats)
    if np.mean(abs_val) > 1e-5 or np.max(abs_val) > 1e-4:
        print 'error!'
        print pn, fd
        exit(0)
    abs_val=np.abs(sfeats_tf-sfeats)
    if np.mean(abs_val) > 1e-5 or np.max(abs_val) > 1e-4:
        print 'error!'
        print pn, fd
        exit(0)

def test_copy():

    sess=tf.Session()
    for i in xrange(100):
        pn=np.random.randint(30,1030)
        fd=np.random.randint(30,1030)
        test_copy_single(pn,fd,sess)

    for i in xrange(100):
        pn=np.random.randint(1010,1050)
        fd=np.random.randint(300,500)
        test_copy_single(pn,fd,sess)

    for i in xrange(100):
        pn=np.random.randint(300,500)
        fd=np.random.randint(1010,1050)
        test_copy_single(pn,fd,sess)

    for i in xrange(10):
        pn=np.random.randint(5120,10240)
        fd=np.random.randint(50,100)
        test_copy_single(pn,fd,sess)


def test_diff():
    sess=tf.Session()
    for i in xrange(100):
        pn=np.random.randint(30,1030)
        fd=np.random.randint(30,1030)
        test_diff_single(pn,fd,sess)

    for i in xrange(100):
        pn=np.random.randint(1010,1050)
        fd=np.random.randint(300,500)
        test_diff_single(pn,fd,sess)

    for i in xrange(100):
        pn=np.random.randint(300,500)
        fd=np.random.randint(1010,1050)
        test_diff_single(pn,fd,sess)

    for i in xrange(10):
        pn=np.random.randint(5120,10240)
        fd=np.random.randint(50,100)
        test_diff_single(pn,fd,sess)


def concat_neighboring_feats(feats, nidxs, ncens):
    centering_feats = tf.gather(feats,ncens)
    neighboring_feats = tf.gather(feats,nidxs)
    concat_feats = tf.concat([centering_feats, neighboring_feats], axis=1)
    return concat_feats

def diff_neighboring_feats(feats,nidxs,ncens):
    centering_feats = tf.gather(feats,ncens)
    neighboring_feats = tf.gather(feats,nidxs)
    diff_feats = neighboring_feats-centering_feats
    return diff_feats

def eval_tf_diff_ops_val(feats,dsfeats,nidxs,ncens,sess):
    nidxs_pl=tf.placeholder(tf.int32,[None])
    ncens_pl=tf.placeholder(tf.int32,[None])
    feats_pl=tf.placeholder(tf.float32,[None,None])
    dsfeats_pl=tf.placeholder(tf.float32,[None,None])

    sfeats_op=diff_neighboring_feats(feats_pl, nidxs_pl, ncens_pl)
    dfeats_op=tf.gradients(sfeats_op,feats_pl,dsfeats_pl)[0]

    sfeats,dfeats=sess.run([sfeats_op,dfeats_op],feed_dict={
        nidxs_pl:nidxs,
        ncens_pl:ncens,
        feats_pl:feats,
        dsfeats_pl:dsfeats,
    })
    return sfeats,dfeats

def eval_tf_copy_ops_val(feats,dsfeats,ncens,sess):
    ncens_pl=tf.placeholder(tf.int32,[None])
    feats_pl=tf.placeholder(tf.float32,[None,None])
    dsfeats_pl=tf.placeholder(tf.float32,[None,None])

    sfeats_op=tf.gather(feats_pl,ncens_pl)
    dfeats_op=tf.gradients(sfeats_op,feats_pl,dsfeats_pl)[0]
    sfeats,dfeats=sess.run([sfeats_op,dfeats_op],feed_dict={
        ncens_pl:ncens,
        feats_pl:feats,
        dsfeats_pl:dsfeats
    })
    print 'done'
    return sfeats,dfeats


def test_diff_ops_single(pn,fd,sess):
    feats=np.random.uniform(-1,1,[pn,fd])
    ifeats=np.copy(feats)
    nidxs=[]
    nlens=[]
    for i in xrange(pn):
        num=np.random.randint(0,10)
        nlens.append(num)
        if num>0:
            idxs=np.random.choice(pn,num,False)
            nidxs.append(idxs)

    nlens=np.asarray(nlens)
    nbegs=compute_nidxs_bgs(nlens)
    ncens=compute_cidxs(nlens)
    nidxs=np.concatenate(nidxs,axis=0)

    dsfeats=np.random.uniform(-1,1,[ncens.shape[0],fd])
    sfeats_tf,dfeats_tf=eval_tf_diff_val(ifeats,dsfeats,nidxs,nlens,nbegs,sess)
    sfeats,dfeats=eval_tf_diff_ops_val(ifeats,dsfeats,nidxs,ncens,sess)
    dfeats_new=np.zeros_like(dfeats_tf,np.float32)
    for ii,idx in enumerate(dfeats.indices):
        dfeats_new[idx]+=dfeats.values[ii]
    dfeats=dfeats_new

    abs_val=np.abs(dfeats_tf-dfeats)
    if np.mean(abs_val) > 1e-5 or np.max(abs_val) > 1e-4:
        print 'error!'
        print pn, fd
        exit(0)
    print np.max(abs_val)
    abs_val=np.abs(sfeats_tf-sfeats)
    if np.mean(abs_val) > 1e-5 or np.max(abs_val) > 1e-4:
        print 'error!'
        print pn, fd
        exit(0)
    print np.max(abs_val)

def test_diff_ops():
    sess=tf.Session()
    for i in xrange(100):
        pn=np.random.randint(30,1030)
        fd=np.random.randint(30,1030)
        test_diff_ops_single(pn,fd,sess)

    for i in xrange(100):
        pn=np.random.randint(1010,1050)
        fd=np.random.randint(300,500)
        test_diff_ops_single(pn,fd,sess)

    for i in xrange(100):
        pn=np.random.randint(300,500)
        fd=np.random.randint(1010,1050)
        test_diff_ops_single(pn,fd,sess)

    for i in xrange(10):
        pn=np.random.randint(5120,10240)
        fd=np.random.randint(50,100)
        test_diff_ops_single(pn,fd,sess)

def test_copy_ops_single(pn,fd,sess):
    feats=np.random.uniform(-1,1,[pn,fd])
    nidxs=[]
    nlens=[]
    for i in xrange(pn):
        num=np.random.randint(0,10)
        nlens.append(num)
        if num>0:
            idxs=np.random.choice(pn,num,False)
            nidxs.append(idxs)

    nlens=np.asarray(nlens)
    nbegs=compute_nidxs_bgs(nlens)
    ncens=compute_cidxs(nlens)
    nidxs=np.concatenate(nidxs,axis=0)

    dsfeats=np.random.uniform(-1,1,[ncens.shape[0],fd])
    sfeats,dfeats=eval_tf_copy_ops_val(feats,dsfeats,nidxs,sess)
    sfeats_tf,dfeats_tf=eval_tf_val(feats,dsfeats,nidxs,nlens,nbegs,sess)
    dfeats_new=np.zeros_like(dfeats_tf,np.float32)
    for ii,idx in enumerate(dfeats.indices):
        dfeats_new[idx]+=dfeats.values[ii]
    dfeats=dfeats_new

    abs_val=np.abs(sfeats_tf-sfeats)
    if np.mean(abs_val) > 1e-5 or np.max(abs_val) > 1e-4:
        print 'forward error!'
        # print abs_val
        print pn, fd
        exit(0)

    abs_val=np.abs(dfeats_tf-dfeats)
    if np.mean(abs_val) > 1e-5 or np.max(abs_val) > 1e-4:
        print 'backward error!'
        # print abs_val
        print pn, fd
        exit(0)


def test_copy_ops():
    sess=tf.Session()
    for i in xrange(100):
        pn=np.random.randint(30,1030)
        fd=np.random.randint(30,1030)
        test_copy_ops_single(pn,fd,sess)

    for i in xrange(100):
        pn=np.random.randint(1010,1050)
        fd=np.random.randint(300,500)
        test_copy_ops_single(pn,fd,sess)

    for i in xrange(100):
        pn=np.random.randint(300,500)
        fd=np.random.randint(1010,1050)
        test_copy_ops_single(pn,fd,sess)

    for i in xrange(10):
        pn=np.random.randint(5120,10240)
        fd=np.random.randint(50,100)
        test_copy_ops_single(pn,fd,sess)

if __name__=="__main__":
    # test_diff_np(50,100)
    # test_copy_ops()
    # sess=tf.Session()
    # test_copy_ops_single(10,3,sess)
    test_copy_ops()
