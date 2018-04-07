import numpy as np
from np_ops import eval_numerical_gradient_array
import sys
sys.path.append('..')
from graph_pooling_layer import permutate_feats
import tensorflow as tf

def permutate_forward(feats,maps):
    out_feats=np.empty_like(feats)
    for i,m in enumerate(maps):
        out_feats[i,:]=feats[m,:]

    return out_feats

def test_permutate_np():
    o2p=np.arange(50)
    np.random.shuffle(o2p)
    p2o=np.empty_like(o2p)
    for i,v in enumerate(o2p):
        p2o[v]=i

    feats=np.random.uniform(-1,1,[50,128])
    pfeats=permutate_forward(feats,o2p)
    dpfeats=np.random.uniform(-1,1,[50,128])
    dfeats=permutate_forward(dpfeats,p2o)

    f=lambda feats:permutate_forward(feats,o2p)
    dfeats_eval=eval_numerical_gradient_array(f,feats,dpfeats)

    abs_diff=np.abs(dfeats_eval-dfeats)
    print 'diff {}'.format(np.mean(abs_diff),np.max(abs_diff))

def eval_permutate(feats,dpfeats,o2p,p2o,sess):
    feats_pl=tf.placeholder(tf.float32,[None,None])
    o2p_pl=tf.placeholder(tf.int32,[None])
    p2o_pl=tf.placeholder(tf.int32,[None])
    dpfeats_pl=tf.placeholder(tf.float32,[None,None])

    pfeats=permutate_feats(feats_pl,o2p_pl,p2o_pl)
    dfeats=tf.gradients(pfeats,feats_pl,dpfeats_pl)

    dfeats_val,pfeats_val=sess.run([dfeats,pfeats],feed_dict={
        feats_pl:feats,
        o2p_pl:o2p,
        p2o_pl:p2o,
        dpfeats_pl:dpfeats,
    })

    return dfeats_val,pfeats_val


def test_single(pn,fd,sess):
    o2p=np.arange(pn)
    np.random.shuffle(o2p)
    p2o=np.empty_like(o2p)
    for i,v in enumerate(o2p):
        p2o[v]=i
    feats=np.random.uniform(-1,1,[pn,fd])
    pfeats=permutate_forward(feats,o2p)
    dpfeats=np.random.uniform(-1,1,[pn,fd])
    dfeats=permutate_forward(dpfeats,p2o)
    dfeats_tf, pfeats_tf = eval_permutate(feats, dpfeats, o2p, p2o,sess)

    # f=lambda feats:permutate_forward(feats,o2p)
    # dfeats_eval=eval_numerical_gradient_array(f,feats,dpfeats)
    # abs_diff=np.abs(dfeats_eval-dfeats)
    # print 'diff {} {}'.format(np.mean(abs_diff),np.max(abs_diff))

    abs_diff=np.abs(dfeats_tf-dfeats)
    if np.mean(abs_diff) > 1e-5 or np.max(abs_diff) > 1e-4:
        print 'error!'
        print pn, fd
        exit(0)

    abs_diff=np.abs(pfeats_tf-pfeats)
    if np.mean(abs_diff) > 1e-5 or np.max(abs_diff) > 1e-4:
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

    for i in xrange(10):
        pn=np.random.randint(5120,10240)
        fd=np.random.randint(50,100)
        test_single(pn,fd,sess)




if __name__=="__main__":
    test()