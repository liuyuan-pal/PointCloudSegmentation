from np_ops import np_concat_non_center_forward,np_concat_non_center_backward
from data_util import compute_nidxs_bgs
import numpy as np
import sys
sys.path.append('..')
sys.path.append('../..')
import libPointUtil
import graph_conv_layer
import tensorflow as tf

def eval_concat_non_center_forward(ifeats,nidxs,nidxs_lens,nidxs_bgs,dofeats):
    ifeats_pl=tf.placeholder(tf.float32,[None,None])
    dofeats_pl=tf.placeholder(tf.float32,[None,None])
    nidxs_pl=tf.placeholder(tf.int32,[None])
    nidxs_lens_pl=tf.placeholder(tf.int32,[None],)
    nidxs_bgs_pl=tf.placeholder(tf.int32,[None])


    ofeats=graph_conv_layer.graph_concat_non_center_scatter(ifeats_pl,nidxs_pl,nidxs_lens_pl,nidxs_bgs_pl)
    difeats=tf.gradients(ofeats,ifeats_pl,dofeats_pl)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.Session(config=config) as sess:
        ofeats_val,difeats_val=sess.run(
            [ofeats,difeats],feed_dict={
                ifeats_pl:ifeats,
                nidxs_pl:nidxs,
                nidxs_lens_pl:nidxs_lens,
                nidxs_bgs_pl:nidxs_bgs,
                dofeats_pl:dofeats
            }
        )

    return ofeats_val,difeats_val


def tesst_concat_non_center_scatter():
    pts=np.random.uniform(-1.0,1.0,[20,3])
    pts=np.asarray(pts,dtype=np.float32)
    nidxs=libPointUtil.findNeighborRadiusGPU(pts,1.0)
    nidxs_lens=np.asarray([len(idxs) for idxs in nidxs],dtype=np.int32)
    nidxs_bgs=compute_nidxs_bgs(nidxs_lens)
    nidxs=np.concatenate(nidxs,axis=0)
    nidxs=np.asarray(nidxs,dtype=np.int32)

    pts=np.asarray(pts,np.float64)
    ofeats=np_concat_non_center_forward(pts,nidxs,nidxs_lens,nidxs_bgs)
    dofeats=np.random.uniform(-1,1,ofeats.shape)
    difeats=np_concat_non_center_backward(dofeats,nidxs,nidxs_lens,nidxs_bgs)

    ifeats=np.ascontiguousarray(pts,np.float32)
    dofeats=np.ascontiguousarray(dofeats,np.float32)
    ofeats_tf,difeats_tf=eval_concat_non_center_forward(ifeats,nidxs,nidxs_lens,nidxs_bgs,dofeats)

    print 'forward diff {}'.format(np.max(ofeats_tf-ofeats))
    print 'backward diff {}'.format(np.max(np.abs(difeats_tf-difeats)))
    print np.mean(difeats_tf)

if __name__=="__main__":
    tesst_concat_non_center_scatter()