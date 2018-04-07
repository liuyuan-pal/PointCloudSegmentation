import sys
sys.path.append('..')
from graph_pooling_layer import pooling_ops
import tensorflow as tf
import cPickle
import numpy as np


def read_pkl(filename):
    with open(filename,'rb') as f:
        obj=cPickle.load(f)
    return obj

def eval_tf_permutation(pts,sess):
    pts_pl=tf.placeholder(tf.float32,[None,3],'pts') # pn1
    vidxs1=pooling_ops.compute_voxel_index(pts_pl, voxel_len=0.3, block_size=2.0)
    o2p_idxs1, p2o_idxs1, vlens1, vbegs1, vcens1=pooling_ops.compute_permutation_info(vidxs1)
    o2p,p2o,vl,vb,vc=sess.run(
        [o2p_idxs1, p2o_idxs1, vlens1, vbegs1, vcens1],feed_dict={pts_pl:pts}
    )

    return o2p,p2o,vl,vb,vc


def get_semantic3d_block_train_test_split(
        test_stems=('sg28_station4_intensity_rgb','untermaederbrunnen_station3_xyz_intensity_rgb')
):
    train_list,test_list=[],[]
    with open('../../cached/semantic3d_merged_train.txt','r') as f:
        for line in f.readlines():
            line=line.strip('\n')
            stem=line.split(' ')[0]
            num=int(line.split(' ')[1])
            # print stem
            if stem in test_stems:
                test_list+=[stem+'_{}.pkl'.format(i) for i in xrange(num)]
            else:
                train_list+=[stem+'_{}.pkl'.format(i) for i in xrange(num)]

    return train_list,test_list


def test_single(fn,sess):
    xyzs, rgbs, covars, labels=read_pkl(fn)
    for i in xrange(len(xyzs)):
        o2p, p2o, vl, vb, vc = eval_tf_permutation(xyzs[i],sess)
        assert np.sum(o2p>=len(xyzs[i]))==0
        assert np.sum(p2o>=len(xyzs[i]))==0
        assert np.sum(np.sort(o2p)!=np.arange(len(xyzs[i])))==0
        assert np.sum(np.sort(p2o)!=np.arange(len(xyzs[i])))==0


train_list,test_list=get_semantic3d_block_train_test_split()
train_list = ['data/Semantic3D.Net/block/sampled/merged/' + fn for fn in train_list]
test_list = ['data/Semantic3D.Net/block/sampled/merged/' + fn for fn in test_list]
train_list += test_list

sess = tf.Session()
for i in xrange(30):
    test_single(train_list[i],sess)
