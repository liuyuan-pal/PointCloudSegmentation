import tensorflow as tf
from model_pooling import *
from model_pointnet_semantic3d import *
from train_gpn_semantic3d_dense import build_placeholder,fill_feed_dict,train_fn
from io_util import read_room_pkl,get_semantic3d_testset,read_pkl,save_pkl
from aug_util import compute_nidxs_bgs,rotate
from draw_util import output_points,get_semantic3d_class_colors
import numpy as np
import argparse
import libPointUtil
import time
import pyflann
import os

parser = argparse.ArgumentParser()

parser.add_argument('--restore_model', type=str, default='', help='')
parser.add_argument('--prefix', type=str, default='pointnet_13_dilated_embed_23_revise', help='')
parser.add_argument('--rotation_index', type=int, default=0, help='')
parser.add_argument('--num_classes', type=int, default=8, help='')

FLAGS = parser.parse_args()


def build_network(xyzs, feats, labels, idxs, nidxs, nlens, nbegs, ncens):
    xyzs, feats, labels = dense_feats(xyzs, feats, labels, idxs, nidxs, nlens, nbegs, ncens, False)
    xyzs, dxyzs, feats, labels, vlens, vbegs, vcens = \
        points_pooling_two_layers(xyzs,feats,labels, 0.45, 1.5, 10.0)
    global_feats,local_feats=pointnet_13_dilate_embed_semantic3d(xyzs,dxyzs,feats,vlens,vbegs,vcens,False)

    global_feats_exp=tf.expand_dims(global_feats,axis=0)
    local_feats_exp=tf.expand_dims(local_feats,axis=0)
    logits=classifier_v3(global_feats_exp, local_feats_exp, tf.constant(False),
                         FLAGS.num_classes, False, use_bn=False)  # [1,pn,num_classes]

    flatten_logits=tf.reshape(logits,[-1,FLAGS.num_classes])  # [pn,num_classes]
    probs=tf.nn.softmax(flatten_logits)
    preds=tf.argmax(flatten_logits,axis=1)
    ops={}
    ops['feats']=feats
    ops['probs']=probs
    ops['preds']=preds
    ops['labels']=labels
    ops['xyzs']=xyzs[0]
    return ops


def build_session():
    pls=build_placeholder(1)
    ops=build_network(pls['xyzs'][0],pls['feats'][0],pls['lbls'][0],pls['idxs'][0],
                      pls['nidxs'][0],pls['nlens'][0],pls['nbegs'][0],pls['ncens'][0])

    feed_dict=dict()
    feed_dict[pls['is_training']]=False
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    saver = tf.train.Saver(max_to_keep=500)
    print 'restoring {}'.format(FLAGS.restore_model)
    saver.restore(sess,FLAGS.restore_model)
    return sess, pls, ops, feed_dict


def get_cur_feed_in(all_feed_in,idx):
    xyzs, rgbs, lbls, nidxs, nlens, nbegs, ncens, ds_idxs, block_mins=all_feed_in
    return [xyzs[idx]], [rgbs[idx]],[lbls[idx]],[nidxs[idx]],[nlens[idx]], \
           [nbegs[idx]], [ncens[idx]], [ds_idxs[idx]], [block_mins[idx]]


def eval_room_probs(fn,sess,pls,ops,feed_dict):
    all_xyzs, all_lbls, all_probs = [], [], []
    all_feed_in=train_fn('test','data/Semantic3D.Net/block/test_dense/{}.pkl'.format(fn))
    for i in xrange(len(all_feed_in[0])):
        cur_feed_in=get_cur_feed_in(all_feed_in,i)
        block_min=all_feed_in[-1][i]
        fill_feed_dict(cur_feed_in,feed_dict,pls,1)
        probs,sxyzs,lbls=sess.run([ops['probs'],ops['xyzs'],ops['labels']],feed_dict)
        all_xyzs.append(sxyzs+block_min)
        all_lbls.append(lbls)
        all_probs.append(probs)

    return np.concatenate(all_xyzs,axis=0),np.concatenate(all_lbls,axis=0),np.concatenate(all_probs,axis=0)


def interpolate(sxyzs,sprobs,qxyzs,ratio=1.0/(2*0.15*0.15)):
    nidxs=libPointUtil.findNeighborInAnotherCPU(sxyzs,qxyzs,8)
    nidxs_lens=np.asarray([len(idxs) for idxs in nidxs],dtype=np.int32)
    nidxs_bgs=compute_nidxs_bgs(nidxs_lens)
    nidxs=np.concatenate(nidxs,axis=0)
    qprobs = libPointUtil.interpolateProbsGPU(sxyzs, qxyzs, sprobs, nidxs, nidxs_lens, nidxs_bgs, ratio)

    return qprobs

def read_natural_terrain():
    with open('cached/sg28-natural-terrain-v2.txt', 'r') as f:
        natural_xyzs=[]
        for line in f.readlines():
            parts=line.split(' ')
            x=float(parts[0])
            y=float(parts[1])
            z=float(parts[2])
            natural_xyzs.append([x,y,z])

    natural_xyzs=np.asarray(natural_xyzs,np.float32)
    return natural_xyzs

def interpolate_natural_terrain(sxyzs,qxyzs,qprobs):
    print sxyzs.shape,qxyzs.shape,qprobs.shape
    nidxs=libPointUtil.findNeighborInAnotherCPU(sxyzs,qxyzs,0.22)
    nlens=np.asarray([len(idxs) for idxs in nidxs],dtype=np.int32)
    qpreds=np.argmax(qprobs,axis=1)
    mask=np.logical_and(nlens>0,qpreds==0)
    qprobs[mask,0]=0.0
    qprobs[mask,1]=1.0

    return qprobs

def save_results(sxyzs,qxyzs,sprobs,qprobs,prefix,fs):
    colors=get_semantic3d_class_colors()
    spreds=np.argmax(sprobs,axis=1)+1
    qpreds=np.argmax(qprobs,axis=1)+1

    dir='data/Semantic3D.Net/{}'.format(prefix)
    if not os.path.exists(dir): os.mkdir(dir)
    with open('{}/{}.labels'.format(dir,fs),'w') as f:
        for pred in qpreds:
            f.write('{}\n'.format(pred))

    idxs=libPointUtil.gridDownsampleGPU(sxyzs,0.3,False)
    sxyzs=sxyzs[idxs]
    spreds=spreds[idxs]
    output_points('{}/{}_sparse.txt'.format(dir,fs),sxyzs,colors[spreds])

    idxs=libPointUtil.gridDownsampleGPU(qxyzs,0.3,False)
    qxyzs=qxyzs[idxs]
    qpreds=qpreds[idxs]
    output_points('{}/{}_dense.txt'.format(dir,fs),qxyzs,colors[qpreds])

def process():
    sess, pls, ops, feed_dict = build_session()
    with open('cached/semantic3d_test_stems.txt','r') as f:
        lines=f.readlines()
        fss=[fn.strip('\n').split(' ')[0] for fn in lines]

    for fs in fss[2:3]:
        sxyzs,_,sprobs=eval_room_probs(fs,sess, pls, ops, feed_dict)
        qxyzs,_=read_pkl('data/Semantic3D.Net/pkl/test/{}.pkl'.format(fs))
        qxyzs=np.ascontiguousarray(qxyzs[:,:3],np.float32)
        sxyzs=np.ascontiguousarray(sxyzs[:,:3],np.float32)
        sprobs=np.ascontiguousarray(sprobs,np.float32)
        qprobs=interpolate(sxyzs,sprobs,qxyzs)

        save_results(sxyzs,qxyzs,sprobs,qprobs,FLAGS.prefix,fs)

        save_pkl('cached/sg28_qxyzs.pkl',qxyzs)
        save_pkl('cached/sg28_qprobs.pkl',qprobs)

def process_revise():
    qxyzs=read_pkl('cached/sg28_qxyzs.pkl')
    qprobs=read_pkl('cached/sg28_qprobs.pkl')
    sxyzs=read_natural_terrain()

    qprobs=interpolate_natural_terrain(sxyzs,qxyzs,qprobs)

    save_results(sxyzs, qxyzs, np.random.uniform(0,1.0,[sxyzs.shape[0],8]),
                 qprobs, FLAGS.prefix, 'sg28_revise')

if __name__=="__main__":
    process_revise()