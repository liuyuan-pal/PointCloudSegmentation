import tensorflow as tf
from model_pooling import context_points_pooling_two_layers,graph_conv_pool_edge_simp_2layers_s3d,\
    classifier_v3,context_points_pooling,graph_conv_pool_context_with_pool
from train_gpn_semantic3d_context import build_placeholder,fill_feed_dict
from io_util import read_room_pkl,get_semantic3d_testset,read_pkl,save_pkl
from aug_util import compute_nidxs_bgs,rotate
from draw_util import output_points,get_semantic3d_class_colors
import numpy as np
import argparse
import libPointUtil
import time
import pyflann
import os
import sys

parser = argparse.ArgumentParser()

parser.add_argument('--restore_model', type=str, default='', help='')
parser.add_argument('--rotation_index', type=int, default=0, help='')
parser.add_argument('--num_classes', type=int, default=9, help='')

FLAGS = parser.parse_args()


def build_network(xyzs, feats, ctx_pts, ctx_idxs, labels, is_training,reuse=False):

    with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
        xyzs, dxyzs, feats, labels, vlens, vbegs, vcens, ctx_idxs = \
            context_points_pooling_two_layers(xyzs,feats,labels,ctx_idxs,voxel_size1=0.25,voxel_size2=1.0,block_size=10.0)
        global_feats,local_feats,ops=graph_conv_pool_edge_simp_2layers_s3d(xyzs, dxyzs, feats, vlens, vbegs, vcens,
                                                                           [0.25, 1.0], 10.0, [0.25,0.5,2.0], reuse)
        ctx_xyzs,ctx_feats=tf.split(ctx_pts,[3,13],axis=1)

        ctx_xyzs,ctx_pxyzs,ctx_dxyzs,ctx_feats,ctx_vlens,ctx_vbegs,ctx_vcens,ctx_idxs=\
            context_points_pooling(ctx_xyzs,ctx_feats,ctx_idxs,5,300)
        ctx_graph_feats=graph_conv_pool_context_with_pool(ctx_xyzs,ctx_dxyzs,ctx_pxyzs,ctx_feats,ctx_vlens,ctx_vbegs,ctx_vcens,
                                                          voxel_size=5.0,block_size=50.0,radius1=5.0,radius2=15.0,reuse=reuse)
        ctx_graph_feats=tf.gather(ctx_graph_feats,ctx_idxs)

        global_feats=tf.concat([global_feats,ctx_graph_feats],axis=1)
        global_feats_exp=tf.expand_dims(global_feats,axis=0)
        local_feats_exp=tf.expand_dims(local_feats,axis=0)
        logits=classifier_v3(global_feats_exp, local_feats_exp, is_training,
                             FLAGS.num_classes, reuse, use_bn=False)  # [1,pn,num_classes]

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
    ops=build_network(pls['xyzs'][0],pls['feats'][0],pls['ctx_pts'][0],pls['ctx_idxs'][0],pls['lbls'][0],pls['is_training'])

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


def eval_room_probs(fn,sess,pls,ops,feed_dict):
    all_xyzs, all_lbls, all_probs = [], [], []
    all_feed_in=read_pkl('data/Semantic3D.Net/context/test_block_avg/'+fn+'.pkl')
    for i in xrange(len(all_feed_in[0])):
        cur_feed_in=[[fi[i]] for fi in all_feed_in]
        block_min=all_feed_in[-1][i]
        fill_feed_dict(cur_feed_in,feed_dict,pls,1)
        probs,sxyzs,lbls=sess.run([ops['probs'],ops['xyzs'],ops['labels']],feed_dict)
        all_xyzs.append(sxyzs+block_min)
        all_lbls.append(lbls)
        all_probs.append(probs)

    return np.concatenate(all_xyzs,axis=0),np.concatenate(all_lbls,axis=0),np.concatenate(all_probs,axis=0)


def interpolate(sxyzs,sprobs,qxyzs,ratio=1.0/(2*0.125*0.125)):
    nidxs=libPointUtil.findNeighborInAnotherCPU(sxyzs,qxyzs,8)
    nidxs_lens=np.asarray([len(idxs) for idxs in nidxs],dtype=np.int32)
    nidxs_bgs=compute_nidxs_bgs(nidxs_lens)
    nidxs=np.concatenate(nidxs,axis=0)
    qprobs = libPointUtil.interpolateProbsGPU(sxyzs, qxyzs, sprobs, nidxs, nidxs_lens, nidxs_bgs, ratio)

    return qprobs

def save_results(sxyzs,qxyzs,sprobs,qprobs,prefix,fs):
    colors=get_semantic3d_class_colors()
    spreds=np.argmax(sprobs[:,1:],axis=1)+1
    qpreds=np.argmax(qprobs[:,1:],axis=1)+1

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

if __name__=="__main__":
    sess, pls, ops, feed_dict = build_session()

    with open('cached/semantic3d_test_stems.txt','r') as f:
        lines=f.readlines()
        fss=[fn.strip('\n').split(' ')[0] for fn in lines]

    prefix='context_avg_result_32'

    for fs in fss:
        sxyzs,_,sprobs=eval_room_probs(fs,sess, pls, ops, feed_dict)
        qxyzs,_=read_pkl('data/Semantic3D.Net/pkl/test/{}.pkl'.format(fs))
        qxyzs=np.ascontiguousarray(qxyzs[:,:3],np.float32)
        sxyzs=np.ascontiguousarray(sxyzs[:,:3],np.float32)
        sprobs=np.ascontiguousarray(sprobs,np.float32)
        qprobs=interpolate(sxyzs,sprobs,qxyzs)

        save_results(sxyzs,qxyzs,sprobs,qprobs,prefix,fs)

