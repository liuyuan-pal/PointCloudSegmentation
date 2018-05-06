from tf_ops.graph_layer_new import *
import tensorflow.contrib.framework as framework
import argparse
from train_util import compute_iou,log_str
from io_util import read_pkl,get_block_train_test_split
import numpy as np
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from draw_util import output_points,get_class_colors

parser = argparse.ArgumentParser()
parser.add_argument('--radius1', type=float, default=0.2, help='')
parser.add_argument('--radius2', type=float, default=0.1, help='')
parser.add_argument('--model', type=str, default='pointnet', help='')
parser.add_argument('--epoch_num', type=int, default=60, help='')
parser.add_argument('--decay_steps', type=int, default=10000, help='')
parser.add_argument('--gpu_id', type=int, default=0, help='')

FLAGS = parser.parse_args()

feats_ops={}

def classifier(feats,labels):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=False):
        feats = tf.contrib.layers.fully_connected(feats, num_outputs=128, scope='cls1')
        feats = tf.contrib.layers.fully_connected(feats, num_outputs=64, scope='cls2')
        logits = tf.contrib.layers.fully_connected(feats, num_outputs=13, scope='cls3', activation_fn=None)
        loss=tf.losses.sparse_softmax_cross_entropy(labels,logits)
        preds=tf.argmax(logits,axis=1)

        return logits,loss,preds


def pointnet_model(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1
    feats1=pointnet_conv(sxyzs,feats,[4,4,8],16,'pointnet1',nidxs,nlens,nbegs,ncens,False)
    feats_ops['feats1']=feats1
    feats=tf.concat([feats,feats1],axis=1)
    feats1=pointnet_conv(sxyzs,feats,[4,4,8],16,'pointnet2',nidxs,nlens,nbegs,ncens,False)
    feats_ops['feats2']=feats1
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=pointnet_conv(sxyzs,feats,[8,8,16],32,'pointnet3',nidxs,nlens,nbegs,ncens,False)
    feats_ops['feats3']=feats2
    feats = tf.concat([feats,feats2], axis=1)

    return feats


def pointnet_model_no_concat(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1
    feats1=pointnet_conv_v2(sxyzs,feats,[8,8,16],16,'pointnet1',nidxs,nlens,nbegs,ncens,False)
    feats=tf.concat([feats,feats1],axis=1)
    feats1=pointnet_conv_v2(sxyzs,feats,[8,8,16],16,'pointnet2',nidxs,nlens,nbegs,ncens,False)
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=pointnet_conv_v2(sxyzs,feats,[16,16,32],32,'pointnet3',nidxs,nlens,nbegs,ncens,False)
    feats = tf.concat([feats,feats2], axis=1)

    return feats

def diff_ecd_model(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1=diff_xyz_ecd(sxyzs,3,[4,4],[4,4],16,nidxs,nlens,nbegs,ncens,'xyz',False)
    feats=tf.concat([feats,feats1],axis=1)
    feats1=diff_feats_ecd(sxyzs,feats,19,[4,4],[4,4],16,nidxs,nlens,nbegs,ncens,'feats1',False)
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=diff_feats_ecd(sxyzs,feats,35,[8,8],[8,8],32,nidxs,nlens,nbegs,ncens,'feats2',False)
    feats = tf.concat([feats,feats2], axis=1)
    return feats

def pointnet_diff_ecd_model(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1 = pointnet_conv(sxyzs, feats, [4, 4, 8], 16, 'pointnet1', nidxs, nlens, nbegs, ncens, False)
    feats=tf.concat([feats,feats1],axis=1)
    feats1=diff_feats_ecd(sxyzs,feats,19,[4,4],[4,4],16,nidxs,nlens,nbegs,ncens,'feats1',False)
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=diff_feats_ecd(sxyzs,feats,35,[8,8],[8,8],32,nidxs,nlens,nbegs,ncens,'feats2',False)
    feats = tf.concat([feats,feats2], axis=1)
    return feats

def pointnet_diff_ecd_model_v2(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1 = pointnet_conv(sxyzs, feats, [4, 4, 8], 16, 'pointnet1', nidxs, nlens, nbegs, ncens, False)
    feats_ops['feats1']=feats1
    feats=tf.concat([feats,feats1],axis=1)
    feats1=diff_feats_ecd_v2(sxyzs,feats,19,[16],[],16,nidxs,nlens,nbegs,ncens,'feats1',False)
    feats_ops['feats2']=feats1
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=diff_feats_ecd_v2(sxyzs,feats,35,[32],[], 32,nidxs,nlens,nbegs,ncens,'feats2',False)
    feats = tf.concat([feats,feats2], axis=1)
    # feats2=diff_feats_ecd_v2(sxyzs,feats,67,[32],[], 32,nidxs,nlens,nbegs,ncens,'feats3',False)
    # feats = tf.concat([feats,feats2], axis=1)
    # feats2=diff_feats_ecd_v2(sxyzs,feats,99,[32],[], 32,nidxs,nlens,nbegs,ncens,'feats4',False)
    feats_ops['feats3']=feats2
    feats = tf.concat([feats,feats2], axis=1)
    return feats

def pointnet_diff_ecd_model_v3(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1 = pointnet_conv(sxyzs, feats, [4, 4, 8], 16, 'pointnet1', nidxs, nlens, nbegs, ncens, False)
    feats_ops['feats1']=feats1
    feats=tf.concat([feats,feats1],axis=1)
    feats1=diff_feats_ecd_v2(sxyzs,feats,19,[16],[],16,nidxs,nlens,nbegs,ncens,'feats1',use_l2_norm=False,reuse=False)
    feats_ops['feats2']=feats1
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=diff_feats_ecd_v2(sxyzs,feats,35,[32],[],32,nidxs,nlens,nbegs,ncens,'feats2',use_l2_norm=False,reuse=False)
    feats_ops['feats3']=feats2
    feats = tf.concat([feats,feats2], axis=1)
    return feats

def pointnet_diff_ecd_model_v4(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1 = pointnet_conv(sxyzs, feats, [4, 4, 8], 16, 'pointnet1', nidxs, nlens, nbegs, ncens, False)
    feats_ops['feats1']=feats1
    feats=tf.concat([feats,feats1],axis=1)
    feats1=diff_feats_ecd_v2(sxyzs,feats,19,[16],[8,8],16,nidxs,nlens,nbegs,ncens,'feats1',False)
    feats_ops['feats2']=feats1
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=diff_feats_ecd_v2(sxyzs,feats,35,[32],[16,16],32,nidxs,nlens,nbegs,ncens,'feats2',False)
    feats_ops['feats3']=feats2
    feats = tf.concat([feats,feats2], axis=1)
    return feats

def pointnet_diff_ecd_model_v5(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1 = pointnet_conv(sxyzs, feats, [4, 4, 8], 16, 'pointnet1', nidxs, nlens, nbegs, ncens, False)
    feats_ops['feats1']=feats1
    feats=tf.concat([feats,feats1],axis=1)
    feats1=diff_feats_ecd_v2(sxyzs,feats,19,[16],[],16,nidxs,nlens,nbegs,ncens,'feats1',final_activation=tf.nn.relu,reuse=False)
    feats_ops['feats2']=feats1
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=diff_feats_ecd_v2(sxyzs,feats,35,[32],[],32,nidxs,nlens,nbegs,ncens,'feats2',final_activation=tf.nn.relu,reuse=False)
    feats_ops['feats3']=feats2
    feats = tf.concat([feats,feats2], axis=1)
    return feats

def pointnet_diff_ecd_model_v6(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1 = pointnet_conv(sxyzs, feats, [4, 4, 8], 16, 'pointnet1', nidxs, nlens, nbegs, ncens, False)
    feats_ops['feats1']=feats1
    feats=tf.concat([feats,feats1],axis=1)
    feats1=diff_feats_ecd_v2(sxyzs,feats,19,[8,8],[],16,nidxs,nlens,nbegs,ncens,'feats1',reuse=False)
    feats_ops['feats2']=feats1
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=diff_feats_ecd_v2(sxyzs,feats,35,[16,16],[],32,nidxs,nlens,nbegs,ncens,'feats2',reuse=False)
    feats_ops['feats3']=feats2
    feats = tf.concat([feats,feats2], axis=1)
    return feats

def pointnet_diff_ecd_model_v7(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1 = pointnet_conv(sxyzs, feats, [4, 4, 8], 16, 'pointnet1', nidxs, nlens, nbegs, ncens, False)
    feats_ops['feats1']=feats1
    feats=tf.concat([feats,feats1],axis=1)
    feats1=diff_feats_ecd_v2(sxyzs,feats,19,[16],[],16,nidxs,nlens,nbegs,ncens,'feats1',use_l2_norm=False,
                             weight_activation=tf.nn.relu,reuse=False)
    feats_ops['feats2']=feats1
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=diff_feats_ecd_v2(sxyzs,feats,35,[32],[],32,nidxs,nlens,nbegs,ncens,'feats2',use_l2_norm=False,
                             weight_activation=tf.nn.relu,reuse=False)
    feats_ops['feats3']=feats2
    feats = tf.concat([feats,feats2], axis=1)
    return feats

def diff_ecd_model_v8(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1=tf.contrib.layers.fully_connected(feats, num_outputs=16,scope='embed0',activation_fn=tf.nn.relu, reuse=False)
    feats1=diff_feats_ecd_v2(sxyzs,feats1,16,[16],[],16,nidxs,nlens,nbegs,ncens,'feats0',reuse=False)
    feats_ops['feats1']=feats1
    feats=tf.concat([feats,feats1],axis=1)
    feats1=diff_feats_ecd_v2(sxyzs,feats,19,[16],[],16,nidxs,nlens,nbegs,ncens,'feats1',reuse=False)
    feats_ops['feats2']=feats1
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=diff_feats_ecd_v2(sxyzs,feats,35,[32],[],32,nidxs,nlens,nbegs,ncens,'feats2',reuse=False)
    feats_ops['feats3']=feats2
    feats = tf.concat([feats,feats2], axis=1)
    return feats

def concat_ecd_model(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1=diff_xyz_ecd(sxyzs,3,[4,4],[4,4],16,nidxs,nlens,nbegs,ncens,'xyz',False)
    feats=tf.concat([feats,feats1],axis=1)
    feats1=concat_feats_ecd(sxyzs,feats,19,[4,4],[4,4],16,'feats1',nidxs,nlens,nbegs,ncens,False)
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2

    feats2=concat_feats_ecd(sxyzs,feats,35,[8,8],[8,8],32,'feats2',nidxs,nlens,nbegs,ncens,False)
    feats = tf.concat([feats,feats2], axis=1)
    return feats

def pointnet_concat_ecd_model(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1
    feats1=diff_xyz_ecd(sxyzs,3,[4,4],[4,4],16,nidxs,nlens,nbegs,ncens,'xyz',False)
    feats=tf.concat([feats,feats1],axis=1)
    feats1=concat_feats_ecd(sxyzs,feats,19,[4,4],[4,4],16,'feats1',nidxs,nlens,nbegs,ncens,False)
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=concat_feats_ecd(sxyzs,feats,35,[8,8],[8,8],32,'feats2',nidxs,nlens,nbegs,ncens,False)
    feats = tf.concat([feats,feats2], axis=1)
    return feats

def anchor_conv_model(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1=pointnet_conv(sxyzs, feats, [4, 4, 8], 16, 'pointnet1', nidxs, nlens, nbegs, ncens, False)
    feats=tf.concat([feats,feats1],axis=1)
    feats1=tf.contrib.layers.fully_connected(feats, num_outputs=8,scope='embed1',activation_fn=tf.nn.relu, reuse=False)
    feats1=anchor_conv(sxyzs,feats1,8,16,9,'feats1',nidxs,nlens,nbegs,ncens,False)
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=tf.contrib.layers.fully_connected(feats, num_outputs=16,scope='embed2',activation_fn=tf.nn.relu, reuse=False)
    feats2=anchor_conv(sxyzs,feats2,16,32,9,'feats2',nidxs,nlens,nbegs,ncens,False)
    feats = tf.concat([feats,feats2], axis=1)
    return feats

def mlp_anchor_conv_model(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1=pointnet_conv(sxyzs, feats, [4, 4, 8], 16, 'pointnet1', nidxs, nlens, nbegs, ncens, False)
    feats=tf.concat([feats,feats1],axis=1)
    feats1=tf.contrib.layers.fully_connected(feats, num_outputs=8,scope='embed1',activation_fn=tf.nn.relu, reuse=False)
    feats1=edge_condition_diffusion_anchor(sxyzs,feats1,8,[4,4],16,9,'feats1',nidxs,nlens,nbegs,ncens,False)
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2

    feats2 = tf.contrib.layers.fully_connected(feats, num_outputs=16,scope='embed2',activation_fn=tf.nn.relu, reuse=False)
    feats2=edge_condition_diffusion_anchor(sxyzs,feats2,16,[8,8],32,9,'feats2',nidxs,nlens,nbegs,ncens,False)
    feats = tf.concat([feats,feats2], axis=1)
    return feats

def mlp_anchor_conv_model_v2(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1=anchor_conv_v2(sxyzs,feats,16,27,3,'anchor1',nidxs,nlens,nbegs,ncens,trainable_anchor=False,reuse=False)
    feats_ops['feats1']=feats1
    feats=tf.concat([feats,feats1],axis=1)
    feats1=anchor_conv_v2(sxyzs,feats,16,27,8,'anchor2',nidxs,nlens,nbegs,ncens,trainable_anchor=False,reuse=False)
    feats_ops['feats2'] = feats1
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2

    feats2=anchor_conv_v2(sxyzs,feats,32,27,16,'anchor3',nidxs,nlens,nbegs,ncens,trainable_anchor=False,reuse=False)
    feats_ops['feats3']=feats2
    feats = tf.concat([feats,feats2], axis=1)
    return feats

def mlp_anchor_conv_model_nonorm(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1=pointnet_conv(sxyzs, feats, [4, 4, 8], 16, 'pointnet1', nidxs, nlens, nbegs, ncens, False)
    feats=tf.concat([feats,feats1],axis=1)
    # feats1=tf.contrib.layers.fully_connected(feats, num_outputs=8,scope='embed1',activation_fn=tf.nn.relu, reuse=False)
    feats1=edge_condition_diffusion_anchor_v2(sxyzs,feats,[4,4],16,9,8,'feats1',nidxs,nlens,nbegs,ncens,False)
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    # feats2 = tf.contrib.layers.fully_connected(feats, num_outputs=16,scope='embed2',activation_fn=tf.nn.relu, reuse=False)
    feats2=edge_condition_diffusion_anchor_v2(sxyzs,feats,[8,8],32,9,16,'feats2',nidxs,nlens,nbegs,ncens,False)
    feats = tf.concat([feats,feats2], axis=1)
    return feats

def mlp_anchor_conv_model_v3(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1=pointnet_conv(sxyzs, feats, [4, 4, 8], 16, 'pointnet1', nidxs, nlens, nbegs, ncens, False)
    feats=tf.concat([feats,feats1],axis=1)
    feats1=edge_condition_diffusion_anchor_v3(sxyzs,feats,[16],16,9,8,'feats1',nidxs,nlens,nbegs,ncens,False)
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=edge_condition_diffusion_anchor_v3(sxyzs,feats,[32],32,9,16,'feats2',nidxs,nlens,nbegs,ncens,False)
    feats = tf.concat([feats,feats2], axis=1)
    return feats

def mlp_anchor_conv_model_v4(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1=pointnet_conv(sxyzs, feats, [4, 4, 8], 16, 'pointnet1', nidxs, nlens, nbegs, ncens, False)
    feats=tf.concat([feats,feats1],axis=1)
    feats1=edge_condition_diffusion_anchor_v4(sxyzs,feats,19,[16],16,9,'feats1',nidxs,nlens,nbegs,ncens,False,False)
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=edge_condition_diffusion_anchor_v4(sxyzs,feats,35,[32],32,9,'feats2',nidxs,nlens,nbegs,ncens,False,False)
    feats = tf.concat([feats,feats2], axis=1)
    return feats

def mlp_anchor_conv_model_v5(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1=edge_condition_diffusion_anchor_v4(sxyzs,feats,3,[9],16,27,'feats0',nidxs,nlens,nbegs,ncens,False,False)
    feats=tf.concat([feats,feats1],axis=1)
    feats1=edge_condition_diffusion_anchor_v4(sxyzs,feats,19,[16],16,9,'feats1',nidxs,nlens,nbegs,ncens,False,False)
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=edge_condition_diffusion_anchor_v4(sxyzs,feats,35,[32],32,9,'feats2',nidxs,nlens,nbegs,ncens,False,False)
    feats = tf.concat([feats,feats2], axis=1)
    return feats

def mlp_anchor_conv_model_v6(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1=pointnet_conv(sxyzs, feats, [4, 4, 8], 16, 'pointnet1', nidxs, nlens, nbegs, ncens, False)
    feats=tf.concat([feats,feats1],axis=1)
    feats1=edge_condition_diffusion_anchor_v4(sxyzs,feats,19,[16],16,9,'feats1',nidxs,nlens,nbegs,ncens,True,False)
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=edge_condition_diffusion_anchor_v4(sxyzs,feats,35,[32],32,9,'feats2',nidxs,nlens,nbegs,ncens,True,False)
    feats = tf.concat([feats,feats2], axis=1)
    return feats

def mlp_anchor_conv_model_v7(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1=pointnet_conv(sxyzs, feats, [4, 4, 8], 16, 'pointnet1', nidxs, nlens, nbegs, ncens, False)
    feats=tf.concat([feats,feats1],axis=1)
    feats1=edge_condition_diffusion_anchor_v4(sxyzs,feats,19,[16],16,9,'feats1',nidxs,nlens,nbegs,ncens,False,False,tf.nn.relu)
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=edge_condition_diffusion_anchor_v4(sxyzs,feats,35,[32],32,9,'feats2',nidxs,nlens,nbegs,ncens,False,False,tf.nn.relu)
    feats = tf.concat([feats,feats2], axis=1)
    return feats

def mlp_anchor_conv_model_v8(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1=pointnet_conv(sxyzs, feats, [4, 4, 8], 16, 'pointnet1', nidxs, nlens, nbegs, ncens, False)
    feats=tf.concat([feats,feats1],axis=1)
    feats1=edge_condition_diffusion_anchor_v4(sxyzs,feats,19,[16],16,9,'feats1',nidxs,nlens,nbegs,ncens,False,False,None,tf.nn.relu)
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=edge_condition_diffusion_anchor_v4(sxyzs,feats,35,[32],32,9,'feats2',nidxs,nlens,nbegs,ncens,False,False,None,tf.nn.relu)
    feats = tf.concat([feats,feats2], axis=1)
    return feats

def mlp_anchor_conv_model_v9(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1=pointnet_conv(sxyzs, feats, [4, 4, 8], 16, 'pointnet1', nidxs, nlens, nbegs, ncens, False)
    feats=tf.concat([feats,feats1],axis=1)
    feats1=edge_condition_diffusion_anchor_v4(sxyzs,feats,19,[16],16,9,'feats1',nidxs,nlens,nbegs,ncens,False,False,None,tf.nn.sigmoid)
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=edge_condition_diffusion_anchor_v4(sxyzs,feats,35,[32],32,9,'feats2',nidxs,nlens,nbegs,ncens,False,False,None,tf.nn.sigmoid)
    feats = tf.concat([feats,feats2], axis=1)
    return feats

def mlp_anchor_conv_model_v10(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1=pointnet_conv(sxyzs, feats, [4, 4, 8], 16, 'pointnet1', nidxs, nlens, nbegs, ncens, False)
    feats=tf.concat([feats,feats1],axis=1)
    feats1=edge_condition_diffusion_anchor_v4(sxyzs,feats,19,[16],16,9,'feats1',nidxs,nlens,nbegs,ncens,False,False,tf.nn.leaky_relu)
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=edge_condition_diffusion_anchor_v4(sxyzs,feats,35,[32],32,9,'feats2',nidxs,nlens,nbegs,ncens,False,False,tf.nn.leaky_relu)
    feats = tf.concat([feats,feats2], axis=1)
    return feats


def mlp_anchor_conv_model_v11(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1=pointnet_conv(sxyzs, feats, [4, 4, 8], 16, 'pointnet1', nidxs, nlens, nbegs, ncens, False)
    feats=tf.concat([feats,feats1],axis=1)
    feats1=edge_condition_diffusion_anchor_v4(sxyzs,feats,19,[16],16,9,'feats1',nidxs,nlens,nbegs,ncens,use_concat=True)
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=edge_condition_diffusion_anchor_v4(sxyzs,feats,35,[32],32,9,'feats2',nidxs,nlens,nbegs,ncens,use_concat=True)
    feats = tf.concat([feats,feats2], axis=1)
    return feats


def mlp_anchor_conv_model_v12(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1=pointnet_conv(sxyzs, feats, [4, 4, 8], 16, 'pointnet1', nidxs, nlens, nbegs, ncens, False)
    feats=tf.concat([feats,feats1],axis=1)
    feats1=edge_condition_diffusion_anchor_v4(sxyzs,feats,19,[16],16,9,'feats1',nidxs,nlens,nbegs,ncens,
                                              final_activation=tf.nn.leaky_relu,use_concat=True)
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=edge_condition_diffusion_anchor_v4(sxyzs,feats,35,[32],32,9,'feats2',nidxs,nlens,nbegs,ncens,
                                              final_activation=tf.nn.leaky_relu,use_concat=True)
    feats = tf.concat([feats,feats2], axis=1)
    return feats


def mlp_anchor_conv_model_v13(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1=pointnet_conv(sxyzs, feats, [4, 4, 8], 16, 'pointnet1', nidxs, nlens, nbegs, ncens, False)
    feats=tf.concat([feats,feats1],axis=1)
    feats1=edge_condition_diffusion_anchor_v4(sxyzs,feats,19,[16],16,9,'feats1',nidxs,nlens,nbegs,ncens,
                                              final_activation=tf.nn.relu,use_concat=True)
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=edge_condition_diffusion_anchor_v4(sxyzs,feats,35,[32],32,9,'feats2',nidxs,nlens,nbegs,ncens,
                                              final_activation=tf.nn.relu,use_concat=True)
    feats = tf.concat([feats,feats2], axis=1)
    return feats


def mlp_anchor_conv_model_v14(xyzs,feats):
    nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,FLAGS.radius1)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius1

    feats1=pointnet_conv(sxyzs, feats, [4, 4, 8], 16, 'pointnet1', nidxs, nlens, nbegs, ncens, False)
    feats=tf.concat([feats,feats1],axis=1)
    feats1=edge_condition_diffusion_anchor_v4(sxyzs,feats,19,[16],16,9,'feats1',nidxs,nlens,nbegs,ncens,l2_norm=True,
                                              final_activation=tf.nn.leaky_relu,use_concat=True)
    feats=tf.concat([feats,feats1],axis=1)

    nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs, FLAGS.radius2)
    sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
    sxyzs /= FLAGS.radius2
    feats2=edge_condition_diffusion_anchor_v4(sxyzs,feats,35,[32],32,9,'feats2',nidxs,nlens,nbegs,ncens,l2_norm=True,
                                              final_activation=tf.nn.leaky_relu,use_concat=True)
    feats = tf.concat([feats,feats2], axis=1)
    return feats


def build_model():
    pls={}
    pls['xyzs'] = tf.placeholder(tf.float32, [None, 3], 'xyzs')
    pls['feats'] = tf.placeholder(tf.float32, [None, 3], 'feats')
    pls['labels'] = tf.placeholder(tf.int32, [None], 'labels')
    pls['is_training'] = tf.placeholder(tf.bool, name='is_training')

    with tf.device('gpu:{}'.format(FLAGS.gpu_id)):
        if FLAGS.model=='diff_ecd':
            feats=diff_ecd_model(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='pointnet_diff_ecd_v2':
            feats=pointnet_diff_ecd_model_v2(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='pointnet2':
            feats=pointnet_model_no_concat(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='concat_ecd':
            feats=concat_ecd_model(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='pointnet_diff_ecd':
            feats=pointnet_diff_ecd_model(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='pointnet_concat_ecd':
            feats=pointnet_concat_ecd_model(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='anchor_conv':
            feats=anchor_conv_model(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='anchor_conv_v2':
            feats=mlp_anchor_conv_model_v2(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='mlp_anchor_conv':
            feats=mlp_anchor_conv_model(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='mlp_anchor_conv_nonorm':
            feats=mlp_anchor_conv_model_nonorm(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='mlp_anchor_conv_v3':
            feats=mlp_anchor_conv_model_v3(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='mlp_anchor_conv_v4':
            feats=mlp_anchor_conv_model_v4(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='mlp_anchor_conv_v5':
            feats=mlp_anchor_conv_model_v5(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='mlp_anchor_conv_v6':
            feats=mlp_anchor_conv_model_v6(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='mlp_anchor_conv_v7':
            feats=mlp_anchor_conv_model_v7(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='mlp_anchor_conv_v8':
            feats=mlp_anchor_conv_model_v8(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='mlp_anchor_conv_v9':
            feats=mlp_anchor_conv_model_v9(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='mlp_anchor_conv_v10':
            feats=mlp_anchor_conv_model_v10(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='mlp_anchor_conv_v11':
            feats=mlp_anchor_conv_model_v11(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='mlp_anchor_conv_v12':
            feats=mlp_anchor_conv_model_v12(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='mlp_anchor_conv_v13':
            feats=mlp_anchor_conv_model_v13(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='v3':
            feats=pointnet_diff_ecd_model_v3(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='v4':
            feats=pointnet_diff_ecd_model_v4(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='v5':
            feats=pointnet_diff_ecd_model_v5(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='v6':
            feats=pointnet_diff_ecd_model_v6(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='v7':
            feats=pointnet_diff_ecd_model_v7(pls['xyzs'],pls['feats'])
        elif FLAGS.model=='v8':
            feats=diff_ecd_model_v8(pls['xyzs'],pls['feats'])
        else:
            feats=pointnet_model(pls['xyzs'],pls['feats'])

        logits, loss, preds=classifier(feats,pls['labels'])

        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        lr = tf.train.exponential_decay(1e-3, global_step, FLAGS.decay_steps, 0.5, staircase=True)
        lr = tf.maximum(1e-5, lr)
        opt=tf.train.AdamOptimizer(lr)
        update_op=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_op):
            train_op=opt.minimize(loss,global_step=global_step)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    ops={}
    ops['logits']=logits
    ops['loss']=loss
    ops['preds']=preds
    ops['train']=train_op
    ops['lr']=lr

    return sess,pls,ops

def process_one_epoch(epoch_num,dataset,sess,ops,pls,is_training=True):
    begin_time=time.time()
    xyzs, rgbs, covars, labels, block_mins = dataset

    losses,preds,slabels=[],[],[]

    indices=np.arange(len(xyzs))
    if is_training:
        np.random.shuffle(indices)
    for k in indices:
        feed_dict={}
        feed_dict[pls['xyzs']]=xyzs[k]
        feed_dict[pls['feats']]=rgbs[k]
        feed_dict[pls['labels']]=labels[k]
        feed_dict[pls['is_training']]=is_training
        if is_training:
            _,pred,loss,lr=sess.run([ops['train'],ops['preds'],ops['loss'],ops['lr']],feed_dict=feed_dict)
        else:
            pred,loss = sess.run([ops['preds'], ops['loss']], feed_dict=feed_dict)
        preds.append(pred)
        losses.append(loss)
        slabels.append(labels[k])

    preds_concat=np.concatenate(preds,axis=0)
    slabels=np.concatenate(slabels,axis=0)
    loss=np.mean(losses)

    if is_training:
        log_str('training epoch {} loss {:.3} cost {:.3} s lr {:.3}'.format(
            epoch_num,loss,time.time()-begin_time,lr),FLAGS.model+'.log')
        iou, miou, oiou, acc, macc, oacc=compute_iou(slabels,preds_concat)
        log_str('training miou {:.3} oiou {:.3} macc {:.3} oacc {:.3} '.format(miou, oiou, macc, oacc,),FLAGS.model+'.log')
    else:
        log_str('testing epoch {} loss {:.3} cost {:.3} s'.format(
            epoch_num,loss,time.time()-begin_time),FLAGS.model+'.log')
        iou, miou, oiou, acc, macc, oacc=compute_iou(slabels,preds_concat)
        log_str('testing miou {:.3} oiou {:.3} macc {:.3} oacc {:.3} '.format(miou, oiou, macc, oacc,),FLAGS.model+'.log')

    return miou,preds


def prepare_dataset():
    train_list,test_list=get_block_train_test_split()
    train_list=[fn for fn in train_list if fn.split('_')[-2]=='office']
    test_list=[fn for fn in test_list if fn.split('_')[-2]=='office']

    train_list=['data/S3DIS/office_block/'+fn for fn in train_list[:30]]
    test_list=['data/S3DIS/office_block/'+fn for fn in test_list[:8]]

    trainset=[[] for _ in xrange(5)]
    for fn in train_list:
        data=read_pkl(fn)
        for i in xrange(5):
            trainset[i]+=data[i]

    testset=[[] for _ in xrange(5)]
    test_lens=[]
    for fn in test_list:
        data=read_pkl(fn)
        test_lens.append(len(data[0]))
        for i in xrange(5):
            testset[i]+=data[i]

    return trainset,testset,test_lens

def draw_miou(train_mious,test_mious,name):
    plt.figure(0)
    plt.plot(np.arange(len(train_mious)),train_mious,'-')
    plt.plot(np.arange(len(test_mious)),test_mious,'-')
    plt.savefig('model_compare/{}.png'.format(name))
    plt.close()

def draw_points(dataset,preds,lens,name):
    colors=get_class_colors()
    xyzs, rgbs, covars, labels, block_mins = dataset
    cur=0
    for i,l in enumerate(lens):
        cur_preds=[]
        cur_pts=[]
        for k in xrange(cur,cur+l):
            pts=xyzs[k]+block_mins[k]
            cur_preds.append(preds[k])
            cur_pts.append(pts)

        cur_pts=np.concatenate(cur_pts,axis=0)
        cur_preds=np.concatenate(cur_preds,axis=0)
        output_points('test_result/{}_{}.txt'.format(name,i),cur_pts,colors[cur_preds])
        cur+=l

def train():
    trainset, testset, test_lens = prepare_dataset()
    print len(trainset[0]),len(testset[0])
    sess, pls, ops = build_model()
    sess.run(tf.global_variables_initializer())
    train_mious,test_mious=[],[]
    for i in xrange(FLAGS.epoch_num):
        miou,_=process_one_epoch(i,trainset,sess,ops,pls,True)
        train_mious.append(miou)
        miou,preds=process_one_epoch(i,testset,sess,ops,pls,False)
        test_mious.append(miou)
        if i%20==0:
            draw_miou(train_mious,test_mious,FLAGS.model+'{}'.format(i))
            draw_points(testset,preds,test_lens,FLAGS.model+'{}'.format(i))
            # draw_feats_distribution(testset,sess,pls)

    draw_miou(train_mious,test_mious,FLAGS.model)

def draw_feats_hist(feats,name):
    from analysis import draw_hist
    for i in xrange(feats.shape[1]):
        draw_hist(feats[:,i],name+'{}'.format(i))

def draw_feats_distribution(dataset,sess,pls):
    xyzs, rgbs, covars, labels, block_mins = dataset

    indices=np.arange(len(xyzs))
    feats1,feats2,feats3=[],[],[]
    for k in indices:
        feed_dict={}
        feed_dict[pls['xyzs']]=xyzs[k]
        feed_dict[pls['feats']]=rgbs[k]
        feed_dict[pls['labels']]=labels[k]
        feed_dict[pls['is_training']]=False
        feat1,feat2,feat3=sess.run([feats_ops['feats1'],feats_ops['feats2'],feats_ops['feats3']],feed_dict=feed_dict)
        feats1.append(feat1)
        feats2.append(feat2)
        feats3.append(feat3)

    feats1=np.concatenate(feats1,axis=0)
    feats2=np.concatenate(feats2,axis=0)
    feats3=np.concatenate(feats3,axis=0)

    draw_feats_hist(feats1,'{}_feats1_'.format(FLAGS.model))
    draw_feats_hist(feats2,'{}_feats2_'.format(FLAGS.model))
    draw_feats_hist(feats3,'{}_feats3_'.format(FLAGS.model))


def read_log(fn):
    with open(fn,'r') as f:
        train_mious=[]
        test_mious=[]
        for line in f.readlines():
            if line.startswith('training miou'):
                train_mious.append(float(line.split(' ')[2]))
            elif line.startswith('testing miou'):
                test_mious.append(float(line.split(' ')[2]))

    return train_mious,test_mious

def plot_compare(fns,name):
    plt.figure(0,figsize=(16, 12), dpi=80)
    for fn in fns:
        train_mious, test_mious=read_log(fn+'.log')
        plt.plot(np.arange(len(train_mious)),train_mious,'-',label='train_{}'.format(fn))
        plt.plot(np.arange(len(test_mious)),test_mious,'-',label='test_{}'.format(fn))

    plt.grid()
    plt.legend()
    plt.savefig('model_compare/{}.png'.format(name))
    plt.close()


if __name__=="__main__":
    # train()
    # plot_compare()

    fns=['mlp_anchor_conv_v{}'.format(i) for i in xrange(5,14)]
    plot_compare(fns,'all_mlp')
    for i in xrange(9):
        plot_compare([fns[i],'mlp_anchor_conv_v4'],'compare_{}'.format(i+5))
