from tf_ops.graph_conv_layer import *
from tf_ops.graph_pooling_layer import *
import tensorflow.contrib.framework as framework
from functools import partial
from model import graph_max_pool_stage,graph_unpool_stage,classifier_v3,classifier_v5,graph_avg_pool_stage
from tensorflow.python.client import timeline
import numpy as np
from draw_util import output_points


def preprocess(xyzs,feats,labels):
    xyzs, pxyzs, dxyzs, feats, labels, vlens, vbegs, vcens=\
        points_pooling(xyzs,feats,labels,voxel_size=0.2,block_size=3.0)
    return xyzs, pxyzs, dxyzs, feats, labels, vlens, vbegs, vcens


def graph_conv_pool_block_edge_new(xyzs, feats, stage_idx, layer_idx, ofn, ncens, nidxs, nlens, nbegs, reuse):
    feats = tf.contrib.layers.fully_connected(feats, num_outputs=ofn, scope='{}_{}_fc'.format(stage_idx, layer_idx),
                                              activation_fn=tf.nn.relu, reuse=reuse)
    feats = graph_conv_edge(xyzs, feats, ofn, [ofn/2, ofn/2], ofn, nidxs, nlens, nbegs, ncens,
                            '{}_{}_gc'.format(stage_idx,layer_idx), reuse=reuse)
    return feats


def graph_conv_pool_block_edge_xyz_new(sxyzs, stage_idx, gxyz_dim, ncens, nidxs, nlens, nbegs, reuse):
    xyz_gc=graph_conv_edge_xyz_v2(sxyzs, gxyz_dim, [gxyz_dim/2, gxyz_dim/2], gxyz_dim, nidxs, nlens, nbegs, ncens,
                                  '{}_xyz_gc'.format(stage_idx),reuse=reuse)
    return xyz_gc


def graph_conv_pool_stage_edge_new(stage_idx, xyzs, dxyz, feats, feats_dim, gxyz_dim, gc_dims, gfc_dims, final_dim,
                                   radius, voxel_size, reuse):
    ops=[]
    with tf.name_scope('stage_{}'.format(stage_idx)):
        nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,radius)

        sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
        sxyzs /= radius   # rescale

        xyz_gc=graph_conv_pool_block_edge_xyz_new(sxyzs,stage_idx,gxyz_dim,ncens,nidxs,nlens,nbegs,reuse)
        ops.append(xyz_gc)
        cfeats = tf.concat([xyz_gc, feats], axis=1)

        cdim = feats_dim + gxyz_dim
        conv_fn = partial(graph_conv_pool_block_edge_new, ncens=ncens, nidxs=nidxs, nlens=nlens, nbegs=nbegs, reuse=reuse)

        layer_idx = 1
        for gd in gc_dims:
            conv_feats = conv_fn(sxyzs, cfeats, stage_idx, layer_idx, gd)
            cfeats = tf.concat([cfeats, conv_feats], axis=1)
            ops.append(conv_feats)
            layer_idx += 1
            cdim += gd

        with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
            with tf.name_scope('fc_global{}'.format(stage_idx)):
                dxyz= dxyz / voxel_size

                fc = tf.concat([cfeats, dxyz], axis=1)
                for i, gfd in enumerate(gfc_dims):
                    fc = tf.contrib.layers.fully_connected(fc, num_outputs=gfd,
                                                           scope='{}_{}_gfc'.format(stage_idx, i))
                fc_final = tf.contrib.layers.fully_connected(fc, num_outputs=final_dim, activation_fn=None,
                                                             scope='{}_final_gfc'.format(stage_idx))

    return fc_final, cfeats, ops  # cfeats: [pn,fc_dims+gxyz_dim+feats_dim]


def graph_conv_pool_edge_new(xyzs, dxyzs, pxyzs, feats, vlens, vbegs, vcens, voxel_len, block_size, reuse=False):
    with tf.name_scope('base_graph_conv_edge_net'):
        with tf.variable_scope('base_graph_conv_edge_net',reuse=reuse):
            with tf.name_scope('conv_stage0'):
                fc0, lf0, ops0 = graph_conv_pool_stage_edge_new(0, xyzs, dxyzs, feats, tf.shape(feats)[1], radius=0.1, reuse=reuse,
                                                                gxyz_dim=8, gc_dims=[8,16], gfc_dims=[16,32,64], final_dim=64,
                                                                voxel_size=voxel_len)
                fc0_pool = graph_max_pool_stage(0, fc0, vlens, vbegs)

            with tf.name_scope('conv_stage1'):
                fc1, lf1, ops1 = graph_conv_pool_stage_edge_new(1, pxyzs, pxyzs, fc0_pool, 64, radius=0.5, reuse=reuse, voxel_size=block_size,
                                                                gxyz_dim=8, gc_dims=[32,32,64,64,128], gfc_dims=[128,256,384], final_dim=384)
                fc1_pool = tf.reduce_max(fc1, axis=0)

            with tf.name_scope('unpool_stage1'):
                upfeats1 = tf.tile(tf.expand_dims(fc1_pool, axis=0), [tf.shape(fc1)[0], 1])
                upf1 = tf.concat([upfeats1, fc1, lf1], axis=1)

            with tf.name_scope('unpool_stage0'):
                upfeats0 = graph_unpool_stage(0, upf1, vlens, vbegs, vcens)
                upf0 = tf.concat([upfeats0, fc0, lf0], axis=1)

            lf = tf.concat([fc0, lf0], axis=1)

            ops1=[graph_unpool_stage(1+idx, op, vlens, vbegs, vcens) for idx,op in enumerate(ops1)]
            ops0+=ops1

    return upf0, lf, ops0


def graph_conv_pool_edge_new_v2(xyzs, dxyzs, pxyzs, feats, vlens, vbegs, vcens, voxel_size, block_size, reuse=False):
    with tf.name_scope('base_graph_conv_edge_net'):
        with tf.variable_scope('base_graph_conv_edge_net',reuse=reuse):
            with tf.name_scope('conv_stage0'):
                # 8 64 64*2
                fc0, lf0, ops0 = graph_conv_pool_stage_edge_new(0, xyzs, dxyzs, feats, tf.shape(feats)[1], radius=0.1,
                                                                reuse=reuse, voxel_size=voxel_size,
                                                                gxyz_dim=16, gc_dims=[16,16,16,16,16],
                                                                gfc_dims=[64,64,64], final_dim=64)
                fc0_pool = graph_max_pool_stage(0, fc0, vlens, vbegs)

            with tf.name_scope('conv_stage1'):
                # 16 288 512*2
                fc1, lf1, ops1 = graph_conv_pool_stage_edge_new(1, pxyzs, pxyzs, fc0_pool, 64, radius=0.5,
                                                                reuse=reuse, voxel_size=block_size,
                                                                gxyz_dim=16, gc_dims=[32,32,32,64,64,64],
                                                                gfc_dims=[256,256,256], final_dim=512)
                fc1_pool = tf.reduce_max(fc1, axis=0)

            with tf.name_scope('unpool_stage1'):
                upfeats1 = tf.tile(tf.expand_dims(fc1_pool, axis=0), [tf.shape(fc1)[0], 1])
                upf1 = tf.concat([upfeats1, fc1, lf1], axis=1)

            with tf.name_scope('unpool_stage0'):
                upfeats0 = graph_unpool_stage(0, upf1, vlens, vbegs, vcens)
                upf0 = tf.concat([upfeats0, fc0, lf0], axis=1)

            lf = tf.concat([fc0, lf0], axis=1)

            ops1=[graph_unpool_stage(1+idx, op, vlens, vbegs, vcens) for idx,op in enumerate(ops1)]
            ops0+=ops1
    # 1528 + 132
    return upf0, lf, ops0


def graph_conv_semantic_pool_stage(stage_idx, dxyz, feats, gfc_dims, final_dim, reuse):
    with tf.name_scope('stage_{}'.format(stage_idx)):
        with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
            with tf.name_scope('fc_global{}'.format(stage_idx)):
                fc = tf.concat([feats, dxyz], axis=1)
                for i, gfd in enumerate(gfc_dims):
                    fc = tf.contrib.layers.fully_connected(fc, num_outputs=gfd,scope='{}_gfc{}'.format(stage_idx, i))
                fc_final = tf.contrib.layers.fully_connected(fc, num_outputs=final_dim, activation_fn=None,
                                                             scope='{}_gfc_final'.format(stage_idx))

    return fc_final  # cfeats: [pn,fc_dims+gxyz_dim+feats_dim]


def graph_conv_semantic_pool_v1(xyzs, dxyzs, pxyzs, feats, vlens, vbegs, vcens, voxel_size, block_size, reuse=False):
    with tf.name_scope('refine_graph_conv_edge_net'):
        with tf.variable_scope('refine_graph_conv_edge_net',reuse=reuse):
            feats=tf.contrib.layers.fully_connected(feats, num_outputs=256,scope='semantic_embed',
                                                    activation_fn=tf.nn.relu, reuse=reuse)
            with tf.name_scope('conv_stage0'):
                fc0, lf0 , _ = graph_conv_pool_stage_edge_new(0, xyzs, dxyzs, feats, 256, radius=0.1,
                                                              reuse=reuse, voxel_size=voxel_size,
                                                              gxyz_dim=16, gc_dims=[16,16],
                                                              gfc_dims=[128,128,128], final_dim=256)
                fc0_pool = graph_max_pool_stage(0, fc0, vlens, vbegs)

            with tf.name_scope('conv_stage1'):
                fc1, lf1 , _ = graph_conv_pool_stage_edge_new(1, pxyzs, pxyzs, fc0_pool, 256, radius=1.5,
                                                              reuse=reuse, voxel_size=block_size,
                                                              gxyz_dim=16, gc_dims=[64,64,64,64],
                                                              gfc_dims=[128,128,128], final_dim=256)
                fc1_pool = tf.reduce_max(fc1, axis=0)

            with tf.name_scope('unpool_stage1'):
                upfeats1 = tf.tile(tf.expand_dims(fc1_pool, axis=0), [tf.shape(fc1)[0], 1])
                upf1 = tf.concat([upfeats1, fc1, lf1], axis=1)

            with tf.name_scope('unpool_stage0'):
                upfeats0 = graph_unpool_stage(0, upf1, vlens, vbegs, vcens)
                upf0 = tf.concat([upfeats0, fc0, lf0], axis=1)

            lf=tf.concat([lf0,fc0],axis=1)

    return upf0,lf


def graph_conv_pool_block_edge_simp(xyzs, feats, stage_idx, layer_idx, ofn, ncens, nidxs, nlens, nbegs, reuse, name=''):
    feats = tf.contrib.layers.fully_connected(feats, num_outputs=ofn, scope='{}{}_{}_fc'.format(name,stage_idx, layer_idx),
                                              activation_fn=tf.nn.relu, reuse=reuse)
    feats = graph_conv_edge_simp(xyzs, feats, ofn, [ofn/2, ofn/2], [ofn/2, ofn/2], ofn, nidxs, nlens, nbegs, ncens,
                                 '{}{}_{}_gc'.format(name,stage_idx,layer_idx), reuse=reuse)
    return feats


def graph_conv_pool_block_edge_xyz_simp(sxyzs, stage_idx, gxyz_dim, ncens, nidxs, nlens, nbegs, reuse, name=''):
    xyz_gc=graph_conv_edge_xyz_simp(sxyzs, gxyz_dim, [gxyz_dim/2, gxyz_dim/2], [gxyz_dim/2, gxyz_dim/2], gxyz_dim,
                                    nidxs, nlens, nbegs, ncens, '{}{}_xyz_gc'.format(name,stage_idx), reuse=reuse)
    return xyz_gc


def graph_conv_pool_stage_edge_simp(stage_idx, xyzs, dxyz, feats, feats_dim, gxyz_dim, gc_dims, gfc_dims, final_dim,
                                    radius, voxel_size, reuse, xyz_fn=graph_conv_pool_block_edge_xyz_simp,
                                    feats_fn=graph_conv_pool_block_edge_simp,name=''):
    ops=[]
    with tf.name_scope('stage_{}'.format(stage_idx)):
        nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,radius)

        sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
        sxyzs /= radius   # rescale

        xyz_gc=xyz_fn(sxyzs,stage_idx,gxyz_dim,ncens,nidxs,nlens,nbegs,reuse,name=name)
        ops.append(xyz_gc)
        cfeats = tf.concat([xyz_gc, feats], axis=1)

        cdim = feats_dim + gxyz_dim
        conv_fn = partial(feats_fn, ncens=ncens, nidxs=nidxs, nlens=nlens, nbegs=nbegs, reuse=reuse, name=name)

        layer_idx = 1
        for gd in gc_dims:
            conv_feats = conv_fn(sxyzs, cfeats, stage_idx, layer_idx, gd)
            cfeats = tf.concat([cfeats, conv_feats], axis=1)
            ops.append(conv_feats)
            layer_idx += 1
            cdim += gd

        with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
            with tf.name_scope('fc_global{}'.format(stage_idx)):
                dxyz= dxyz / voxel_size
                fc_feats = tf.concat([cfeats, dxyz], axis=1)
                for i, gfd in enumerate(gfc_dims):
                    fc = tf.contrib.layers.fully_connected(fc_feats, num_outputs=gfd,
                                                           scope='{}{}_{}_gfc'.format(name,stage_idx, i))
                    fc_feats=tf.concat([fc,fc_feats],axis=1)

                fc_final = tf.contrib.layers.fully_connected(fc_feats, num_outputs=final_dim, activation_fn=None,
                                                             scope='{}{}_final_gfc'.format(name,stage_idx))

    return fc_final, cfeats, ops  # cfeats: [pn,fc_dims+gxyz_dim+feats_dim]


def graph_conv_pool_edge_simp(xyzs, dxyzs, pxyzs, feats, vlens, vbegs, vcens, voxel_size, block_size, reuse=False):
    with tf.name_scope('base_graph_conv_edge_net'):
        with tf.variable_scope('base_graph_conv_edge_net',reuse=reuse):
            with tf.name_scope('conv_stage0'):
                # 8 64 64*2
                fc0, lf0, ops0 = graph_conv_pool_stage_edge_simp(0, xyzs, dxyzs, feats, tf.shape(feats)[1], radius=0.1,
                                                                 reuse=reuse, voxel_size=voxel_size,
                                                                 gxyz_dim=16, gc_dims=[16,16,16,16,16,16],
                                                                 gfc_dims=[16,16,16], final_dim=128)
                fc0_pool = graph_max_pool_stage(0, fc0, vlens, vbegs)

            with tf.name_scope('conv_stage1'):
                # 16 288 512*2
                fc1, lf1, ops1 = graph_conv_pool_stage_edge_simp(1, pxyzs, pxyzs, fc0_pool, 128, radius=0.5,
                                                                 reuse=reuse, voxel_size=block_size,
                                                                 gxyz_dim=16, gc_dims=[32,32,32,32,32,32],
                                                                 gfc_dims=[32,32,32], final_dim=512)
                fc1_pool = tf.reduce_max(fc1, axis=0)

            with tf.name_scope('unpool_stage1'):
                upfeats1 = tf.tile(tf.expand_dims(fc1_pool, axis=0), [tf.shape(fc1)[0], 1])
                upf1 = tf.concat([upfeats1, fc1, lf1], axis=1)

            with tf.name_scope('unpool_stage0'):
                upfeats0 = graph_unpool_stage(0, upf1, vlens, vbegs, vcens)
                upf0 = tf.concat([upfeats0, fc0, lf0], axis=1)

            lf = tf.concat([fc0, lf0], axis=1)

            ops1=[graph_unpool_stage(1+idx, op, vlens, vbegs, vcens) for idx,op in enumerate(ops1)]
            ops0+=ops1
    # 1528 + 132
    return upf0, lf, ops0


def graph_conv_pool_edge_simp_2layers(xyzs, dxyzs, feats, vlens, vbegs, vcens, voxel_sizes, block_size,
                                      radius=(0.15,0.3,0.5), reuse=False):
    with tf.name_scope('base_graph_conv_edge_net'):
        with tf.variable_scope('base_graph_conv_edge_net',reuse=reuse):
            with tf.name_scope('conv_stage0'):
                fc0, lf0, ops0 = graph_conv_pool_stage_edge_simp(0, xyzs[0], dxyzs[0], feats, tf.shape(feats)[1], radius=radius[0],
                                                                 reuse=reuse, voxel_size=voxel_sizes[0],
                                                                 gxyz_dim=16, gc_dims=[16,16],
                                                                 gfc_dims=[8,8,8], final_dim=64)
                fc0_pool = graph_max_pool_stage(0, fc0, vlens[0], vbegs[0])             # 64
                lf0_avg = graph_avg_pool_stage(0, lf0, vlens[0], vbegs[0], vcens[0])    # 61
                ifeats_0 = tf.concat([fc0_pool,lf0_avg],axis=1)

            with tf.name_scope('conv_stage1'):
                fc1, lf1, ops1 = graph_conv_pool_stage_edge_simp(1, xyzs[1], xyzs[1], ifeats_0, tf.shape(ifeats_0)[1], radius=radius[1],
                                                                 reuse=reuse, voxel_size=voxel_sizes[1],
                                                                 gxyz_dim=16, gc_dims=[32,32,32,32,32,32,32,32,32],
                                                                 gfc_dims=[32,32,32], final_dim=256)
                fc1_pool = graph_max_pool_stage(1, fc1, vlens[1], vbegs[1])         # 256
                lf1_avg = graph_avg_pool_stage(1, lf1, vlens[1], vbegs[1], vcens[1])# 429
                ifeats_1 = tf.concat([fc1_pool,lf1_avg],axis=1)                     # 685

            with tf.name_scope('conv_stage2'):
                fc2, lf2, ops2 = graph_conv_pool_stage_edge_simp(2, xyzs[2], xyzs[2], ifeats_1, tf.shape(ifeats_1)[1], radius=radius[2],
                                                                 reuse=reuse, voxel_size=block_size,
                                                                 gxyz_dim=16, gc_dims=[32,32,32,32,32,32,32,32,32],
                                                                 gfc_dims=[32,32,32], final_dim=512)
                fc2_pool = tf.reduce_max(fc2, axis=0)
                lf2_avg = tf.reduce_mean(lf2, axis=0)
                ifeats_2 = tf.concat([fc2_pool,lf2_avg],axis=0)

            with tf.name_scope('unpool_stage2'):
                upfeats2 = tf.tile(tf.expand_dims(ifeats_2, axis=0), [tf.shape(fc2)[0], 1])
                upf2 = tf.concat([upfeats2, fc2, lf2], axis=1)

            with tf.name_scope('unpool_stage1'):
                upfeats1 = graph_unpool_stage(1, upf2, vlens[1], vbegs[1], vcens[1])
                upf1 = tf.concat([upfeats1, fc1, lf1], axis=1)

            with tf.name_scope('unpool_stage0'):
                upfeats0 = graph_unpool_stage(0, upf1, vlens[0], vbegs[0], vcens[0])
                upf0 = tf.concat([upfeats0, fc0, lf0], axis=1)

            lf = tf.concat([fc0, lf0], axis=1)

            # ops1=[graph_unpool_stage(1+idx, op, vlens, vbegs, vcens) for idx,op in enumerate(ops1)]
            # ops0+=ops1
            ops=[fc0,lf0,fc1,lf1,fc2,lf2]

    return upf0, lf, ops


def graph_conv_pool_edge_simp_2layers_s3d(xyzs, dxyzs, feats, vlens, vbegs, vcens, voxel_sizes, block_size,
                                          radius=(0.15,0.3,0.5), reuse=False):
    with tf.name_scope('base_graph_conv_edge_net'):
        with tf.variable_scope('base_graph_conv_edge_net',reuse=reuse):
            with tf.name_scope('conv_stage0'):
                fc0, lf0, ops0 = graph_conv_pool_stage_edge_simp(0, xyzs[0], dxyzs[0], feats, tf.shape(feats)[1], radius=radius[0],
                                                                 reuse=reuse, voxel_size=voxel_sizes[0]/2.0,
                                                                 gxyz_dim=16, gc_dims=[16],
                                                                 gfc_dims=[16,16,16], final_dim=64)
                fc0_pool = graph_max_pool_stage(0, fc0, vlens[0], vbegs[0])             # 64
                lf0_avg = graph_avg_pool_stage(0, lf0, vlens[0], vbegs[0], vcens[0])    # 61
                ifeats_0 = tf.concat([fc0_pool,lf0_avg],axis=1)

            with tf.name_scope('conv_stage1'):
                fc1, lf1, ops1 = graph_conv_pool_stage_edge_simp(1, xyzs[1], xyzs[1], ifeats_0, tf.shape(ifeats_0)[1], radius=radius[1],
                                                                 reuse=reuse, voxel_size=voxel_sizes[1]/2.0,
                                                                 gxyz_dim=16, gc_dims=[16,16,32,32],
                                                                 gfc_dims=[32,32,32], final_dim=128)
                fc1_pool = graph_max_pool_stage(1, fc1, vlens[1], vbegs[1])         # 256
                lf1_avg = graph_avg_pool_stage(1, lf1, vlens[1], vbegs[1], vcens[1])# 429
                ifeats_1 = tf.concat([fc1_pool,lf1_avg],axis=1)                     # 685

            with tf.name_scope('conv_stage2'):
                fc2, lf2, ops2 = graph_conv_pool_stage_edge_simp(2, xyzs[2], xyzs[2], ifeats_1, tf.shape(ifeats_1)[1], radius=radius[2],
                                                                 reuse=reuse, voxel_size=block_size/2.0,
                                                                 gxyz_dim=16, gc_dims=[32,32,64,64],
                                                                 gfc_dims=[64,64,64], final_dim=384)
                fc2_pool = tf.reduce_max(fc2, axis=0)
                lf2_avg = tf.reduce_mean(lf2, axis=0)
                ifeats_2 = tf.concat([fc2_pool,lf2_avg],axis=0)

            with tf.name_scope('unpool_stage2'):
                upfeats2 = tf.tile(tf.expand_dims(ifeats_2, axis=0), [tf.shape(fc2)[0], 1])
                upf2 = tf.concat([upfeats2, fc2, lf2], axis=1)

            with tf.name_scope('unpool_stage1'):
                upfeats1 = graph_unpool_stage(1, upf2, vlens[1], vbegs[1], vcens[1])
                upf1 = tf.concat([upfeats1, fc1, lf1], axis=1)

            with tf.name_scope('unpool_stage0'):
                upfeats0 = graph_unpool_stage(0, upf1, vlens[0], vbegs[0], vcens[0])
                upf0 = tf.concat([upfeats0, fc0, lf0], axis=1)

            lf = tf.concat([fc0, lf0], axis=1)

            # ops1=[graph_unpool_stage(1+idx, op, vlens, vbegs, vcens) for idx,op in enumerate(ops1)]
            # ops0+=ops1
            ops=[fc0,lf0,fc1,lf1,fc2,lf2]

    return upf0, lf, ops


def graph_conv_pool_context(pxyzs, feats, block_size, radius, reuse=False):
    with tf.name_scope('base_graph_conv_context_net'):
        with tf.variable_scope('base_graph_conv_context_net',reuse=reuse):
            with tf.name_scope('context_conv_stage0'):
                fc0, lf0, _ = graph_conv_pool_stage_edge_simp(0, pxyzs, pxyzs, feats, tf.shape(feats)[1],
                                                              radius=radius, reuse=reuse, voxel_size=block_size,
                                                              gxyz_dim=16, gc_dims=[16,16,16,32,32,32],
                                                              gfc_dims=[32,32,64], final_dim=256,name='context')
                fc0_pool = tf.reduce_max(fc0, axis=0)
                lf0_avg = tf.reduce_mean(lf0, axis=0)
                ifeats_0 = tf.concat([fc0_pool,lf0_avg],axis=0)

            with tf.name_scope('context_unpool_stage0'):
                upfeats0 = tf.tile(tf.expand_dims(ifeats_0, axis=0), [tf.shape(fc0)[0], 1])
                upf0 = tf.concat([upfeats0, fc0, lf0], axis=1)
                # upf0 = tf.concat([fc0, lf0], axis=1)

    return upf0



def graph_conv_pool_context_with_pool(xyzs, dxyzs, pxyzs, feats, vlens, vbegs, vcens,
                                      voxel_size, block_size, radius1, radius2, reuse=False):
    with tf.name_scope('base_graph_conv_context_net'):
        with tf.variable_scope('base_graph_conv_context_net',reuse=reuse):
            with tf.name_scope('context_conv_stage0'):
                fc0, lf0, _ = graph_conv_pool_stage_edge_simp(0, xyzs, dxyzs, feats, tf.shape(feats)[1],
                                                              radius=radius1, reuse=reuse, voxel_size=voxel_size,
                                                              gxyz_dim=16, gc_dims=[16,16,16],
                                                              gfc_dims=[16,16,16], final_dim=64,name='context')

                fc0_pool = graph_max_pool_stage(0, fc0, vlens, vbegs)             # 64
                lf0_avg = graph_avg_pool_stage(0, lf0, vlens, vbegs, vcens)    # 61
                ifeats_0 = tf.concat([fc0_pool,lf0_avg],axis=1)

            with tf.name_scope('context_conv_stage1'):
                fc1, lf1, _ = graph_conv_pool_stage_edge_simp(1, pxyzs, pxyzs, ifeats_0, tf.shape(ifeats_0)[1],
                                                              radius=radius2, reuse=reuse, voxel_size=block_size,
                                                              gxyz_dim=16, gc_dims=[32,32,32],
                                                              gfc_dims=[32,32,64], final_dim=256,name='context')

                fc1_pool = tf.reduce_max(fc1, axis=0)
                lf1_avg = tf.reduce_mean(lf1, axis=0)
                ifeats_1 = tf.concat([fc1_pool,lf1_avg],axis=0)

            with tf.name_scope('context_unpool_stage1'):
                upfeats1 = tf.tile(tf.expand_dims(ifeats_1, axis=0), [tf.shape(fc1)[0], 1])
                upf1 = tf.concat([upfeats1, fc1, lf1], axis=1)

            with tf.name_scope('context_unpool_stage0'):
                upfeats0 = graph_unpool_stage(0, upf1, vlens, vbegs, vcens)
                upf0 = tf.concat([upfeats0, fc0, lf0], axis=1)

    return upf0


def graph_conv_pool_edge_simp_2layers_no_avg(xyzs, dxyzs, feats, vlens, vbegs, vcens, voxel_sizes, block_size, reuse=False):
    with tf.name_scope('base_graph_conv_edge_net'):
        with tf.variable_scope('base_graph_conv_edge_net',reuse=reuse):
            with tf.name_scope('conv_stage0'):
                fc0, lf0, ops0 = graph_conv_pool_stage_edge_simp(0, xyzs[0], dxyzs[0], feats, tf.shape(feats)[1], radius=0.15,
                                                                 reuse=reuse, voxel_size=voxel_sizes[0],
                                                                 gxyz_dim=16, gc_dims=[16,16],
                                                                 gfc_dims=[8,8,8], final_dim=64)
                fc0_pool = graph_max_pool_stage(0, fc0, vlens[0], vbegs[0])

            with tf.name_scope('conv_stage1'):
                fc1, lf1, ops1 = graph_conv_pool_stage_edge_simp(1, xyzs[1], xyzs[1], fc0_pool, tf.shape(fc0_pool)[1], radius=0.3,
                                                                 reuse=reuse, voxel_size=voxel_sizes[1],
                                                                 gxyz_dim=16, gc_dims=[32,32,32,32,32,32,32,32,32],
                                                                 gfc_dims=[32,32,32], final_dim=256)
                fc1_pool = graph_max_pool_stage(1, fc1, vlens[1], vbegs[1])

            with tf.name_scope('conv_stage2'):
                fc2, lf2, ops2 = graph_conv_pool_stage_edge_simp(2, xyzs[2], xyzs[2], fc1_pool, tf.shape(fc1_pool)[1], radius=0.5,
                                                                 reuse=reuse, voxel_size=block_size,
                                                                 gxyz_dim=16, gc_dims=[32,32,32,32,32,32,32,32,32],
                                                                 gfc_dims=[32,32,32], final_dim=512)
                fc2_pool = tf.reduce_max(fc2, axis=0)
                lf2_avg = tf.reduce_mean(lf2, axis=0)
                ifeats_2 = tf.concat([fc2_pool,lf2_avg],axis=0)

            with tf.name_scope('unpool_stage2'):
                upfeats2 = tf.tile(tf.expand_dims(ifeats_2, axis=0), [tf.shape(fc2)[0], 1])
                upf2 = tf.concat([upfeats2, fc2, lf2], axis=1)

            with tf.name_scope('unpool_stage1'):
                upfeats1 = graph_unpool_stage(1, upf2, vlens[1], vbegs[1], vcens[1])
                upf1 = tf.concat([upfeats1, fc1, lf1], axis=1)

            with tf.name_scope('unpool_stage0'):
                upfeats0 = graph_unpool_stage(0, upf1, vlens[0], vbegs[0], vcens[0])
                upf0 = tf.concat([upfeats0, fc0, lf0], axis=1)

            lf = tf.concat([fc0, lf0], axis=1)

            # ops1=[graph_unpool_stage(1+idx, op, vlens, vbegs, vcens) for idx,op in enumerate(ops1)]
            # ops0+=ops1

    # 1528 + 132
    return upf0, lf, ops0


def graph_conv_pool_block_edge_simp_v2(xyzs, feats, stage_idx, layer_idx, ofn, ncens, nidxs, nlens, nbegs, reuse):
    feats = tf.contrib.layers.fully_connected(feats, num_outputs=ofn*2, scope='{}_{}_fc'.format(stage_idx, layer_idx),
                                              activation_fn=tf.nn.relu, reuse=reuse)
    feats = graph_conv_edge_simp(xyzs, feats, ofn*2, [ofn/2, ofn/2], [ofn, ofn], ofn, nidxs, nlens, nbegs, ncens,
                                 '{}_{}_gc'.format(stage_idx,layer_idx), reuse=reuse)
    return feats


def graph_conv_pool_block_edge_xyz_simp_v2(sxyzs, stage_idx, gxyz_dim, ncens, nidxs, nlens, nbegs, reuse):
    xyz_gc=graph_conv_edge_xyz_simp(sxyzs, gxyz_dim*2, [gxyz_dim, gxyz_dim], [gxyz_dim, gxyz_dim], gxyz_dim,
                                    nidxs, nlens, nbegs, ncens, '{}_xyz_gc'.format(stage_idx), reuse=reuse)
    return xyz_gc


def graph_conv_pool_stage_edge_simp_v2(stage_idx, xyzs, dxyz, feats, feats_dim, gxyz_dim, gc_dims, gfc_dims, final_dim,
                                       radius, voxel_size, reuse):
    ops=[]
    with tf.name_scope('stage_{}'.format(stage_idx)):
        nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,radius)

        sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
        sxyzs /= radius   # rescale

        xyz_gc=graph_conv_pool_block_edge_xyz_simp_v2(sxyzs,stage_idx,gxyz_dim,ncens,nidxs,nlens,nbegs,reuse)
        ops.append(xyz_gc)
        cfeats = tf.concat([xyz_gc, feats], axis=1)

        cdim = feats_dim + gxyz_dim
        conv_fn = partial(graph_conv_pool_block_edge_simp_v2, ncens=ncens, nidxs=nidxs, nlens=nlens, nbegs=nbegs, reuse=reuse)

        layer_idx = 1
        for gd in gc_dims:
            conv_feats = conv_fn(sxyzs, cfeats, stage_idx, layer_idx, gd)
            cfeats = tf.concat([cfeats, conv_feats], axis=1)
            ops.append(conv_feats)
            layer_idx += 1
            cdim += gd

        with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
            with tf.name_scope('fc_global{}'.format(stage_idx)):
                dxyz= dxyz / voxel_size
                fc_feats = tf.concat([cfeats, dxyz], axis=1)
                for i, gfd in enumerate(gfc_dims):
                    fc = tf.contrib.layers.fully_connected(fc_feats, num_outputs=gfd,
                                                           scope='{}_{}_gfc'.format(stage_idx, i))
                    fc_feats=tf.concat([fc,fc_feats],axis=1)

                fc_final = tf.contrib.layers.fully_connected(fc_feats, num_outputs=final_dim, activation_fn=None,
                                                             scope='{}_final_gfc'.format(stage_idx))

    return fc_final, cfeats, ops  # cfeats: [pn,fc_dims+gxyz_dim+feats_dim]


def graph_conv_pool_edge_simp_v2(xyzs, dxyzs, pxyzs, feats, vlens, vbegs, vcens, voxel_size, block_size, reuse=False):
    with tf.name_scope('base_graph_conv_edge_net'):
        with tf.variable_scope('base_graph_conv_edge_net',reuse=reuse):
            with tf.name_scope('conv_stage0'):
                # 8 64 64*2
                fc0, lf0, ops0 = graph_conv_pool_stage_edge_simp_v2(0, xyzs, dxyzs, feats, tf.shape(feats)[1], radius=0.1,
                                                                    reuse=reuse, voxel_size=voxel_size,
                                                                    gxyz_dim=16, gc_dims=[16,16,16],
                                                                    gfc_dims=[16,16,16], final_dim=128)
                feats0=tf.concat([fc0,lf0],axis=1)
                fc2, lf2, ops2 = graph_conv_pool_stage_edge_simp(2, xyzs, dxyzs, feats0, tf.shape(feats0)[1], radius=0.2,
                                                                 reuse=reuse, voxel_size=voxel_size,
                                                                 gxyz_dim=16, gc_dims=[16,16,16],
                                                                 gfc_dims=[16,16,16], final_dim=256)

                fc2_pool = graph_max_pool_stage(0, fc2, vlens, vbegs)

            with tf.name_scope('conv_stage1'):
                # 16 288 512*2
                fc1, lf1, ops1 = graph_conv_pool_stage_edge_simp_v2(1, pxyzs, pxyzs, fc2_pool, 128, radius=0.5,
                                                                 reuse=reuse, voxel_size=block_size,
                                                                 gxyz_dim=16, gc_dims=[32,32,32,64,64,64],
                                                                 gfc_dims=[32,32,32], final_dim=512)
                fc1_pool = tf.reduce_max(fc1, axis=0)

            with tf.name_scope('unpool_stage1'):
                upfeats1 = tf.tile(tf.expand_dims(fc1_pool, axis=0), [tf.shape(fc1)[0], 1])
                upf1 = tf.concat([upfeats1, fc1, lf1], axis=1)

            with tf.name_scope('unpool_stage0'):
                upfeats0 = graph_unpool_stage(0, upf1, vlens, vbegs, vcens)
                upf0 = tf.concat([upfeats0, fc2, lf2], axis=1)

            lf = tf.concat([fc2, lf2], axis=1)

            ops1=[graph_unpool_stage(1+idx, op, vlens, vbegs, vcens) for idx,op in enumerate(ops1)]
            ops0+=ops1
    # 1528 + 132
    return upf0, lf, ops0


def test_model():
    num_classes = 13
    from io_util import read_pkl,get_block_train_test_split
    import numpy as np
    import random
    import time
    train_list,test_list=get_block_train_test_split()
    random.shuffle(train_list)
    cxyzs, dxyzs, rgbs, covars, lbls, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens, block_mins = \
        read_pkl('data/S3DIS/sampled_train/{}'.format(train_list[0]))

    xyzs_pl = tf.placeholder(tf.float32, [None, 3], 'xyzs')
    feats_pl = tf.placeholder(tf.float32, [None, 12], 'feats')
    labels_pl = tf.placeholder(tf.int32, [None], 'labels')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.Session(config=config) as sess:
        xyzs, pxyzs, dxyzs, feats, labels, vlens, vbegs, vcens = points_pooling(xyzs_pl,feats_pl,labels_pl,0.3,3.0)
        global_feats,local_feats,_=graph_conv_pool_edge_new(xyzs,dxyzs,pxyzs,feats,vlens,vbegs,vcens,False)
        global_feats = tf.expand_dims(global_feats, axis=0)
        local_feats = tf.expand_dims(local_feats, axis=0)
        logits = classifier_v3(global_feats, local_feats, tf.Variable(False,trainable=False,dtype=tf.bool),
                               num_classes, False, use_bn=False)

        labels = tf.cast(labels, tf.int64)
        flatten_logits = tf.reshape(logits, [-1, num_classes])  # [pn,num_classes]
        acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(flatten_logits,axis=1),labels),tf.float32),axis=0)

        loss=tf.losses.sparse_softmax_cross_entropy(labels,flatten_logits)
        opt=tf.train.GradientDescentOptimizer(1e-2)
        train_op=opt.minimize(loss)

        sess.run(tf.global_variables_initializer())

        for k in xrange(20):
            bg=time.time()
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            _,acc_val=sess.run([train_op,acc],feed_dict={
                xyzs_pl:cxyzs[0][0],
                feats_pl:np.concatenate([rgbs[0],covars[0]],axis=1),
                labels_pl:lbls[0]
            },options=options, run_metadata=run_metadata)
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open('timeline.json', 'w') as f:
                f.write(chrome_trace)
            print 'cost {} {} s'.format(time.time()-bg,1/(time.time()-bg))
            print acc_val


def test_block():

    from io_util import read_pkl,get_block_train_test_split
    import numpy as np
    from draw_util import output_points,get_class_colors
    import random
    train_list,test_list=get_block_train_test_split()
    random.shuffle(train_list)

    xyzs_pl = tf.placeholder(tf.float32, [None, 3], 'xyzs')
    feats_pl = tf.placeholder(tf.float32, [None, 3], 'feats')
    labels_pl = tf.placeholder(tf.int32, [None], 'labels')
    [pts1, pts2, pts3], [dpts1, dpts2], feats, _, [vlens1, vlens2], [vbegs1, vbegs2], [vcens1, vcens2], vidx1, vidx2 = \
        points_pooling_two_layers_tmp(xyzs_pl, feats_pl, labels_pl, 0.15, 0.45, 3.0)

    nidxs1, nlens1, nbegs1, ncens1 = search_neighborhood(pts3, 0.9)
    nidxs2, nlens2, nbegs2, ncens2 = search_neighborhood_range(pts2, 0.3, 0.45)
    nidxs3, nlens3, nbegs3, ncens3 = search_neighborhood_range(pts2, 0.45, 0.6)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.Session(config=config) as sess:
        pn1,pn2,pn3,nn1,nn2,nn3,vn1,vn2=[],[],[],[],[],[],[],[]
        for i in xrange(len(train_list[:5])):
            xyzs,rgbs,covars,lbls,block_mins= read_pkl('data/S3DIS/sampled_train_nolimits/{}'.format(train_list[0]))
            for t in xrange(len(xyzs[:5])):
                p1,p2,p3,nl1,nl2,nl3,vl1,vl2,dp1,dp2,vc1,vc2=\
                    sess.run([pts1, pts2, pts3, nlens1, nlens2, nlens3, vlens1, vlens2, dpts1, dpts2,vcens1,vcens2],
                             feed_dict={xyzs_pl:xyzs[t],feats_pl:rgbs[t],labels_pl:lbls[t]},)
                # output_points('test_result/{}_1.txt'.format(t),p1)
                # output_points('test_result/{}_2.txt'.format(t),p2)
                # output_points('test_result/{}_3.txt'.format(t),p3)
                # check_dxyzs(p1,p2,dp1,vc1)
                # check_dxyzs(p2,p3,dp2,vc2)

                print 'lvl {} pn {} nr {} vl {}'.format(1,p1.shape[0],np.mean(nl1),np.mean(vl1))
                print 'lvl {} pn {} nr {} vl {}'.format(2,p2.shape[0],np.mean(nl2),np.mean(vl2))
                print 'lvl {} pn {} nr {}'.format(3,p3.shape[0],np.mean(nl3))
                print '/////////////////////////////////////////////////////'

                pn1.append(p1.shape[0])
                pn2.append(p2.shape[0])
                pn3.append(p3.shape[0])
                nn1.append(np.mean(nl1))
                nn2.append(np.mean(nl2))
                nn3.append(np.mean(nl3))
                vn1.append(np.mean(vl1))
                vn2.append(np.mean(vl2))

        pn1=np.mean(pn1)
        pn2=np.mean(pn2)
        pn3=np.mean(pn3)
        nn1=np.mean(nn1)
        nn2=np.mean(nn2)
        nn3=np.mean(nn3)
        vn1=np.mean(vn1)
        vn2=np.mean(vn2)

        print pn1,pn2,pn3
        print nn1,nn2,nn3
        print vn1,vn2



def test_three_pooling_block():

    from io_util import read_pkl,get_block_train_test_split
    import numpy as np
    from draw_util import output_points,get_class_colors
    import random
    train_list,test_list=get_block_train_test_split()
    random.shuffle(train_list)

    xyzs_pl = tf.placeholder(tf.float32, [None, 3], 'xyzs')
    feats_pl = tf.placeholder(tf.float32, [None, 3], 'feats')
    labels_pl = tf.placeholder(tf.int32, [None], 'labels')
    [pts1, pts2, pts3, pts4], [dpts1, dpts2, dpts3], feats, _, [vlens1, vlens2, vlens3], \
            [vbegs1, vbegs2, vbegs3], [vcens1, vcens2, vcens3] = \
        points_pooling_three_layers(xyzs_pl, feats_pl, labels_pl, 0.1, 0.2, 0.4, 3.0)

    nidxs1, nlens1, nbegs1, ncens1 = search_neighborhood(pts1, 0.1)
    nidxs2, nlens2, nbegs2, ncens2 = search_neighborhood(pts2, 0.2)
    nidxs3, nlens3, nbegs3, ncens3 = search_neighborhood(pts3, 0.4)
    nidxs4, nlens4, nbegs4, ncens4 = search_neighborhood(pts4, 0.8)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.Session(config=config) as sess:
        pn1,pn2,pn3,pn4,nn1,nn2,nn3,nn4,vn1,vn2,vn3=[],[],[],[],[],[],[],[],[],[],[]
        for i in xrange(len(train_list[:1])):
            xyzs, rgbs, covars, lbls, block_mins = read_pkl('data/S3DIS/sampled_train_new/{}'.format(train_list[0]))
            # for t in xrange(len(xyzs[:5])):
            #     p1,p2,p3,p4,vc1,vc2,vc3=\
            #     sess.run([pts1, pts2, pts3, pts4, vcens1, vcens2, vcens3],
            #              feed_dict={xyzs_pl: xyzs[t], feats_pl: rgbs[t], labels_pl: lbls[t]})
            #
            #     output_hierarchy(p1,p2,np.asarray(vc1,np.int32),'{}_12'.format(t))
            #     output_hierarchy(p2,p3,np.asarray(vc2,np.int32),'{}_23'.format(t))
            #     output_hierarchy(p3,p4,np.asarray(vc3,np.int32),'{}_34'.format(t))
            #
            #     output_hierarchy(p2,p4,vc3[vc2],'{}_24'.format(t))
            #     output_hierarchy(p1,p4,vc3[vc2[vc1]],'{}_14'.format(t))
            for t in xrange(len(xyzs)):
                p1,p2,p3,p4,nl1,nl2,nl3,nl4,vl1,vl2,vl3=\
                    sess.run([pts1, pts2, pts3, pts4, nlens1, nlens2, nlens3, nlens4, vlens1, vlens2, vlens3],
                             feed_dict={xyzs_pl:xyzs[t],feats_pl:rgbs[t],labels_pl:lbls[t]},)

                print 'lvl {} pn {} nr {} vl {}'.format(1,p1.shape[0],np.mean(nl1),np.mean(vl1))
                print 'lvl {} pn {} nr {} vl {}'.format(2,p2.shape[0],np.mean(nl2),np.mean(vl2))
                print 'lvl {} pn {} nr {} vl {}'.format(3,p3.shape[0],np.mean(nl3),np.mean(vl3))
                print 'lvl {} pn {} nr {}'.format(4,p4.shape[0],np.mean(nl4))
                print '/////////////////////////////////////////////////////'
                pn1.append(p1.shape[0])
                pn2.append(p2.shape[0])
                pn3.append(p3.shape[0])
                pn4.append(p4.shape[0])
                nn1.append(np.mean(nl1))
                nn2.append(np.mean(nl2))
                nn3.append(np.mean(nl3))
                nn4.append(np.mean(nl4))
                vn1.append(np.mean(vl1))
                vn2.append(np.mean(vl2))
                vn3.append(np.mean(vl3))

        pn1=np.mean(pn1)
        pn2=np.mean(pn2)
        pn3=np.mean(pn3)
        pn4=np.mean(pn4)
        nn1=np.mean(nn1)
        nn2=np.mean(nn2)
        nn3=np.mean(nn3)
        nn4=np.mean(nn4)
        vn1=np.mean(vn1)
        vn2=np.mean(vn2)
        vn3=np.mean(vn3)

        print pn1,pn2,pn3,pn4
        print nn1,nn2,nn3,nn4
        print vn1,vn2,vn3






def check_vidxs(max_cens,max_len,lens,begs,cens):
    nbegs=np.cumsum(lens)
    assert np.sum(nbegs[:-1]!=begs[1:])==0
    assert begs[0]==0

    assert np.sum(cens>=max_cens)==0
    assert max_len==lens[-1]+begs[-1]


def output_hierarchy(pts1,pts2,cens,name):
    colors=np.random.randint(0,256,[len(pts2),3])
    output_points('test_result/{}_dense.txt'.format(name),pts1,colors[cens,:])
    output_points('test_result/{}_sparse.txt'.format(name),pts2,colors)


def check_dxyzs(pts1,pts2,dpts1,vcens):
    pn1=pts1.shape[0]
    tmp_dpts1=np.copy(dpts1)
    for i in xrange(pn1):
        tmp_dpts1[i]+=pts2[vcens[i]]

    print np.mean(np.abs(tmp_dpts1-pts1),axis=0),np.max(np.abs(tmp_dpts1-pts1),axis=0)

def check_nn(pn,idxs,lens,begs,cens):
    assert begs[-1]+lens[-1]==len(idxs)
    assert np.sum(idxs>=pn)==0
# nr1 = 0.125
# nr2 = 0.5
# nr3 = 2.0
# vc1 = 0.25
# vc2 = 1.0
def test_semantic3d_block():
    from io_util import read_pkl,semantic3d_read_train_block_list
    import numpy as np
    import random
    train_list=semantic3d_read_train_block_list()
    train_list=['data/Semantic3D.Net/block/sampled/'+fn for fn in train_list]
    random.shuffle(train_list)

    xyzs_pl = tf.placeholder(tf.float32, [None, 3], 'xyzs')
    feats_pl = tf.placeholder(tf.float32, [None, 4], 'feats')
    labels_pl = tf.placeholder(tf.int32, [None], 'labels')
    [pts1, pts2, pts3], [dpts1, dpts2], feats, _, [vlens1, vlens2], [vbegs1, vbegs2], [vcens1, vcens2], vidx1, vidx2 = \
        points_pooling_two_layers_tmp(xyzs_pl, feats_pl, labels_pl, 0.25, 0.75, 10.0)

    nidxs1, nlens1, nbegs1, ncens1 = search_neighborhood(pts1, 0.2)
    nidxs2, nlens2, nbegs2, ncens2 = search_neighborhood(pts2, 0.4)
    nidxs3, nlens3, nbegs3, ncens3 = search_neighborhood(pts3, 1.5)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.Session(config=config) as sess:
        pn1,pn2,pn3,nn1,nn2,nn3,vn1,vn2=[],[],[],[],[],[],[],[]
        for i in xrange(len(train_list[:15])):
            xyzs,rgbs,covars,lbls,block_mins= read_pkl(train_list[i])
            for t in xrange(len(xyzs[:10])):
                pt_num = len(xyzs[t])
                if pt_num>20480:
                    idxs = np.random.choice(pt_num, 20480, False)
                    xyzs[t] = xyzs[t][idxs]
                    rgbs[t] = rgbs[t][idxs]
                    covars[t] = covars[t][idxs]
                    lbls[t] = lbls[t][idxs]

                p1,p2,p3,nl1,nl2,nl3,vl1,vl2,dp1,dp2,vc1,vc2=\
                    sess.run([pts1, pts2, pts3, nlens1, nlens2, nlens3, vlens1, vlens2, dpts1, dpts2,vcens1,vcens2],
                             feed_dict={xyzs_pl:xyzs[t],feats_pl:rgbs[t],labels_pl:lbls[t]},)
                # output_points('test_result/{}_1.txt'.format(t),p1)
                # output_points('test_result/{}_2.txt'.format(t),p2)
                # output_points('test_result/{}_3.txt'.format(t),p3)
                # check_dxyzs(p1,p2,dp1,vc1)
                # check_dxyzs(p2,p3,dp2,vc2)

                print 'lvl {} pn {} nr {} vl {}'.format(1,p1.shape[0],np.mean(nl1),np.mean(vl1))
                print 'lvl {} pn {} nr {} vl {}'.format(2,p2.shape[0],np.mean(nl2),np.mean(vl2))
                print 'lvl {} pn {} nr {}'.format(3,p3.shape[0],np.mean(nl3))
                print '/////////////////////////////////////////////////////'

                pn1.append(p1.shape[0])
                pn2.append(p2.shape[0])
                pn3.append(p3.shape[0])
                nn1.append(np.mean(nl1))
                nn2.append(np.mean(nl2))
                nn3.append(np.mean(nl3))
                vn1.append(np.mean(vl1))
                vn2.append(np.mean(vl2))

        pn1=np.mean(pn1)
        pn2=np.mean(pn2)
        pn3=np.mean(pn3)
        nn1=np.mean(nn1)
        nn2=np.mean(nn2)
        nn3=np.mean(nn3)
        vn1=np.mean(vn1)
        vn2=np.mean(vn2)

        print pn1,pn2,pn3
        print nn1,nn2,nn3
        print vn1,vn2

def test_semantic3d_block_dense():
    from io_util import read_pkl,semantic3d_read_train_block_list
    import numpy as np
    import random
    import libPointUtil
    import time
    train_list=semantic3d_read_train_block_list()
    train_list=['/data/Semantic3D.Net/'+fn for fn in train_list]
    random.shuffle(train_list)

    xyzs_pl = tf.placeholder(tf.float32, [None, 3], 'xyzs')
    feats_pl = tf.placeholder(tf.float32, [None, 4], 'feats')
    labels_pl = tf.placeholder(tf.int32, [None], 'labels')
    idxs_pl = tf.placeholder(tf.int32, [None], 'idxs')

    xyzs_pl=tf.gather(xyzs_pl,idxs_pl)
    feats_pl=tf.gather(feats_pl,idxs_pl)
    labels_pl=tf.gather(labels_pl,idxs_pl)

    [pts1, pts2, pts3], [dpts1, dpts2], feats, _, [vlens1, vlens2], [vbegs1, vbegs2], [vcens1, vcens2], vidx1, vidx2 = \
        points_pooling_two_layers_tmp(xyzs_pl, feats_pl, labels_pl, 0.45, 1.5, 10.0)

    nidxs1, nlens1, nbegs1, ncens1 = search_neighborhood(pts2, 0.9)
    nidxs2, nlens2, nbegs2, ncens2 = search_neighborhood_range(pts2, 0.9, 1.25)
    nidxs3, nlens3, nbegs3, ncens3 = search_neighborhood_range(pts2, 1.25, 1.6)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.Session(config=config) as sess:
        pn1,pn2,pn3,nn1,nn2,nn3,vn1,vn2=[],[],[],[],[],[],[],[]
        for i in xrange(len(train_list[:15])):
            xyzs, rgbs, lbls, block_mins, _, _, _= read_pkl(train_list[i])
            for t in xrange(len(xyzs[:10])):
                begin=time.time()
                idxs = libPointUtil.gridDownsampleGPU(xyzs[t], 0.15, False)
                np.random.shuffle(idxs)
                idxs = idxs[:20480]
                print 'cost {} s'.format(time.time()-begin)

                p1,p2,p3,nl1,nl2,nl3,vl1,vl2,dp1,dp2,vc1,vc2=\
                    sess.run([pts1, pts2, pts3, nlens1, nlens2, nlens3, vlens1, vlens2, dpts1, dpts2,vcens1,vcens2],
                             feed_dict={xyzs_pl:xyzs[t],feats_pl:rgbs[t],labels_pl:lbls[t],idxs_pl:idxs},
                             )
                # output_points('test_result/{}_1.txt'.format(t),p1)
                # output_points('test_result/{}_2.txt'.format(t),p2)
                # output_points('test_result/{}_3.txt'.format(t),p3)
                # check_dxyzs(p1,p2,dp1,vc1)
                # check_dxyzs(p2,p3,dp2,vc2)

                # print 'lvl {} pn {} nr {} vl {}'.format(1,p1.shape[0],np.mean(nl1),np.mean(vl1))
                # print 'lvl {} pn {} nr {} vl {}'.format(2,p2.shape[0],np.mean(nl2),np.mean(vl2))
                # print 'lvl {} pn {} nr {}'.format(3,p3.shape[0],np.mean(nl3))
                # print '/////////////////////////////////////////////////////'

                pn1.append(p1.shape[0])
                pn2.append(p2.shape[0])
                pn3.append(p3.shape[0])
                nn1.append(np.mean(nl1))
                nn2.append(np.mean(nl2))
                nn3.append(np.mean(nl3))
                vn1.append(np.mean(vl1))
                vn2.append(np.mean(vl2))

        pn1=np.mean(pn1)
        pn2=np.mean(pn2)
        pn3=np.mean(pn3)
        nn1=np.mean(nn1)
        nn2=np.mean(nn2)
        nn3=np.mean(nn3)
        vn1=np.mean(vn1)
        vn2=np.mean(vn2)

        print pn1,pn2,pn3
        print nn1,nn2,nn3
        print vn1,vn2

def test_context_neighborhood():
    from semantic3d_context_util import get_context_train_test
    from io_util import read_pkl
    train_list,test_list=get_context_train_test()

    pts_pl=tf.placeholder(tf.float32,[None,3],'pts')
    nidxs, nlens, nbegs, ncens=search_neighborhood(pts_pl,4.0)
    with tf.Session() as sess:
        for fs in train_list[:3]:
            xyzs, rgbs, covars, lbls, ctx_xyzs, ctx_idxs, block_mins = read_pkl('data/Semantic3D.Net/context/block/'+fs)
            for i in xrange(len(xyzs)):
                lens=sess.run(nlens,feed_dict={pts_pl:ctx_xyzs[i][:,:3]})
                print np.mean(lens)


def test_semantic3d_block_context():
    from semantic3d_context_util import read_large_block_list
    from io_util import read_pkl
    import random
    train_list=read_large_block_list()
    train_list=['data/Semantic3D.Net/context/block_avg/'+fn for fn in train_list]
    random.shuffle(train_list)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess=tf.Session(config=config)
    for k in xrange(1):
        xyzs, rgbs, covars, lbls, ctx_xyzs_val, ctx_idxs_val, block_mins=read_pkl(train_list[k])

        xyzs_pl = tf.placeholder(tf.float32, [None, 3], 'xyzs')
        feats_pl = tf.placeholder(tf.float32, [None, 4], 'feats')
        labels_pl = tf.placeholder(tf.int32, [None], 'labels')
        ctx_idxs_pl = tf.placeholder(tf.int32, [None], 'ctx_idxs')
        ctx_pts_pl = tf.placeholder(tf.float32, [None, 16], 'ctx_pts')

        [pts1, pts2, pts3], [dpts1, dpts2], feats, labels, [vlens1, vlens2], [vbegs1, vbegs2], [vcens1, vcens2], ctx_idxs = \
            context_points_pooling_two_layers(xyzs_pl, feats_pl, labels_pl, ctx_idxs_pl, 0.25, 1.0, 10.0)
        ctx_xyzs,ctx_feats=tf.split(ctx_pts_pl,[3,13],axis=1)
        ctx_xyzs,ctx_pxyzs,ctx_dxyzs,ctx_feats,ctx_vlens,ctx_vbegs,ctx_vcens,ctx_idxs=\
            context_points_pooling(ctx_xyzs,ctx_feats,ctx_idxs,5,300)

        nidxs1, nlens1, nbegs1, ncens1 = search_neighborhood(ctx_xyzs, 4.0)
        nidxs2, nlens2, nbegs2, ncens2 = search_neighborhood(ctx_pxyzs, 15.0)

        for t in xrange(len(xyzs[:10])):
            p1,cp,ci,nl1,nl2,cpp=sess.run([pts1,ctx_xyzs,ctx_idxs,nlens1,nlens2,ctx_pxyzs],
                      feed_dict={xyzs_pl:xyzs[t],feats_pl:rgbs[t],labels_pl:lbls[t],
                                 ctx_idxs_pl:ctx_idxs_val[t],ctx_pts_pl:ctx_xyzs_val[t]})
            colors=np.random.randint(0,256,[len(cp),3])

            output_points('test_result/{}xyzs.txt'.format(t),p1,colors[ci,:])
            output_points('test_result/{}ctxs.txt'.format(t),cp,colors)
            print 'lvl 0 {} lvl 1 {} pool {}'.format(np.mean(nl1),np.mean(nl2),len(cpp))


        print '{} done'.format(train_list[k])

if __name__=="__main__":
    test_block()
