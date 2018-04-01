from tf_ops.graph_conv_layer import *
from tf_ops.graph_pooling_layer import *
import tensorflow.contrib.framework as framework
from functools import partial
from model import graph_pool_stage,graph_unpool_stage,classifier_v3
from tensorflow.python.client import timeline


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
    # sxyzs = tf.contrib.layers.fully_connected(sxyzs, num_outputs=gxyz_dim, scope='{}_xyz_fc'.format(stage_idx),
    #                                           activation_fn=tf.nn.relu, reuse=reuse)
    xyz_gc=graph_conv_edge_xyz(sxyzs, 3, [gxyz_dim/2, gxyz_dim/2], gxyz_dim, nidxs, nlens, nbegs, ncens,
                               '{}_xyz_gc'.format(stage_idx),reuse=reuse)
    return xyz_gc


def graph_conv_pool_stage_edge_new(stage_idx, xyzs, dxyz, feats, feats_dim, gxyz_dim, gc_dims, gfc_dims, final_dim, radius, reuse):
    with tf.name_scope('stage_{}'.format(stage_idx)):
        nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,radius)

        sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
        xyz_gc=graph_conv_pool_block_edge_xyz_new(sxyzs,stage_idx,gxyz_dim,ncens,nidxs,nlens,nbegs,reuse)
        cfeats = tf.concat([xyz_gc, feats], axis=1)

        cdim = feats_dim + gxyz_dim
        conv_fn = partial(graph_conv_pool_block_edge_new, ncens=ncens, nidxs=nidxs, nlens=nlens, nbegs=nbegs, reuse=reuse)

        layer_idx = 1
        for gd in gc_dims:
            conv_feats = conv_fn(sxyzs, cfeats, stage_idx, layer_idx, gd)
            cfeats = tf.concat([cfeats, conv_feats], axis=1)
            layer_idx += 1
            cdim += gd

        with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
            with tf.name_scope('fc_global{}'.format(stage_idx)):
                fc = tf.concat([cfeats, dxyz], axis=1)
                for i, gfd in enumerate(gfc_dims):
                    fc = tf.contrib.layers.fully_connected(fc, num_outputs=gfd,
                                                           scope='{}_gfc{}'.format(stage_idx, i))
                fc_final = tf.contrib.layers.fully_connected(fc, num_outputs=final_dim, activation_fn=None,
                                                             scope='{}_gfc_final'.format(stage_idx))

    return fc_final, cfeats  # cfeats: [pn,fc_dims+gxyz_dim+feats_dim]


def graph_conv_pool_edge_new(xyzs, dxyzs, pxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with tf.name_scope('base_graph_conv_edge_net'):
        with tf.variable_scope('base_graph_conv_edge_net',reuse=reuse):
            with tf.name_scope('conv_stage0'):
                fc0, lf0 = graph_conv_pool_stage_edge_new(0,xyzs,dxyzs,feats,tf.shape(feats)[1],radius=0.1,reuse=reuse,
                                                          gxyz_dim=8,gc_dims=[8,16],gfc_dims=[16,32,64],final_dim=64)
                fc0_pool = graph_pool_stage(0, fc0, vlens, vbegs)

            with tf.name_scope('conv_stage1'):
                fc1, lf1 = graph_conv_pool_stage_edge_new(1,pxyzs,pxyzs,fc0_pool,64,radius=0.5,reuse=reuse,
                                                          gxyz_dim=8,gc_dims=[32,32,64,64,128],gfc_dims=[128,256,384],final_dim=384)
                fc1_pool = tf.reduce_max(fc1, axis=0)

            with tf.name_scope('unpool_stage1'):
                upfeats1 = tf.tile(tf.expand_dims(fc1_pool, axis=0), [tf.shape(fc1)[0], 1])
                upf1 = tf.concat([upfeats1, fc1, lf1], axis=1)

            with tf.name_scope('unpool_stage0'):
                upfeats0 = graph_unpool_stage(0, upf1, vlens, vbegs, vcens)
                upf0 = tf.concat([upfeats0, fc0, lf0], axis=1)

            lf = tf.concat([fc0, lf0], axis=1)

    return upf0, lf

def graph_conv_pool_edge_new_v2(xyzs, dxyzs, pxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with tf.name_scope('base_graph_conv_edge_net'):
        with tf.variable_scope('base_graph_conv_edge_net',reuse=reuse):
            with tf.name_scope('conv_stage0'):
                fc0, lf0 = graph_conv_pool_stage_edge_new(0,xyzs,dxyzs,feats,tf.shape(feats)[1],radius=0.1,reuse=reuse,
                                                          gxyz_dim=8,gc_dims=[16,16,16,16],gfc_dims=[64,64,64],final_dim=64)
                fc0_pool = graph_pool_stage(0, fc0, vlens, vbegs)

            with tf.name_scope('conv_stage1'):
                fc1, lf1 = graph_conv_pool_stage_edge_new(1,pxyzs,pxyzs,fc0_pool,64,radius=0.5,reuse=reuse,
                                                          gxyz_dim=16,gc_dims=[32,32,32,64,64,64],gfc_dims=[256,256,256],final_dim=512)
                fc1_pool = tf.reduce_max(fc1, axis=0)

            with tf.name_scope('unpool_stage1'):
                upfeats1 = tf.tile(tf.expand_dims(fc1_pool, axis=0), [tf.shape(fc1)[0], 1])
                upf1 = tf.concat([upfeats1, fc1, lf1], axis=1)

            with tf.name_scope('unpool_stage0'):
                upfeats0 = graph_unpool_stage(0, upf1, vlens, vbegs, vcens)
                upf0 = tf.concat([upfeats0, fc0, lf0], axis=1)

            lf = tf.concat([fc0, lf0], axis=1)

    return upf0, lf


def graph_conv_semantic_pool_stage(stage_idx, dxyz, feats, gfc_dims, final_dim, reuse):
    with tf.name_scope('stage_{}'.format(stage_idx)):
        with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
            with tf.name_scope('fc_global{}'.format(stage_idx)):
                fc = tf.concat([feats, dxyz], axis=1)
                for i, gfd in enumerate(gfc_dims):
                    fc = tf.contrib.layers.fully_connected(fc, num_outputs=gfd,
                                                           scope='{}_gfc{}'.format(stage_idx, i))
                fc_final = tf.contrib.layers.fully_connected(fc, num_outputs=final_dim, activation_fn=None,
                                                             scope='{}_gfc_final'.format(stage_idx))

    return fc_final  # cfeats: [pn,fc_dims+gxyz_dim+feats_dim]


def graph_conv_semantic_pool_v1(xyzs, dxyzs, pxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with tf.name_scope('refine_graph_conv_edge_net'):
        with tf.variable_scope('refine_graph_conv_edge_net',reuse=reuse):
            with tf.name_scope('conv_stage0'):
                fc0=graph_conv_semantic_pool_stage(0,dxyzs,feats,gfc_dims=[128,128,128],final_dim=128,reuse=reuse)
                fc0_pool = graph_pool_stage(0, fc0, vlens, vbegs)

            with tf.name_scope('conv_stage1'):
                fc1, lf1 = graph_conv_pool_stage_edge_new(1,pxyzs,pxyzs,fc0_pool,128,radius=0.5,reuse=reuse,
                                                          gxyz_dim=16,gc_dims=[64,64,64,64],gfc_dims=[256,256,256],final_dim=256)
                fc1_pool = tf.reduce_max(fc1, axis=0)

            with tf.name_scope('unpool_stage1'):
                upfeats1 = tf.tile(tf.expand_dims(fc1_pool, axis=0), [tf.shape(fc1)[0], 1])
                upf1 = tf.concat([upfeats1, fc1, lf1], axis=1)

            with tf.name_scope('unpool_stage0'):
                upfeats0 = graph_unpool_stage(0, upf1, vlens, vbegs, vcens)
                upf0 = tf.concat([upfeats0, fc0], axis=1)

    return upf0


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
        global_feats,local_feats=graph_conv_pool_edge_new(xyzs,dxyzs,pxyzs,feats,vlens,vbegs,vcens,False)
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
    cxyzs, dxyzs, rgbs, covars, lbls, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens, block_mins = \
        read_pkl('data/S3DIS/sampled_train/{}'.format(train_list[0]))

    xyzs_pl = tf.placeholder(tf.float32, [None, 3], 'xyzs')
    feats_pl = tf.placeholder(tf.float32, [None, 3], 'feats')
    labels_pl = tf.placeholder(tf.int32, [None], 'labels')
    xyzs_op, pxyzs_op, dxyzs_op, feats_op, labels_op, vlens_op, vbegs_op, vcens_op = class_pooling(xyzs_pl, feats_pl, labels_pl,
                                                                                                   labels_pl, 0.5, 3.0)
    nidxs_op, nlens_op, nbegs_op, ncens_op=search_neighborhood(xyzs_pl,0.1)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.Session(config=config) as sess:
        for t in xrange(10):
            # for l in xrange(10):
                # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # run_metadata = tf.RunMetadata()

            xyzs, pxyzs, dxyzs, feats, labels, vlens, vbegs, vcens, nidxs=\
                sess.run([xyzs_op, pxyzs_op, dxyzs_op, feats_op, labels_op, vlens_op, vbegs_op, vcens_op, nidxs_op],
                         feed_dict={xyzs_pl:cxyzs[t][0],feats_pl:rgbs[t],labels_pl:lbls[t]},)
                             # options=options, run_metadata=run_metadata)

            #     fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            #     chrome_trace = fetched_timeline.generate_chrome_trace_format()
            #     with open('timeline.json', 'w') as f:
            #         f.write(chrome_trace)
            #
            # exit(0)

            print 'before avg {} after {}'.format(np.mean(nidxs_lens[t][0]),len(nidxs)/float(len(xyzs)))

            colors = np.random.randint(0, 256, [len(vlens), 3])
            pcolors = []
            for c, l in zip(colors, vlens):
                pcolors += [c for _ in xrange(l)]

            pcolors = np.asarray(pcolors, np.int32)
            output_points('test_result/before{}.txt'.format(t), xyzs, pcolors)
            output_points('test_result/after{}.txt'.format(t), pxyzs, colors)
            output_points('test_result/colors{}.txt'.format(t), xyzs, feats*127+128)

            colors=get_class_colors()
            output_points('test_result/labels{}.txt'.format(t), xyzs, colors[labels.flatten(),:])

            # test begs
            cur_len = 0
            for i in xrange(len(vlens)):
                assert cur_len == vbegs[i]
                cur_len += vlens[i]

            # test dxyzs
            for i in xrange(len(vlens)):
                bg = vbegs[i]
                ed = bg + vlens[i]
                dxyzs[bg:ed] += pxyzs[i]

            print 'diff max {} mean {} sum {}'.format(np.max(dxyzs - xyzs), np.mean(dxyzs - xyzs), np.sum(dxyzs - xyzs))

            print 'pn {} mean voxel pn {} voxel num {}'.format(len(dxyzs),np.mean(vlens),len(vlens))


if __name__=="__main__":
    test_block()
