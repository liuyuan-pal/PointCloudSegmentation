from tf_ops.graph_conv_layer import *
import tensorflow.contrib.framework as framework
from functools import partial

def variable_summaries(var,name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('{}_summaries'.format(name)):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean'.format(name), mean)
        with tf.name_scope('{}_stddev'.format(name)):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('{}_stddev'.format(name), stddev)
        tf.summary.histogram('{}_histogram'.format(name), var)


def graph_conv_net_v1(xyz, feats, cidxs, nidxs, nidxs_lens, nidxs_bgs, m, pmiu=None, reuse=False, final_dim=512):
    '''
    :param xyz:     [pn,3] xyz
    :param feats:   [pn,9] rgb+covars
    :param cidxs:
    :param nidxs:
    :param nidxs_lens:
    :param nidxs_bgs:
    :param reuse:
    :param trainable:
    :param final_dim:
    :return:
    '''
    with tf.name_scope('encoder'):
        with framework.arg_scope([tf.contrib.layers.fully_connected],activation_fn=tf.nn.relu,reuse=reuse):
            xyz_gc,lw,lw_sum=graph_conv_xyz(xyz,cidxs,nidxs,nidxs_lens,nidxs_bgs,'xyz_gc',3,m,16,compute_lw=True,pmiu=pmiu)
            sfeats=tf.concat([xyz_gc,feats],axis=1)

            gc1=graph_conv_feats(sfeats,cidxs,nidxs,nidxs_lens,nidxs_bgs,'gc1',16+12,m,16,lw,lw_sum)
            gc1=tf.concat([gc1, sfeats], axis=1)
            fc1=tf.contrib.layers.fully_connected(gc1, num_outputs=32, scope='fc1_1')
            fc1=tf.concat([fc1, sfeats], axis=1)
            fc1 = tf.contrib.layers.fully_connected(fc1, num_outputs=32, scope='fc1_2')
            fc1 = tf.concat([fc1, sfeats], axis=1)
            fc1 = tf.contrib.layers.fully_connected(fc1, num_outputs=32, scope='fc1_3')
            fc1 = tf.concat([fc1, sfeats], axis=1)

            gc2=graph_conv_feats(fc1,cidxs,nidxs,nidxs_lens,nidxs_bgs,'gc2',16+12+32,m,32,lw,lw_sum)
            gc2=tf.concat([gc2, fc1], axis=1)
            fc2=tf.contrib.layers.fully_connected(gc2, num_outputs=64, scope='fc2_1')
            fc2=tf.concat([fc2, fc1], axis=1)
            fc2=tf.contrib.layers.fully_connected(fc2, num_outputs=64, scope='fc2_2')
            fc2=tf.concat([fc2, fc1], axis=1)

            gc3=graph_conv_feats(fc2,cidxs,nidxs,nidxs_lens,nidxs_bgs,'gc3',32+16+12+64,m,64,lw,lw_sum)
            gc3=tf.concat([gc3, fc2], axis=1)
            fc3=tf.contrib.layers.fully_connected(gc3, num_outputs=128, scope='fc3_1')
            fc3=tf.concat([fc3, fc2], axis=1)
            fc3=tf.contrib.layers.fully_connected(fc3,num_outputs=128,scope='fc3_2')
            fc3=tf.concat([fc3, fc2], axis=1)

            fc4=tf.contrib.layers.fully_connected(fc3,num_outputs=final_dim,scope='fc5',activation_fn=None)

        fc4_reduce = tf.reduce_max(fc4, axis=0)

    return fc4_reduce, fc4


def graph_conv_net_v2(xyz, feats, cidxs, nidxs, nidxs_lens, nidxs_bgs, m, pmiu=None, reuse=False, final_dim=512):
    '''
    :param xyz:     [pn,3] xyz
    :param feats:   [pn,9] rgb+covars
    :param cidxs:
    :param nidxs:
    :param nidxs_lens:
    :param nidxs_bgs:
    :param reuse:
    :param trainable:
    :param final_dim:
    :return:
    '''
    with tf.name_scope('encoder'):
        with framework.arg_scope([tf.contrib.layers.fully_connected],activation_fn=tf.nn.relu,reuse=reuse):
            xyz_gc,lw,lw_sum=graph_conv_xyz(xyz,cidxs,nidxs,nidxs_lens,nidxs_bgs,'xyz_gc',3,m,16,compute_lw=True,pmiu=pmiu)
            sfeats=tf.concat([xyz_gc,feats],axis=1)

            gc1=graph_conv_feats(sfeats,cidxs,nidxs,nidxs_lens,nidxs_bgs,'gc1',16+12,m,32,lw,lw_sum)
            gc1=tf.concat([gc1,sfeats],axis=1)
            fc1=tf.contrib.layers.fully_connected(gc1, num_outputs=32, scope='fc1')
            fc1=tf.concat([fc1,sfeats], axis=1)

            gc2=graph_conv_feats(fc1,cidxs,nidxs,nidxs_lens,nidxs_bgs,'gc2',16+12+32,m,64,lw,lw_sum)
            gc2=tf.concat([gc2, fc1], axis=1)
            fc2=tf.contrib.layers.fully_connected(gc2, num_outputs=64, scope='fc2')
            fc2=tf.concat([fc2, fc1], axis=1)

            gc3=graph_conv_feats(fc2,cidxs,nidxs,nidxs_lens,nidxs_bgs,'gc3',32+16+12+64,m,128,lw,lw_sum)
            gc3=tf.concat([gc3, fc2], axis=1)
            fc3=tf.contrib.layers.fully_connected(gc3, num_outputs=128, scope='fc3')
            fc3=tf.concat([fc3, fc2], axis=1) # fc3:128(0.6) +fc2:64(0.4) +fc1:32(0.2) +sfeats:28(0.2)
            fc3_xyz=tf.concat([fc3, xyz],axis=1)

            fc4=tf.contrib.layers.fully_connected(fc3_xyz,num_outputs=256,scope='fc4')
            fc5=tf.contrib.layers.fully_connected(fc4,num_outputs=256,scope='fc5')
            fc6=tf.contrib.layers.fully_connected(fc5,num_outputs=final_dim,scope='fc6',activation_fn=None)

        fc6_reduce = tf.reduce_max(fc6, axis=0)

    return fc6_reduce, fc6, fc3


def graph_conv_block(ifeats,layer_idx,ifn,ofn,num_out,m,lw,lw_sum,cidxs,nidxs,nidxs_lens,nidxs_bgs,reuse,num_monitor=False):
    with tf.name_scope('gc_fc{}'.format(layer_idx)):

        gc=graph_conv_feats(ifeats,cidxs,nidxs,nidxs_lens,nidxs_bgs,'gc{}'.format(layer_idx),ifn,m,ofn,lw,lw_sum)    # 28
        if num_monitor:   variable_summaries(gc,'gc{}'.format(layer_idx))
        gc=tf.concat([gc,ifeats],axis=1)  # 16+28

        fc=tf.contrib.layers.fully_connected(gc, num_outputs=num_out, scope='fc{}'.format(layer_idx),
                                             activation_fn=tf.nn.relu,reuse=reuse)
        fc=tf.concat([fc,ifeats], axis=1)

    return fc


def graph_conv_net_v3(xyz, feats, cidxs, nidxs, nidxs_lens, nidxs_bgs, m, pmiu=None, reuse=False, final_dim=512, num_monitor=False):
    '''
    :param xyz:     [pn,3] xyz
    :param feats:   [pn,9] rgb+covars
    :param cidxs:
    :param nidxs:
    :param nidxs_lens:
    :param nidxs_bgs:
    :param reuse:
    :param trainable:
    :param final_dim:
    :return:
    '''
    with tf.name_scope('encoder'):
        with framework.arg_scope([tf.contrib.layers.fully_connected],activation_fn=tf.nn.relu,reuse=reuse):
            with tf.name_scope('gc_xyz'):
                xyz_gc,lw,lw_sum=graph_conv_xyz(xyz,cidxs,nidxs,nidxs_lens,nidxs_bgs,'xyz_gc',3,m,16,compute_lw=True,pmiu=pmiu)
                if num_monitor:
                    variable_summaries(xyz_gc,'xyz_gc')
                sfeats=tf.concat([xyz_gc,feats],axis=1)

            fc1=graph_conv_block(sfeats, 1, 16+12, 16, 32, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)
            fc2=graph_conv_block(fc1, 2, 32+16+12, 16, 32, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)
            fc3=graph_conv_block(fc2, 3, 32*2+16+12, 16, 32, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)
            fc4=graph_conv_block(fc3, 4, 32*3+16+12, 32, 32, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)
            fc5=graph_conv_block(fc4, 5, 32*4+16+12, 32, 32, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)
            fc6=graph_conv_block(fc5, 6, 32*5+16+12, 32, 32, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)
            fc7=graph_conv_block(fc6, 7, 32*6+16+12, 64, 64, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)

            with tf.name_scope('fc_global'):
                fc7_xyz=tf.concat([fc7, xyz],axis=1)
                fc8=tf.contrib.layers.fully_connected(fc7_xyz,num_outputs=256,scope='fc8')
                if num_monitor:
                    variable_summaries(fc8,'fc8')
                fc9=tf.contrib.layers.fully_connected(fc8,num_outputs=256,scope='fc9')
                if num_monitor:
                    variable_summaries(fc9,'fc9')
                fc10=tf.contrib.layers.fully_connected(fc9,num_outputs=final_dim,scope='fc10',activation_fn=None)
                if num_monitor:
                    variable_summaries(fc10,'fc10')

        fc10_reduce = tf.reduce_max(fc10, axis=0)

    return fc10_reduce, fc10, fc7


def graph_conv_net_v4(xyz, feats, cidxs, nidxs, nidxs_lens, nidxs_bgs, m, pmiu=None, reuse=False, final_dim=512, num_monitor=False):
    '''
    :param xyz:     [pn,3] xyz
    :param feats:   [pn,9] rgb+covars
    :param cidxs:
    :param nidxs:
    :param nidxs_lens:
    :param nidxs_bgs:
    :param reuse:
    :param trainable:
    :param final_dim:
    :return:
    '''
    with tf.name_scope('encoder'):
        with framework.arg_scope([tf.contrib.layers.fully_connected],activation_fn=tf.nn.relu,reuse=reuse):
            with tf.name_scope('gc_xyz'):
                xyz_gc,lw,lw_sum=graph_conv_xyz(xyz,cidxs,nidxs,nidxs_lens,nidxs_bgs,'xyz_gc',3,m,16,compute_lw=True,pmiu=pmiu)
                if num_monitor:
                    variable_summaries(xyz_gc,'xyz_gc')
                sfeats=tf.concat([xyz_gc,feats],axis=1)

            fc1=graph_conv_block(sfeats, 1, 16+12, 16, 32, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)
            fc2=graph_conv_block(fc1, 2, 32+16+12, 16, 32, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)
            fc3=graph_conv_block(fc2, 3, 32*2+16+12, 16, 32, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)
            fc4=graph_conv_block(fc3, 4, 32*3+16+12, 32, 32, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)
            fc5=graph_conv_block(fc4, 5, 32*4+16+12, 32, 32, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)
            fc6=graph_conv_block(fc5, 6, 32*5+16+12, 32, 32, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)
            fc7=graph_conv_block(fc6, 7, 32*6+16+12, 64, 64, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)

            with tf.name_scope('fc_global'):
                fc7_xyz=tf.concat([fc7, xyz],axis=1)
                fc8=tf.contrib.layers.fully_connected(fc7_xyz,num_outputs=128,scope='fc8')
                if num_monitor:  variable_summaries(fc8,'fc8')
                fc8=tf.concat([fc8,fc7_xyz],axis=1)

                fc9=tf.contrib.layers.fully_connected(fc8,num_outputs=128,scope='fc9')
                if num_monitor:  variable_summaries(fc9,'fc9')
                fc9=tf.concat([fc9,fc7_xyz],axis=1)

                fc10=tf.contrib.layers.fully_connected(fc9,num_outputs=256,scope='fc10')
                if num_monitor:  variable_summaries(fc10,'fc10')
                fc10=tf.concat([fc10,fc7_xyz],axis=1)

                fc11=tf.contrib.layers.fully_connected(fc10,num_outputs=256,scope='fc11')
                if num_monitor:  variable_summaries(fc11,'fc11')
                fc11=tf.concat([fc11,fc7_xyz],axis=1)

                fc12=tf.contrib.layers.fully_connected(fc11,num_outputs=final_dim,scope='fc12')

            fc12_reduce = tf.reduce_max(fc12, axis=0)

    return fc12_reduce, fc12, fc7


def graph_conv_net_v5_bn(xyz, feats, cidxs, nidxs, nidxs_lens, nidxs_bgs, m, is_training, pmiu=None, reuse=False,
                         final_dim=512, num_monitor=False):
    '''
    :param xyz:     [pn,3] xyz
    :param feats:   [pn,9] rgb+covars
    :param cidxs:
    :param nidxs:
    :param nidxs_lens:
    :param nidxs_bgs:
    :param reuse:
    :param trainable:
    :param final_dim:
    :return:
    '''
    with tf.name_scope('encoder'):
        with framework.arg_scope([tf.contrib.layers.fully_connected],activation_fn=tf.nn.relu,reuse=reuse):
            with framework.arg_scope([tf.contrib.layers.batch_norm],reuse=reuse,is_training=is_training):
                with tf.name_scope('gc_xyz'):
                    xyz_gc,lw,lw_sum=graph_conv_xyz(xyz,cidxs,nidxs,nidxs_lens,nidxs_bgs,'xyz_gc',3,m,16,
                                                    compute_lw=True,pmiu=pmiu)
                    xyz_gc=tf.contrib.layers.batch_norm(xyz_gc,scope='xyz_gc')
                    sfeats=tf.concat([xyz_gc,feats],axis=1)

                with tf.name_scope('gc_fc1'):
                    gc1=graph_conv_feats(sfeats,cidxs,nidxs,nidxs_lens,nidxs_bgs,'gc1',16+12,m,16,lw,lw_sum)    # 28
                    gc1=tf.contrib.layers.batch_norm(gc1,scope='gc1')
                    gc1=tf.concat([gc1,sfeats],axis=1)  # 16+28

                    fc1=tf.contrib.layers.fully_connected(gc1, num_outputs=32, scope='fc1')
                    fc1=tf.contrib.layers.batch_norm(fc1,scope='fc1')
                    fc1=tf.concat([fc1,sfeats], axis=1)

                with tf.name_scope('gc_fc2'):
                    gc2=graph_conv_feats(fc1,cidxs,nidxs,nidxs_lens,nidxs_bgs,'gc2',32+16+12,m,16,lw,lw_sum)
                    gc2=tf.contrib.layers.batch_norm(gc2,scope='gc2')
                    gc2=tf.concat([gc2, fc1], axis=1)

                    fc2=tf.contrib.layers.fully_connected(gc2, num_outputs=32, scope='fc2')
                    fc2=tf.contrib.layers.batch_norm(fc2,scope='fc2')
                    fc2=tf.concat([fc2, fc1], axis=1)

                with tf.name_scope('gc_fc3'):
                    gc3=graph_conv_feats(fc2,cidxs,nidxs,nidxs_lens,nidxs_bgs,'gc3',32*2+16+12,m,16,lw,lw_sum)
                    gc3=tf.contrib.layers.batch_norm(gc3,scope='gc3')
                    gc3=tf.concat([gc3, fc2], axis=1)

                    fc3=tf.contrib.layers.fully_connected(gc3, num_outputs=32, scope='fc3')
                    fc3=tf.contrib.layers.batch_norm(fc3,scope='fc3')
                    fc3=tf.concat([fc3, fc2], axis=1)

                with tf.name_scope('gc_fc4'):
                    gc4=graph_conv_feats(fc3,cidxs,nidxs,nidxs_lens,nidxs_bgs,'gc4',32*3+16+12,m,32,lw,lw_sum)
                    gc4=tf.contrib.layers.batch_norm(gc4,scope='gc4')
                    gc4=tf.concat([gc4, fc3], axis=1)

                    fc4=tf.contrib.layers.fully_connected(gc4, num_outputs=32, scope='fc4')
                    fc4=tf.contrib.layers.batch_norm(fc4,scope='fc4')
                    fc4=tf.concat([fc4, fc3], axis=1)

                with tf.name_scope('gc_fc5'):
                    gc5=graph_conv_feats(fc4,cidxs,nidxs,nidxs_lens,nidxs_bgs,'gc5',32*4+16+12,m,32,lw,lw_sum)
                    gc5=tf.contrib.layers.batch_norm(gc5,scope='gc5')
                    gc5=tf.concat([gc5, fc4], axis=1)

                    fc5=tf.contrib.layers.fully_connected(gc5, num_outputs=32, scope='fc5')
                    fc5=tf.contrib.layers.batch_norm(fc5,scope='fc5')
                    fc5=tf.concat([fc5, fc4], axis=1)

                with tf.name_scope('gc_fc6'):
                    gc6=graph_conv_feats(fc5,cidxs,nidxs,nidxs_lens,nidxs_bgs,'gc6',32*5+16+12,m,32,lw,lw_sum)
                    gc6=tf.contrib.layers.batch_norm(gc6,scope='gc6')
                    gc6=tf.concat([gc6, fc5], axis=1)

                    fc6=tf.contrib.layers.fully_connected(gc6, num_outputs=32, scope='fc6')
                    fc6=tf.contrib.layers.batch_norm(fc6,scope='fc6')
                    fc6=tf.concat([fc6, fc5], axis=1)

                with tf.name_scope('gc_fc7'):
                    gc7=graph_conv_feats(fc6,cidxs,nidxs,nidxs_lens,nidxs_bgs,'gc7',32*6+16+12,m,64,lw,lw_sum)
                    gc7=tf.contrib.layers.batch_norm(gc7,scope='gc7')
                    gc7=tf.concat([gc7, fc6], axis=1)

                    fc7=tf.contrib.layers.fully_connected(gc7, num_outputs=64, scope='fc7')
                    fc7=tf.contrib.layers.batch_norm(fc7,scope='fc7')
                    fc7=tf.concat([fc7, fc6], axis=1)       # 64+32*6+16+12=284

                with tf.name_scope('fc_global'):
                    fc7_xyz=tf.concat([fc7, xyz],axis=1)
                    fc8=tf.contrib.layers.fully_connected(fc7_xyz,num_outputs=256,scope='fc8')
                    fc8=tf.contrib.layers.batch_norm(fc8,scope='fc8')
                    fc8=tf.concat([fc8,fc7_xyz],axis=1)

                    fc9=tf.contrib.layers.fully_connected(fc8,num_outputs=256,scope='fc9')
                    fc9=tf.contrib.layers.batch_norm(fc9,scope='fc9')
                    fc9=tf.concat([fc9,fc7_xyz],axis=1)

                    fc10=tf.contrib.layers.fully_connected(fc9,num_outputs=final_dim,scope='fc10',activation_fn=None)
                    if num_monitor:
                        variable_summaries(fc10,'fc10')

        fc10_reduce = tf.reduce_max(fc10, axis=0)

    return fc10_reduce, fc10, fc7


def graph_conv_net_v6(xyz, feats, cidxs, nidxs, nidxs_lens, nidxs_bgs, m, pmiu=None, reuse=False, final_dim=512, num_monitor=False):
    '''
    :param xyz:     [pn,3] xyz
    :param feats:   [pn,9] rgb+covars
    :param cidxs:
    :param nidxs:
    :param nidxs_lens:
    :param nidxs_bgs:
    :param reuse:
    :param trainable:
    :param final_dim:
    :return:
    '''
    with tf.name_scope('encoder'):
        with framework.arg_scope([tf.contrib.layers.fully_connected],activation_fn=tf.nn.relu,reuse=reuse):
            with tf.name_scope('gc_xyz'):
                xyz_gc,lw,lw_sum=graph_conv_xyz(xyz,cidxs,nidxs,nidxs_lens,nidxs_bgs,'xyz_gc',3,m,16,compute_lw=True,pmiu=pmiu)
                if num_monitor:
                    variable_summaries(xyz_gc,'xyz_gc')
                sfeats=tf.concat([xyz_gc,feats],axis=1)

            fc1=graph_conv_block(sfeats, 1, 16+12, 16, 32, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)
            fc2=graph_conv_block(fc1, 2, 32+16+12, 16, 32, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)
            fc3=graph_conv_block(fc2, 3, 32*2+16+12, 16, 32, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)
            fc4=graph_conv_block(fc3, 4, 32*3+16+12, 32, 32, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)
            fc5=graph_conv_block(fc4, 5, 32*4+16+12, 32, 32, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)
            fc6=graph_conv_block(fc5, 6, 32*5+16+12, 32, 32, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)
            fc7=graph_conv_block(fc6, 7, 32*6+16+12, 64, 64, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)
            fc8=graph_conv_block(fc7, 8, 64+32*6+16+12, 64, 64, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)
            fc9=graph_conv_block(fc8, 9, 64*2+32*6+16+12, 64, 64, m, lw, lw_sum, cidxs, nidxs, nidxs_lens, nidxs_bgs, reuse, num_monitor)

            with tf.name_scope('fc_global'):
                fc9_xyz=tf.concat([fc9, xyz],axis=1)
                fc10=tf.contrib.layers.fully_connected(fc9_xyz,num_outputs=256,scope='fc10')
                if num_monitor:  variable_summaries(fc10,'fc10')
                fc11=tf.contrib.layers.fully_connected(fc10,num_outputs=256,scope='fc11')
                if num_monitor:  variable_summaries(fc11,'fc9')
                fc12=tf.contrib.layers.fully_connected(fc11,num_outputs=final_dim,scope='fc12',activation_fn=None)
                if num_monitor:  variable_summaries(fc12,'fc12')

        fc12_reduce = tf.reduce_max(fc12, axis=0)

    return fc12_reduce, fc12, fc9


def graph_conv_pool_block(ifeats,stage_idx,layer_idx,ifn,ofn,num_out,
                          nidxs,nidxs_lens,nidxs_bgs,cidxs,
                          m,lw,lw_sum,reuse):
    with tf.name_scope('{}_gc_fc{}'.format(stage_idx,layer_idx)):
        gc=graph_conv_feats(ifeats,cidxs,nidxs,nidxs_lens,nidxs_bgs,'{}_gc{}'.format(stage_idx,layer_idx),ifn,m,ofn,lw,lw_sum)    # 28
        gc=tf.concat([gc,ifeats],axis=1)  # 16+28
        fc=tf.contrib.layers.fully_connected(gc, num_outputs=num_out, scope='{}_fc{}'.format(stage_idx,layer_idx),
                                             activation_fn=tf.nn.relu,reuse=reuse)
        fc=tf.concat([fc,ifeats], axis=1)
    return fc


def graph_conv_pool_stage(stage_idx, cxyzs, dxyz, feats, cidxs, nidxs, nidxs_lens, nidxs_bgs,
                          m, feats_dim, gxyz_dim, gc_dims, fc_dims, gfc_dims, final_dim, pmiu, reuse):
    with tf.name_scope('gc_xyz{}'.format(stage_idx)):
        xyz_gc,lw,lw_sum=graph_conv_xyz(cxyzs,cidxs,nidxs,nidxs_lens,nidxs_bgs,'xyz_gc{}'.format(stage_idx),
                                        3,m,gxyz_dim,compute_lw=True,pmiu=pmiu)

        cfeats=tf.concat([xyz_gc, feats], axis=1)
        cdim=feats_dim+gxyz_dim

        block_fn=partial(graph_conv_pool_block,nidxs=nidxs,nidxs_lens=nidxs_lens,nidxs_bgs=nidxs_bgs,
                                               cidxs=cidxs,m=m,lw=lw,lw_sum=lw_sum,reuse=reuse)
        layer_idx=1
        for gd,fd in zip(gc_dims,fc_dims):
            cfeats=block_fn(cfeats,stage_idx,layer_idx,cdim,gd,fd)
            layer_idx+=1
            cdim+=fd

        with framework.arg_scope([tf.contrib.layers.fully_connected],activation_fn=tf.nn.relu,reuse=reuse):
            with tf.name_scope('fc_global{}'.format(stage_idx)):
                fc=tf.concat([cfeats, dxyz],axis=1)
                for i,gfd in enumerate(gfc_dims):
                    fc=tf.contrib.layers.fully_connected(fc,num_outputs=gfd,scope='{}_gfc{}'.format(stage_idx,i))
                fc_final=tf.contrib.layers.fully_connected(fc,num_outputs=final_dim,activation_fn=None,
                                                           scope='{}_gfc_final'.format(stage_idx))

    return fc_final, cfeats  # cfeats: [pn,fc_dims+gxyz_dim+feats_dim]


def graph_pool_stage(stage_idx, feats, vlens, vlens_bgs):
    with tf.name_scope('{}_pool'.format(stage_idx)):
        pfeats=graph_pool(feats,vlens,vlens_bgs)
    return pfeats

def graph_unpool_stage(stage_idx, feats, vlens, vlens_bgs,vcidxs):
    with tf.name_scope('{}_unpool'.format(stage_idx)):
        pfeats=graph_unpool(feats,vlens,vlens_bgs,vcidxs)
    return pfeats


def graph_conv_pool_v1(cxyzs, dxyzs, rgbs, covars, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_lens, nidxs_bgs, m, pmiu, reuse=False):
    with tf.name_scope('graph_conv_pool_net'):
        with tf.name_scope('conv_stage0'):
            feats0=tf.concat([rgbs,covars],axis=1)
            # fc0 [pn0,32] lf0 [pn0,76]
            fc0,lf0=graph_conv_pool_stage(0,cxyzs[0],dxyzs[0],feats0,cidxs[0],nidxs[0],nidxs_lens[0],nidxs_bgs[0],
                                          m, 12, 8, [8, 16, 32], [8, 16, 32], [32, 32, 32], 32, pmiu, reuse)
            # fc0_pool [pn1,]
            fc0_pool=graph_pool_stage(0,fc0,vlens[0],vlens_bgs[0])

        with tf.name_scope('conv_stage1'):
            # fc1 [pn1,128] lf1 [pn1,480]
            fc1,lf1=graph_conv_pool_stage(1,cxyzs[1],dxyzs[1],fc0_pool,cidxs[1],nidxs[1],nidxs_lens[1],nidxs_bgs[1],
                                          m, 32, 8, [32, 32, 32, 64, 64, 64], [32, 32, 32, 64, 64, 64],
                                          [128, 128, 128], 128, pmiu, reuse)
            # fc1_pool [pn2,]
            fc1_pool=graph_pool_stage(1,fc1,vlens[1],vlens_bgs[1])

        with tf.name_scope('conv_stage2'):
            # fc2 [pn2,256] lf2 [pn2,648]
            fc2,lf2=graph_conv_pool_stage(2,cxyzs[2],cxyzs[2],fc1_pool,cidxs[2],nidxs[2],nidxs_lens[2],nidxs_bgs[2],
                                          m, 128, 8, [128, 128, 256], [128, 128, 256], [256, 256], 256, pmiu, reuse)
            # [1,256]
            fc2_pool=tf.reduce_max(fc2,axis=0)


        with tf.name_scope('unpool_stage2'):
            # [pn3,256]
            upfeats2=tf.tile(tf.expand_dims(fc2_pool,axis=0),[tf.shape(fc2)[0],1])
            # [pn3,256+256+648=1260] [fc2_pool,fc2,lf2]
            upf2=tf.concat([upfeats2,fc2,lf2],axis=1)

        with tf.name_scope('unpool_stage1'):
            # [pn2,1260]
            upfeats1=graph_unpool_stage(1,upf2,vlens[1],vlens_bgs[1],vcidxs[1])
            # [pn2,1260+128+480=1868] [fc2_pool,fc2,lf2,fc1,lf1]
            upf1=tf.concat([upfeats1,fc1,lf1],axis=1)

        with tf.name_scope('unpool_stage0'):
            # [pn1,1868+32+76=1976] [fc2_pool,fc2,lf2,fc1,lf1,fc0,lf0]
            upfeats0=graph_unpool_stage(0,upf1,vlens[0],vlens_bgs[0],vcidxs[0])
            upf0=tf.concat([upfeats0,fc0,lf0],axis=1)

        return upf0


def classifier(feats, pfeats, is_training, num_classes, reuse=False, use_bn=False):
    '''

    :param feats: n,k,f
    :param pfeats:
    :param is_training:
    :param num_classes:
    :param reuse:
    :return:
    '''
    normalizer_params={'scale':False,'is_training':is_training,'reuse':reuse}
    feats=tf.expand_dims(feats,axis=2)              # n,k,1,2048+6
    pfeats=tf.expand_dims(pfeats, axis=2)  # n,k,1,6
    bn=tf.contrib.layers.batch_norm if use_bn else None
    with tf.name_scope('segmentation_classifier'):
        with framework.arg_scope([tf.contrib.layers.conv2d],kernel_size=[1,1],stride=1,
                                 padding='VALID',activation_fn=tf.nn.relu,reuse=reuse,
                                 normalizer_fn=bn):

            normalizer_params['scope']='class_mlp1_bn'
            class_mlp1 = tf.contrib.layers.conv2d(
                feats, num_outputs=512, scope='class_mlp1',normalizer_params=normalizer_params)
            class_mlp1=tf.concat([class_mlp1, pfeats], axis=3)

            normalizer_params['scope']='class_mlp2_bn'
            class_mlp2 = tf.contrib.layers.conv2d(
                class_mlp1, num_outputs=256, scope='class_mlp2',normalizer_params=normalizer_params)
            class_mlp2=tf.concat([class_mlp2, pfeats], axis=3)
            # tf.cond(is_training,lambda:tf.nn.dropout(class_mlp2,0.7),lambda:class_mlp2)

            logits = tf.contrib.layers.conv2d(
                class_mlp2, num_outputs=num_classes, scope='class_mlp3',activation_fn=None,normalizer_fn=None)

        logits=tf.squeeze(logits,axis=2,name='logits')

    return logits


def classifier_v3(feats, pfeats, is_training, num_classes, reuse=False, use_bn=False):
    '''

    :param feats: n,k,f
    :param pfeats:
    :param is_training:
    :param num_classes:
    :param reuse:
    :return:
    '''
    normalizer_params={'scale':False,'is_training':is_training,'reuse':reuse}
    feats=tf.expand_dims(feats,axis=2)              # n,k,1,2048+6
    pfeats=tf.expand_dims(pfeats, axis=2)  # n,k,1,6
    bn=tf.contrib.layers.batch_norm if use_bn else None
    with tf.name_scope('segmentation_classifier'):
        with framework.arg_scope([tf.contrib.layers.conv2d],kernel_size=[1,1],stride=1,
                                 padding='VALID',activation_fn=tf.nn.relu,reuse=reuse,
                                 normalizer_fn=bn):

            # pfeats = tf.cond(is_training,lambda:tf.nn.dropout(pfeats,0.7),lambda:pfeats)
            # feats = tf.cond(is_training,lambda:tf.nn.dropout(feats,0.7),lambda:feats)
            normalizer_params['scope']='class_mlp1_bn'
            class_mlp1 = tf.contrib.layers.conv2d(
                feats, num_outputs=512, scope='class_mlp1',normalizer_params=normalizer_params)
            class_mlp1=tf.concat([class_mlp1, pfeats], axis=3)
            class_mlp1 = tf.cond(is_training, lambda: tf.nn.dropout(class_mlp1, 0.7), lambda: class_mlp1)

            normalizer_params['scope']='class_mlp2_bn'
            class_mlp2 = tf.contrib.layers.conv2d(
                class_mlp1, num_outputs=256, scope='class_mlp2',normalizer_params=normalizer_params)
            class_mlp2=tf.concat([class_mlp2, pfeats], axis=3)
            class_mlp2=tf.cond(is_training,lambda:tf.nn.dropout(class_mlp2,0.7),lambda:class_mlp2)

            logits = tf.contrib.layers.conv2d(
                class_mlp2, num_outputs=num_classes, scope='class_mlp3',activation_fn=None,normalizer_fn=None)

        logits=tf.squeeze(logits,axis=2,name='logits')

    return logits


def classifier_v2(feats, is_training, num_classes, reuse=False, use_bn=False):
    '''

    :param feats: n,k,f
    :param pfeats:
    :param is_training:
    :param num_classes:
    :param reuse:
    :return:
    '''
    normalizer_params={'scale':False,'is_training':is_training,'reuse':reuse}
    feats=tf.expand_dims(feats,axis=2)              # n,k,1,2048+6
    bn=tf.contrib.layers.batch_norm if use_bn else None
    with tf.name_scope('segmentation_classifier'):
        with framework.arg_scope([tf.contrib.layers.conv2d],kernel_size=[1,1],stride=1,
                                 padding='VALID',activation_fn=tf.nn.relu,reuse=reuse,
                                 normalizer_fn=bn):

            normalizer_params['scope']='class_mlp1_bn'
            class_mlp1 = tf.contrib.layers.conv2d(
                feats, num_outputs=512, scope='class_mlp1',normalizer_params=normalizer_params)
            class_mlp1=tf.cond(is_training,lambda:tf.nn.dropout(class_mlp1,0.7),lambda:class_mlp1)

            normalizer_params['scope']='class_mlp2_bn'
            class_mlp2 = tf.contrib.layers.conv2d(
                class_mlp1, num_outputs=256, scope='class_mlp2',normalizer_params=normalizer_params)
            class_mlp2=tf.cond(is_training,lambda:tf.nn.dropout(class_mlp2,0.7),lambda:class_mlp2)

            logits = tf.contrib.layers.conv2d(
                class_mlp2, num_outputs=num_classes, scope='class_mlp3',activation_fn=None,normalizer_fn=None)

        logits=tf.squeeze(logits,axis=2,name='logits')

    return logits