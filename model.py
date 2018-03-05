from tf_ops.graph_conv_layer import *
import tensorflow.contrib.framework as framework


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


def graph_conv_net_v3(xyz, feats, cidxs, nidxs, nidxs_lens, nidxs_bgs, m, pmiu=None, reuse=False, final_dim=512):
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
                sfeats=tf.concat([xyz_gc,feats],axis=1)

            with tf.name_scope('gc_fc1'):
                gc1=graph_conv_feats(sfeats,cidxs,nidxs,nidxs_lens,nidxs_bgs,'gc1',16+12,m,16,lw,lw_sum)    # 28
                gc1=tf.concat([gc1,sfeats],axis=1)  # 16+28
                fc1=tf.contrib.layers.fully_connected(gc1, num_outputs=32, scope='fc1')
                fc1=tf.concat([fc1,sfeats], axis=1)

            with tf.name_scope('gc_fc2'):
                gc2=graph_conv_feats(fc1,cidxs,nidxs,nidxs_lens,nidxs_bgs,'gc2',32+16+12,m,16,lw,lw_sum)
                gc2=tf.concat([gc2, fc1], axis=1)
                fc2=tf.contrib.layers.fully_connected(gc2, num_outputs=32, scope='fc2')
                fc2=tf.concat([fc2, fc1], axis=1)

            with tf.name_scope('gc_fc3'):
                gc3=graph_conv_feats(fc2,cidxs,nidxs,nidxs_lens,nidxs_bgs,'gc3',32*2+16+12,m,16,lw,lw_sum)
                gc3=tf.concat([gc3, fc2], axis=1)
                fc3=tf.contrib.layers.fully_connected(gc3, num_outputs=32, scope='fc3')
                fc3=tf.concat([fc3, fc2], axis=1)

            with tf.name_scope('gc_fc4'):
                gc4=graph_conv_feats(fc3,cidxs,nidxs,nidxs_lens,nidxs_bgs,'gc4',32*3+16+12,m,32,lw,lw_sum)
                gc4=tf.concat([gc4, fc3], axis=1)
                fc4=tf.contrib.layers.fully_connected(gc4, num_outputs=32, scope='fc4')
                fc4=tf.concat([fc4, fc3], axis=1)

            with tf.name_scope('gc_fc5'):
                gc5=graph_conv_feats(fc4,cidxs,nidxs,nidxs_lens,nidxs_bgs,'gc5',32*4+16+12,m,32,lw,lw_sum)
                gc5=tf.concat([gc5, fc4], axis=1)
                fc5=tf.contrib.layers.fully_connected(gc5, num_outputs=32, scope='fc5')
                fc5=tf.concat([fc5, fc4], axis=1)

            with tf.name_scope('gc_fc6'):
                gc6=graph_conv_feats(fc5,cidxs,nidxs,nidxs_lens,nidxs_bgs,'gc6',32*5+16+12,m,32,lw,lw_sum)
                gc6=tf.concat([gc6, fc5], axis=1)
                fc6=tf.contrib.layers.fully_connected(gc6, num_outputs=32, scope='fc6')
                fc6=tf.concat([fc6, fc5], axis=1)

            with tf.name_scope('gc_fc7'):
                gc7=graph_conv_feats(fc6,cidxs,nidxs,nidxs_lens,nidxs_bgs,'gc7',32*6+16+12,m,64,lw,lw_sum)
                gc7=tf.concat([gc7, fc6], axis=1)
                fc7=tf.contrib.layers.fully_connected(gc7, num_outputs=64, scope='fc7')
                fc7=tf.concat([fc7, fc6], axis=1)       # 64+32*6+16+12=284

            with tf.name_scope('fc_global'):
                fc7_xyz=tf.concat([fc7, xyz],axis=1)
                fc8=tf.contrib.layers.fully_connected(fc7_xyz,num_outputs=256,scope='fc8')
                fc9=tf.contrib.layers.fully_connected(fc8,num_outputs=256,scope='fc9')
                fc10=tf.contrib.layers.fully_connected(fc9,num_outputs=final_dim,scope='fc10',activation_fn=None)

        fc10_reduce = tf.reduce_max(fc10, axis=0)

    return fc10_reduce, fc10, fc7



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