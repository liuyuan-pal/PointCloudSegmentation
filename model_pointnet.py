from tf_ops.graph_layer_new import neighbor_ops, pooling_ops, graph_concat_scatter, graph_pool, \
    search_neighborhood, graph_unpool, search_neighborhood_range, graph_avg_pool,\
    search_neighborhood_fixed_range,search_neighborhood_fixed
from model_pgnet import variable_on_cpu

import tensorflow as tf
from tensorflow.contrib import framework

feats_ops={}
def pointnet_conv(sxyzs, feats, fc_dims, ofn, name, nidxs, nlens, nbegs, ncens, reuse=None):
    with tf.name_scope(name):
        sfeats = graph_concat_scatter(feats,nidxs,nlens,nbegs,ncens)
        sfeats = tf.concat([sfeats,sxyzs],axis=1)

        for idx,fd in enumerate(fc_dims):
            cfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=fd, scope='{}_fc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            sfeats=tf.concat([cfeats,sfeats],axis=1)

        sfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=ofn, scope='{}_fc_out'.format(name),
                                                activation_fn=None, reuse=reuse)
        feats=graph_pool(sfeats,nlens,nbegs)
        feats_ops[name]=feats
        return feats

def pointnet_conv_nofeats(sxyzs, fc_dims, ofn, name, nidxs, nlens, nbegs, ncens, reuse=None):
    with tf.name_scope(name):
        sfeats = sxyzs

        for idx,fd in enumerate(fc_dims):
            cfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=fd, scope='{}_fc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            sfeats=tf.concat([cfeats,sfeats],axis=1)

        sfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=ofn, scope='{}_fc_out'.format(name),
                                                activation_fn=None, reuse=reuse)
        feats=graph_pool(sfeats,nlens,nbegs)
        feats_ops[name]=feats
        return feats

def pointnet_conv_noconcat(sxyzs, feats, fc_dims, ofn, name, nidxs, nlens, nbegs, ncens, reuse=None):
    with tf.name_scope(name):
        sfeats = graph_concat_scatter(feats,nidxs,nlens,nbegs,ncens)
        sfeats = tf.concat([sfeats,sxyzs],axis=1)

        for idx,fd in enumerate(fc_dims):
            sfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=fd, scope='{}_fc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)

        sfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=ofn, scope='{}_fc_out'.format(name),
                                                activation_fn=None, reuse=reuse)
        feats=graph_pool(sfeats,nlens,nbegs)
        feats_ops[name]=feats
        return feats

def pointnet_pool(xyzs, feats, fc_dims, ofn, name, nlens, nbegs, reuse=None):
    with tf.name_scope(name):
        sfeats = tf.concat([xyzs,feats],axis=1)

        for idx,fd in enumerate(fc_dims):
            cfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=fd, scope='{}_fc_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            sfeats=tf.concat([cfeats,sfeats],axis=1)

        sfeats=tf.contrib.layers.fully_connected(sfeats, num_outputs=ofn, scope='{}_fc_out'.format(name),
                                                 activation_fn=None, reuse=reuse)
        feats=graph_pool(sfeats,nlens,nbegs)
        feats_ops[name]=feats
        return feats,sfeats

def mlp(feats, fc_dims, final_dim, name, reuse=None):
    for idx,fd in enumerate(fc_dims):
        cfeats=tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='{}_fc_{}'.format(name,idx),
                                                 activation_fn=tf.nn.relu, reuse=reuse)
        feats=tf.concat([cfeats,feats],axis=1)

    feats=tf.contrib.layers.fully_connected(feats, num_outputs=final_dim, scope='{}_fc_out'.format(name),
                                            activation_fn=None, reuse=reuse)

    return feats

def unpool(name, feats, vlens, vlens_bgs,vcidxs):
    with tf.name_scope(name):
        pfeats=graph_unpool(feats,vlens,vlens_bgs,vcidxs)
    return pfeats

def pointnet_deconv(name, fc_dims, final_dim, pfeats, upfeats, vlens, vbegs, vcens, reuse):
    pfeats=graph_unpool(pfeats, vlens, vbegs, vcens)
    feats=tf.concat([pfeats,upfeats],axis=1)

    for idx,fd in enumerate(fc_dims):
        cfeats=tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='{}_fc_{}'.format(name,idx),
                                                 activation_fn=tf.nn.relu, reuse=reuse)
        feats=tf.concat([feats,cfeats],axis=1)

    ofeats=tf.contrib.layers.fully_connected(feats, num_outputs=final_dim, scope='{}_fc_out'.format(name),
                                             activation_fn=None, reuse=reuse)

    return ofeats

def fc_embed(feats, name, embed_dim, reuse):
    ofeats=tf.contrib.layers.fully_connected(feats, num_outputs=embed_dim, scope='{}_fc_embed'.format(name),
                                             activation_fn=tf.nn.leaky_relu, reuse=reuse)
    return ofeats

def pointnet_20_baseline(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[8,8],8,'pointnet0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[8,8],8,'pointnet1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[10,12],12,'pointnet2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[10,12],12,'pointnet3',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.1  # rescale

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[16,16],16,'pointnet4',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[16,16],16,'pointnet5',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[16,16],16,'pointnet6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[16,16],16,'pointnet7',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats_pn],axis=1)

            feats_stage0_pool,_=pointnet_pool(dxyzs[0],feats_stage0,[16,16],64,'pool0',vlens[0],vbegs[0],reuse)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.6)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.6

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[16,16],16,'pointnet8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[16,16],16,'pointnet9',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[16,16],16,'pointnet10',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[16,16],16,'pointnet11',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[24,24],24,'pointnet12',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[24,24],24,'pointnet13',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[32,32],32,'pointnet14',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[32,32],32,'pointnet15',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats_pn],axis=1)

            feats_stage1_pool,_=pointnet_pool(dxyzs[1],feats_stage1,[32,32],128,'pool1',vlens[1],vbegs[1],reuse)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[32,32],32,'pointnet16',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[32,32],32,'pointnet17',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[48,48],48,'pointnet18',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[48,48],48,'pointnet19',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            for idx, fd in enumerate([64,64,128]):
                cfeats = tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='global_{}'.format(idx),
                                                           activation_fn=tf.nn.relu, reuse=reuse)
                feats = tf.concat([feats, cfeats], axis=1)

            feats_stage2_global = tf.contrib.layers.fully_connected(feats, num_outputs=256, scope='global_out',
                                                                    activation_fn=None, reuse=reuse)
        with tf.name_scope('unpool'):
            lf2=tf.concat([feats_stage2_global,feats_stage2],axis=1)
            lf2_up=unpool('unpool1',lf2,vlens[1],vbegs[1],vcens[1])

            lf1=tf.concat([lf2_up,feats_stage1],axis=1)
            lf1_up=unpool('unpool2',lf1,vlens[0],vbegs[0],vcens[0])

            lf0=tf.concat([lf1_up,feats_stage0],axis=1)

        return lf0, feats_stage0

def pointnet_20_baseline_v2(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[8,8],8,'pointnet0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[8,8],8,'pointnet1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[10,12],12,'pointnet2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[10,12],12,'pointnet3',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.1  # rescale

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[16,16],16,'pointnet4',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[16,16],16,'pointnet5',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[16,16],16,'pointnet6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[16,16],16,'pointnet7',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats_pn],axis=1)

            feats_stage0_pool,_=pointnet_pool(dxyzs[0],feats_stage0,[16,16],64,'pool0',vlens[0],vbegs[0],reuse)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.6)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.6

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[16,16],16,'pointnet8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[16,16],16,'pointnet9',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[16,16],16,'pointnet10',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[16,16],16,'pointnet11',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[24,24],24,'pointnet12',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[24,24],24,'pointnet13',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[32,32],32,'pointnet14',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[32,32],32,'pointnet15',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats_pn],axis=1)

            feats_stage1_pool,_=pointnet_pool(dxyzs[1],feats_stage1,[32,32],128,'pool1',vlens[1],vbegs[1],reuse)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale

            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[32,32],32,'pointnet16',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[32,32],32,'pointnet17',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[48,48],48,'pointnet18',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[48,48],48,'pointnet19',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            for idx, fd in enumerate([64,64,128]):
                cfeats = tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='global_{}'.format(idx),
                                                           activation_fn=tf.nn.relu, reuse=reuse)
                feats = tf.concat([feats, cfeats], axis=1)

            feats_stage2_global = tf.contrib.layers.fully_connected(feats, num_outputs=256, scope='global_out',
                                                                    activation_fn=None, reuse=reuse)
            feats_stage2_pool = tf.reduce_max(feats_stage2_global,axis=0)

        with tf.name_scope('unpool'):
            feats_stage2_pool=tf.tile(tf.expand_dims(feats_stage2_pool,axis=0),[tf.shape(xyzs[2])[0],1])
            lf2=tf.concat([feats_stage2_pool,feats_stage2],axis=1)
            lf2_up=unpool('unpool1',lf2,vlens[1],vbegs[1],vcens[1])
            lf1=tf.concat([lf2_up,feats_stage1],axis=1)
            lf1_up=unpool('unpool2',lf1,vlens[0],vbegs[0],vcens[0])

            lf0=tf.concat([lf1_up,feats_stage0],axis=1)

        return lf0, feats_stage0

def pointnet_5_concat(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.1  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats_pn],axis=1)

            feats_stage0_pool,feats_stage0_fc=pointnet_pool(dxyzs[0],feats_stage0,[16,16],64,'pool0',vlens[0],vbegs[0],reuse)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.6)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.6  # rescale

            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[16,16,32],64,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats_pn],axis=1)

            feats_stage1_pool,feats_stage1_fc=pointnet_pool(dxyzs[1],feats_stage1,[32,32],128,'pool1',vlens[1],vbegs[1],reuse)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[32,32,48],96,'feats4',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            feats_stage2_fc=mlp(feats,[64,64,128],256,'global',reuse)
            feats_stage2_pool=tf.reduce_max(feats_stage2_fc,axis=0)

        with tf.name_scope('unpool'):
            feats_stage2_pool=tf.tile(tf.expand_dims(feats_stage2_pool,axis=0),[tf.shape(xyzs[2])[0],1])
            lf2=tf.concat([feats_stage2_pool,feats_stage2],axis=1)
            lf2_up=unpool('1',lf2,vlens[1],vbegs[1],vcens[1])
            lf1=tf.concat([lf2_up,feats_stage1],axis=1)
            lf1_up=unpool('0',lf1,vlens[0],vbegs[0],vcens[0])

            lf0=tf.concat([lf1_up,feats_stage0],axis=1)

        return lf0, feats_stage0

def pointnet_5_concat_pre(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.1  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats_pn],axis=1)

            feats_stage0_pool,feats_stage0_fc=pointnet_pool(dxyzs[0],feats_stage0,[16,16],64,'pool0',vlens[0],vbegs[0],reuse)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.6)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.6  # rescale

            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[16,16,32],64,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats_pn],axis=1)

            feats_stage1_pool,feats_stage1_fc=pointnet_pool(dxyzs[1],feats_stage1,[32,32],128,'pool1',vlens[1],vbegs[1],reuse)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[32,32,48],96,'feats4',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            feats_stage2_fc=mlp(feats,[64,64,128],256,'global',reuse)

        with tf.name_scope('unpool'):
            lf2=tf.concat([feats_stage2_fc,feats_stage2],axis=1)
            lf2_up=unpool('1',lf2,vlens[1],vbegs[1],vcens[1])
            lf1=tf.concat([lf2_up,feats_stage1,feats_stage1_fc],axis=1)
            lf1_up=unpool('0',lf1,vlens[0],vbegs[0],vcens[0])

            lf0=tf.concat([lf1_up,feats_stage0,feats_stage0_fc],axis=1)

        return lf0, feats_stage0

def pointnet_10_concat(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.1  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats_pn],axis=1)

            feats_stage0_pool,feats_stage0_fc=pointnet_pool(dxyzs[0],feats_stage0,[16,16],64,'pool0',vlens[0],vbegs[0],reuse)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.6)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.6  # rescale

            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats4',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats5',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[16,16,24],48,'feats6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[16,16,32],64,'feats7',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats_pn],axis=1)

            feats_stage1_pool,feats_stage1_fc=pointnet_pool(dxyzs[1],feats_stage1,[32,32],128,'pool1',vlens[1],vbegs[1],reuse)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[32,32,32],64,'feats8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[32,32,48],96,'feats9',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            feats_stage2_fc=mlp(feats,[64,64,128],256,'global',reuse)
            feats_stage2_pool=tf.reduce_max(feats_stage2_fc,axis=0)

        with tf.name_scope('unpool'):
            feats_stage2_pool=tf.tile(tf.expand_dims(feats_stage2_pool,axis=0),[tf.shape(xyzs[2])[0],1])
            lf2=tf.concat([feats_stage2_pool,feats_stage2],axis=1)
            lf2_up=unpool('1',lf2,vlens[1],vbegs[1],vcens[1])
            lf1=tf.concat([lf2_up,feats_stage1],axis=1)
            lf1_up=unpool('0',lf1,vlens[0],vbegs[0],vcens[0])

            lf0=tf.concat([lf1_up,feats_stage0],axis=1)

        return lf0, feats_stage0

def pointnet_10_concat_pre(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.1  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats_pn],axis=1)

            feats_stage0_pool,feats_stage0_fc=pointnet_pool(dxyzs[0],feats_stage0,[16,16],64,'pool0',vlens[0],vbegs[0],reuse)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.6)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.6  # rescale

            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats4',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats5',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[16,16,24],48,'feats6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[16,16,32],64,'feats7',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats_pn],axis=1)

            feats_stage1_pool,feats_stage1_fc=pointnet_pool(dxyzs[1],feats_stage1,[32,32],128,'pool1',vlens[1],vbegs[1],reuse)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[32,32,32],64,'feats8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[32,32,48],96,'feats9',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            feats_stage2_fc=mlp(feats,[64,64,128],256,'global',reuse)

            feats_stage2_pool=tf.reduce_max(feats_stage2_fc,axis=0)

        with tf.name_scope('unpool'):
            feats_stage2_pool=tf.tile(tf.expand_dims(feats_stage2_pool,axis=0),[tf.shape(xyzs[2])[0],1])
            lf2=tf.concat([feats_stage2_pool,feats_stage2,feats_stage2_fc],axis=1)
            lf2_up=unpool('1',lf2,vlens[1],vbegs[1],vcens[1])
            lf1=tf.concat([lf2_up,feats_stage1,feats_stage1_fc],axis=1)
            lf1_up=unpool('0',lf1,vlens[0],vbegs[0],vcens[0])
            lf0=tf.concat([lf1_up,feats_stage0,feats_stage0_fc],axis=1)

        return lf0, feats_stage0

def pointnet_10_concat_pre_deconv(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.1  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats_pn],axis=1)

            feats_stage0_pool,feats_stage0_fc=pointnet_pool(dxyzs[0],feats_stage0,[16,16],64,'pool0',vlens[0],vbegs[0],reuse)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.6)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.6  # rescale

            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats4',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats5',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[16,16,24],48,'feats6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[16,16,32],64,'feats7',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats_pn],axis=1)

            feats_stage1_pool,feats_stage1_fc=pointnet_pool(dxyzs[1],feats_stage1,[32,32],128,'pool1',vlens[1],vbegs[1],reuse)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[32,32,32],64,'feats8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[32,32,48],96,'feats9',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            feats_stage2_fc=mlp(feats,[64,64,128],256,'global',reuse)

            feats_stage2_pool=tf.reduce_max(feats_stage2_fc,axis=0)

        with tf.name_scope('unpool'):
            feats_stage2_pool=tf.tile(tf.expand_dims(feats_stage2_pool,axis=0),[tf.shape(xyzs[2])[0],1])
            upfeats2=tf.concat([feats_stage2_pool,feats_stage2,xyzs[2]],axis=1)
            upfeats2=mlp(upfeats2,[64,64],256,'unpool2',reuse)
            lf2=tf.concat([upfeats2,feats_stage2_pool,feats_stage2_fc,feats_stage2],axis=1)

            lf2_up=unpool('1',lf2,vlens[1],vbegs[1],vcens[1])
            upfeats1=tf.concat([lf2_up,feats_stage1,dxyzs[1]],axis=1)
            upfeats1=mlp(upfeats1,[64,128],256,'unpool1',reuse)
            lf1=tf.concat([upfeats1,lf2_up,feats_stage1,feats_stage1_fc],axis=1)

            lf1_up=unpool('0',lf1,vlens[0],vbegs[0],vcens[0])
            upfeats0=tf.concat([lf1_up,feats_stage0,dxyzs[0]],axis=1)
            upfeats0=mlp(upfeats0,[128,128],256,'unpool0',reuse)
            lf0=tf.concat([upfeats0,lf1_up,feats_stage0,feats_stage0_fc],axis=1)

        return lf0, feats_stage0

def pointnet_5_concat_pre_deconv(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.1  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats_pn],axis=1)

            feats_stage0_pool,feats_stage0_fc=pointnet_pool(dxyzs[0],feats_stage0,[16,16],64,'pool0',vlens[0],vbegs[0],reuse)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.6)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.6  # rescale

            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[16,16,32],64,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats_pn],axis=1)

            feats_stage1_pool,feats_stage1_fc=pointnet_pool(dxyzs[1],feats_stage1,[32,32],128,'pool1',vlens[1],vbegs[1],reuse)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[32,32,48],96,'feats4',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            feats_stage2_fc=mlp(feats,[64,64,128],256,'global',reuse)

            feats_stage2_pool=tf.reduce_max(feats_stage2_fc,axis=0)

        with tf.name_scope('unpool'):
            feats_stage2_pool=tf.tile(tf.expand_dims(feats_stage2_pool,axis=0),[tf.shape(xyzs[2])[0],1])
            upfeats2=tf.concat([feats_stage2_pool,feats_stage2,xyzs[2]],axis=1)
            upfeats2=mlp(upfeats2,[64,64],256,'unpool2',reuse)
            lf2=tf.concat([upfeats2,feats_stage2_pool,feats_stage2_fc,feats_stage2],axis=1)

            lf2_up=unpool('1',lf2,vlens[1],vbegs[1],vcens[1])
            upfeats1=tf.concat([lf2_up,feats_stage1,dxyzs[1]],axis=1)
            upfeats1=mlp(upfeats1,[32,32],128,'unpool1',reuse)
            lf1=tf.concat([upfeats1,lf2_up,feats_stage1,feats_stage1_fc],axis=1)

            lf1_up=unpool('0',lf1,vlens[0],vbegs[0],vcens[0])
            upfeats0=tf.concat([lf1_up,feats_stage0,dxyzs[0]],axis=1)
            upfeats0=mlp(upfeats0,[16,16],64,'unpool0',reuse)
            lf0=tf.concat([upfeats0,lf1_up,feats_stage0,feats_stage0_fc],axis=1)

        return lf0, feats_stage0

def pointnet_10_dilated(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[0], 0.1, 0.15)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.1  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats_pn],axis=1)

            feats_stage0_pool,feats_stage0_fc=pointnet_pool(dxyzs[0],feats_stage0,[16,16],64,'pool0',vlens[0],vbegs[0],reuse)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.6)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.6  # rescale

            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats4',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats5',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[1], 0.3, 0.45)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[16,16,24],48,'feats6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[16,16,32],64,'feats7',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats_pn],axis=1)

            feats_stage1_pool,feats_stage1_fc=pointnet_pool(dxyzs[1],feats_stage1,[32,32],128,'pool1',vlens[1],vbegs[1],reuse)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[32,32,32],64,'feats8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[32,32,48],96,'feats9',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            feats_stage2_fc=mlp(feats,[64,64,128],256,'global',reuse)

            feats_stage2_pool=tf.reduce_max(feats_stage2_fc,axis=0)

        with tf.name_scope('unpool'):
            feats_stage2_pool=tf.tile(tf.expand_dims(feats_stage2_pool,axis=0),[tf.shape(xyzs[2])[0],1])
            lf2=tf.concat([feats_stage2_pool,feats_stage2,feats_stage2_fc],axis=1)
            lf2_up=unpool('1',lf2,vlens[1],vbegs[1],vcens[1])
            lf1=tf.concat([lf2_up,feats_stage1,feats_stage1_fc],axis=1)
            lf1_up=unpool('0',lf1,vlens[0],vbegs[0],vcens[0])
            lf0=tf.concat([lf1_up,feats_stage0,feats_stage0_fc],axis=1)

        return lf0, feats_stage0

def pointnet_14_dilated(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[0], 0.1, 0.15)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.1  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats_pn],axis=1)

            feats_stage0_pool,feats_stage0_fc=pointnet_pool(dxyzs[0],feats_stage0,[16,16],64,'pool0',vlens[0],vbegs[0],reuse)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.6)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.6  # rescale

            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats4',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats5',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[1], 0.3, 0.45)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[16,16],32,'feats6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[16,16],32,'feats7',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[16,16],32,'feats8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[24,24],48,'feats9',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[24,24],48,'feats10',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[32,32],64,'feats11',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats_pn],axis=1)

            feats_stage1_pool,feats_stage1_fc=pointnet_pool(dxyzs[1],feats_stage1,[32,32],128,'pool1',vlens[1],vbegs[1],reuse)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[32,32],64,'feats12',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[48,48],96,'feats13',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            feats_stage2_fc=mlp(feats,[64,64,128],256,'global',reuse)

            feats_stage2_pool=tf.reduce_max(feats_stage2_fc,axis=0)

        with tf.name_scope('unpool'):
            feats_stage2_pool=tf.tile(tf.expand_dims(feats_stage2_pool,axis=0),[tf.shape(xyzs[2])[0],1])
            lf2=tf.concat([feats_stage2_pool,feats_stage2,feats_stage2_fc],axis=1)
            lf2_up=unpool('1',lf2,vlens[1],vbegs[1],vcens[1])
            lf1=tf.concat([lf2_up,feats_stage1,feats_stage1_fc],axis=1)
            lf1_up=unpool('0',lf1,vlens[0],vbegs[0],vcens[0])
            lf0=tf.concat([lf1_up,feats_stage0,feats_stage0_fc],axis=1)

        return lf0, feats_stage0

def pointnet_10_concat_pre_embed(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)
            feats_pn=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.1  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=fc_embed(feats,'embed3',32,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[8,8,16],32,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats_pn],axis=1)

            feats_stage0_pool,feats_stage0_fc=pointnet_pool(dxyzs[0],feats_stage0,[16,16],64,'pool0',vlens[0],vbegs[0],reuse)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.6)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.6  # rescale

            feats_pn=fc_embed(feats,'embed4',32,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[8,8,16],32,'feats4',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=fc_embed(feats,'embed5',32,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[8,8,16],32,'feats5',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale

            feats_pn=fc_embed(feats,'embed6',48,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[16,16,24],48,'feats6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=fc_embed(feats,'embed7',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[16,16,32],64,'feats7',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats_pn],axis=1)

            feats_stage1_pool,feats_stage1_fc=pointnet_pool(dxyzs[1],feats_stage1,[32,32],128,'pool1',vlens[1],vbegs[1],reuse)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale

            feats_pn=fc_embed(feats,'embed8',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[32,32,32],64,'feats8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=fc_embed(feats,'embed9',96,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[32,32,48],96,'feats9',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            feats_stage2_fc=mlp(feats,[64,64,128],256,'global',reuse)

            feats_stage2_pool=tf.reduce_max(feats_stage2_fc,axis=0)

        with tf.name_scope('unpool'):
            feats_stage2_pool=tf.tile(tf.expand_dims(feats_stage2_pool,axis=0),[tf.shape(xyzs[2])[0],1])
            lf2=tf.concat([feats_stage2_pool,feats_stage2,feats_stage2_fc],axis=1)
            lf2_up=unpool('1',lf2,vlens[1],vbegs[1],vcens[1])
            lf1=tf.concat([lf2_up,feats_stage1,feats_stage1_fc],axis=1)
            lf1_up=unpool('0',lf1,vlens[0],vbegs[0],vcens[0])
            lf0=tf.concat([lf1_up,feats_stage0,feats_stage0_fc],axis=1)

        return lf0, feats_stage0

def pointnet_13_dilated_embed(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        feats1=graph_avg_pool(feats,vlens[0],vbegs[0],vcens[0])
        feats2=graph_avg_pool(feats1,vlens[1],vbegs[1],vcens[1])
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15) # 29
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[0], 0.15, 0.2) # 22
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[0], 0.1, 0.15) # 16
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)  # 12
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_ed=fc_embed(feats,'embed3',32,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[8,8,16],32,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats_pn],axis=1)

            voxel_stage0_pool,voxel_stage0_fc=pointnet_pool(dxyzs[0],feats_stage0,[8,8,16],32,'pool0',vlens[0],vbegs[0],reuse)
            feats_pool=graph_pool(feats_stage0,vlens[0],vbegs[0])
            feats_stage0_pool=tf.concat([feats1,feats_pool,voxel_stage0_pool],axis=1)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.45) # 30
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale
            feats_ed=fc_embed(feats,'embed4',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,32],64,'feats4',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[1], 0.45, 0.6) # 24
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale
            feats_ed=fc_embed(feats,'embed5',48,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats5',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed6',48,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[1], 0.3, 0.45) # 16
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale

            feats_ed=fc_embed(feats,'embed7',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats7',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed8',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3) # 12
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale

            feats_ed=fc_embed(feats,'embed9',96,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats9',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed10',96,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats10',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats_pn],axis=1)

            voxel_stage1_pool,voxel_stage1_fc=pointnet_pool(dxyzs[1],feats_stage1,[16,16,16],48,'pool1',vlens[1],vbegs[1],reuse)
            feats_pool=graph_pool(feats_stage1,vlens[1],vbegs[1])
            feats_stage1_pool=tf.concat([feats2,feats_pool,voxel_stage1_pool],axis=1)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale

            feats_ed=fc_embed(feats,'embed11',128,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,32],64,'feats11',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed12',128,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,32],64,'feats12',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            feats_stage2_fc=mlp(feats,[32,32,48],128,'global',reuse)

        with tf.name_scope('unpool'):
            lf2=tf.concat([feats_stage2,feats_stage2_fc],axis=1)
            lf2_up=unpool('1',lf2,vlens[1],vbegs[1],vcens[1])
            lf1=tf.concat([lf2_up,feats_stage1],axis=1)
            lf1_up=unpool('0',lf1,vlens[0],vbegs[0],vcens[0])
            lf0=tf.concat([lf1_up,feats_stage0],axis=1)

        return lf0, feats_stage0

def pointnet_13_dilated_embed_pnnoconcat(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        feats1=graph_avg_pool(feats,vlens[0],vbegs[0],vcens[0])
        feats2=graph_avg_pool(feats1,vlens[1],vbegs[1],vcens[1])
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15) # 29
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[32,32,32],32,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[0], 0.15, 0.2) # 22
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[32,32,32],32,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[0], 0.1, 0.15) # 16
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=pointnet_conv_noconcat(sxyzs,feats,[32,32,32],32,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)  # 12
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_ed=fc_embed(feats,'embed3',32,reuse)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats_ed,[32,32,32],32,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats_pn],axis=1)

            voxel_stage0_pool,voxel_stage0_fc=pointnet_pool(dxyzs[0],feats_stage0,[8,8,16],32,'pool0',vlens[0],vbegs[0],reuse)
            feats_pool=graph_pool(feats_stage0,vlens[0],vbegs[0])
            feats_stage0_pool=tf.concat([feats1,feats_pool,voxel_stage0_pool],axis=1)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.45) # 30
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale
            feats_ed=fc_embed(feats,'embed4',64,reuse)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats_ed,[64,64,64],64,'feats4',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[1], 0.45, 0.6) # 24
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale
            feats_ed=fc_embed(feats,'embed5',48,reuse)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats_ed,[48,48,48],48,'feats5',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed6',48,reuse)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats_ed,[48,48,48],48,'feats6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[1], 0.3, 0.45) # 16
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale

            feats_ed=fc_embed(feats,'embed7',64,reuse)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats_ed,[48,48,48],48,'feats7',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed8',64,reuse)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats_ed,[48,48,48],48,'feats8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3) # 12
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale

            feats_ed=fc_embed(feats,'embed9',96,reuse)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats_ed,[48,48,48],48,'feats9',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed10',96,reuse)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats_ed,[48,48,48],48,'feats10',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats_pn],axis=1)

            voxel_stage1_pool,voxel_stage1_fc=pointnet_pool(dxyzs[1],feats_stage1,[16,16,16],48,'pool1',vlens[1],vbegs[1],reuse)
            feats_pool=graph_pool(feats_stage1,vlens[1],vbegs[1])
            feats_stage1_pool=tf.concat([feats2,feats_pool,voxel_stage1_pool],axis=1)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale

            feats_ed=fc_embed(feats,'embed11',128,reuse)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats_ed,[64,64,64],64,'feats11',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed12',128,reuse)
            feats_pn=pointnet_conv_noconcat(sxyzs,feats_ed,[64,64,64],64,'feats12',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            feats_stage2_fc=mlp(feats,[32,32,48],128,'global',reuse)

        with tf.name_scope('unpool'):
            lf2=tf.concat([feats_stage2,feats_stage2_fc],axis=1)
            lf2_up=unpool('1',lf2,vlens[1],vbegs[1],vcens[1])
            lf1=tf.concat([lf2_up,feats_stage1],axis=1)
            lf1_up=unpool('0',lf1,vlens[0],vbegs[0],vcens[0])
            lf0=tf.concat([lf1_up,feats_stage0],axis=1)

        return lf0, feats_stage0

def pointnet_13_dilated_embed_feats_noconcat(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15) # 29
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats=pointnet_conv_noconcat(sxyzs,feats,[32,32,32],32,'feats0',nidxs,nlens,nbegs,ncens,reuse)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[0], 0.15, 0.2) # 22
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats=pointnet_conv_noconcat(sxyzs,feats,[32,64,64],64,'feats1',nidxs,nlens,nbegs,ncens,reuse)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[0], 0.1, 0.15) # 16
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats=pointnet_conv_noconcat(sxyzs,feats,[64,64,96],96,'feats2',nidxs,nlens,nbegs,ncens,reuse)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)  # 12
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats=pointnet_conv_noconcat(sxyzs,feats,[96,96,128],128,'feats3',nidxs,nlens,nbegs,ncens,reuse)

            voxel_stage0_pool,voxel_stage0_fc=pointnet_pool(dxyzs[0],feats,[96,128,128],160,'pool0',vlens[0],vbegs[0],reuse)

        with tf.name_scope('stage1'):
            feats=voxel_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.45) # 30
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale
            feats=pointnet_conv_noconcat(sxyzs,feats,[128,128,160],224,'feats4',nidxs,nlens,nbegs,ncens,reuse)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[1], 0.45, 0.6) # 24
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale

            feats=fc_embed(feats,'embed5',128,reuse)
            feats=pointnet_conv_noconcat(sxyzs,feats,[128,160,192],272,'feats5',nidxs,nlens,nbegs,ncens,reuse)

            feats=fc_embed(feats,'embed6',160,reuse)
            feats=pointnet_conv_noconcat(sxyzs,feats,[160,192,224],320,'feats6',nidxs,nlens,nbegs,ncens,reuse)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[1], 0.3, 0.45) # 16
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale

            feats=fc_embed(feats,'embed7',192,reuse)
            feats=pointnet_conv_noconcat(sxyzs,feats,[192,224,256],368,'feats7',nidxs,nlens,nbegs,ncens,reuse)

            feats=fc_embed(feats,'embed8',224,reuse)
            feats=pointnet_conv_noconcat(sxyzs,feats,[224,256,288],416,'feats8',nidxs,nlens,nbegs,ncens,reuse)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3) # 12
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale

            feats=fc_embed(feats,'embed9',256,reuse)
            feats=pointnet_conv_noconcat(sxyzs,feats,[256,288,320],464,'feats9',nidxs,nlens,nbegs,ncens,reuse)

            feats=fc_embed(feats,'embed10',288,reuse)
            feats=pointnet_conv_noconcat(sxyzs,feats,[288,320,352],512,'feats10',nidxs,nlens,nbegs,ncens,reuse)

            voxel_stage1_pool,voxel_stage1_fc=pointnet_pool(dxyzs[1],feats,[320,352,384],560,'pool1',vlens[1],vbegs[1],reuse)

        with tf.name_scope('stage2'):
            feats=voxel_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale

            feats=fc_embed(feats,'embed11',352,reuse)
            feats=pointnet_conv_noconcat(sxyzs,feats,[352,384,416],560,'feats11',nidxs,nlens,nbegs,ncens,reuse)

            feats=fc_embed(feats,'embed12',384,reuse)
            feats=pointnet_conv_noconcat(sxyzs,feats,[384,416,448],624,'feats12',nidxs,nlens,nbegs,ncens,reuse)

            feats=tf.concat([xyzs[2],feats],axis=1)
            feats_stage2_fc=mlp(feats,[416,448,480],752,'global',reuse)

        with tf.name_scope('unpool'):
            lf2_up=unpool('1',feats_stage2_fc,vlens[1],vbegs[1],vcens[1])
            lf1=tf.concat([lf2_up,voxel_stage1_fc],axis=1)
            lf1_up=unpool('0',lf1,vlens[0],vbegs[0],vcens[0])
            lf0=tf.concat([lf1_up,voxel_stage0_fc],axis=1)

        return lf0, voxel_stage0_fc

def pointnet_13_embed(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        feats1=graph_avg_pool(feats,vlens[0],vbegs[0],vcens[0])
        feats2=graph_avg_pool(feats1,vlens[1],vbegs[1],vcens[1])
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15) # 29
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)  # 12
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed3',32,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[8,8,16],32,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats_pn],axis=1)

            voxel_stage0_pool,voxel_stage0_fc=pointnet_pool(dxyzs[0],feats_stage0,[8,8,16],32,'pool0',vlens[0],vbegs[0],reuse)
            feats_pool=graph_pool(feats_stage0,vlens[0],vbegs[0])
            feats_stage0_pool=tf.concat([feats1,feats_pool,voxel_stage0_pool],axis=1)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.45) # 30
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale
            feats_ed=fc_embed(feats,'embed4',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,32],64,'feats4',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3) # 12
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale
            feats_ed=fc_embed(feats,'embed5',48,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats5',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed6',48,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed7',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats7',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed8',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed9',96,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats9',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed10',96,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats10',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats_pn],axis=1)

            voxel_stage1_pool,voxel_stage1_fc=pointnet_pool(dxyzs[1],feats_stage1,[16,16,16],48,'pool1',vlens[1],vbegs[1],reuse)
            feats_pool=graph_pool(feats_stage1,vlens[1],vbegs[1])
            feats_stage1_pool=tf.concat([feats2,feats_pool,voxel_stage1_pool],axis=1)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale

            feats_ed=fc_embed(feats,'embed11',128,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,32],64,'feats11',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed12',128,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,32],64,'feats12',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            feats_stage2_fc=mlp(feats,[32,32,48],128,'global',reuse)

        with tf.name_scope('unpool'):
            lf2=tf.concat([feats_stage2,feats_stage2_fc],axis=1)
            lf2_up=unpool('1',lf2,vlens[1],vbegs[1],vcens[1])
            lf1=tf.concat([lf2_up,feats_stage1],axis=1)
            lf1_up=unpool('0',lf1,vlens[0],vbegs[0],vcens[0])
            lf0=tf.concat([lf1_up,feats_stage0],axis=1)

        return lf0, feats_stage0

def pointnet_13_dilated_embed_fixed(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        feats1=graph_avg_pool(feats,vlens[0],vbegs[0],vcens[0])
        feats2=graph_avg_pool(feats1,vlens[1],vbegs[1],vcens[1])
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood_fixed(xyzs[0], 0.15, 25) # 29
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_fixed_range(xyzs[0], 0.15, 0.2, 22) # 25
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_fixed_range(xyzs[0], 0.1, 0.15 ,15) # 16
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_fixed(xyzs[0], 0.1, 10) # 12
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_ed=fc_embed(feats,'embed3',32,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[8,8,16],32,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats_pn],axis=1)

            voxel_stage0_pool,voxel_stage0_fc=pointnet_pool(dxyzs[0],feats_stage0,[8,8,16],32,'pool0',vlens[0],vbegs[0],reuse)
            feats_pool=graph_pool(feats_stage0,vlens[0],vbegs[0])
            feats_stage0_pool=tf.concat([feats1,feats_pool,voxel_stage0_pool],axis=1)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood_fixed(xyzs[1], 0.45, 30) # 30
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale
            feats_ed=fc_embed(feats,'embed4',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,32],64,'feats4',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_fixed_range(xyzs[1], 0.45, 0.6, 25) # 24
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale
            feats_ed=fc_embed(feats,'embed5',48,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats5',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed6',48,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_fixed_range(xyzs[1], 0.3, 0.45, 15) # 16
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale

            feats_ed=fc_embed(feats,'embed7',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats7',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed8',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_fixed(xyzs[1], 0.3, 10)   # 12
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale

            feats_ed=fc_embed(feats,'embed9',96,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats9',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed10',96,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats10',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats_pn],axis=1)

            voxel_stage1_pool,voxel_stage1_fc=pointnet_pool(dxyzs[1],feats_stage1,[16,16,16],48,'pool1',vlens[1],vbegs[1],reuse)
            feats_pool=graph_pool(feats_stage1,vlens[1],vbegs[1])
            feats_stage1_pool=tf.concat([feats2,feats_pool,voxel_stage1_pool],axis=1)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood_fixed(xyzs[2], 0.9, 15)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale

            feats_ed=fc_embed(feats,'embed11',128,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,32],64,'feats11',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed12',128,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,32],64,'feats12',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            feats_stage2_fc=mlp(feats,[32,32,48],128,'global',reuse)

        with tf.name_scope('unpool'):
            lf2=tf.concat([feats_stage2,feats_stage2_fc],axis=1)
            lf2_up=unpool('1',lf2,vlens[1],vbegs[1],vcens[1])
            lf1=tf.concat([lf2_up,feats_stage1],axis=1)
            lf1_up=unpool('0',lf1,vlens[0],vbegs[0],vcens[0])
            lf0=tf.concat([lf1_up,feats_stage0],axis=1)

        return lf0, feats_stage0

def pointnet_13_dilated_embed_scannet(xyzs, dxyzs, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15) # 29
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats=pointnet_conv_nofeats(sxyzs,[16,16,16],48,'feats_0',nidxs,nlens,nbegs,ncens,reuse)
            # feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            # feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[0], 0.15, 0.2) # 22
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[0], 0.1, 0.15) # 16
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)  # 12
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_ed=fc_embed(feats,'embed3',32,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[8,8,16],32,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats_pn],axis=1)

            voxel_stage0_pool,voxel_stage0_fc=pointnet_pool(dxyzs[0],feats_stage0,[8,8,16],32,'pool0',vlens[0],vbegs[0],reuse)
            feats_pool=graph_pool(feats_stage0,vlens[0],vbegs[0])
            feats_stage0_pool=tf.concat([feats_pool,voxel_stage0_pool],axis=1)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.45) # 30
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale
            feats_ed=fc_embed(feats,'embed4',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,32],64,'feats4',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[1], 0.45, 0.6) # 24
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale
            feats_ed=fc_embed(feats,'embed5',48,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats5',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed6',48,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[1], 0.3, 0.45) # 16
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale

            feats_ed=fc_embed(feats,'embed7',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats7',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed8',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3) # 12
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale

            feats_ed=fc_embed(feats,'embed9',96,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats9',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed10',96,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,16],48,'feats10',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats_pn],axis=1)

            voxel_stage1_pool,voxel_stage1_fc=pointnet_pool(dxyzs[1],feats_stage1,[16,16,16],48,'pool1',vlens[1],vbegs[1],reuse)
            feats_pool=graph_pool(feats_stage1,vlens[1],vbegs[1])
            feats_stage1_pool=tf.concat([feats_pool,voxel_stage1_pool],axis=1)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale

            feats_ed=fc_embed(feats,'embed11',128,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,32],64,'feats11',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed12',128,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_ed,[16,16,32],64,'feats12',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            feats_stage2_fc=mlp(feats,[32,32,48],128,'global',reuse)

        with tf.name_scope('unpool'):
            lf2=tf.concat([feats_stage2,feats_stage2_fc],axis=1)
            lf2_up=unpool('1',lf2,vlens[1],vbegs[1],vcens[1])
            lf1=tf.concat([lf2_up,feats_stage1],axis=1)
            lf1_up=unpool('0',lf1,vlens[0],vbegs[0],vcens[0])
            lf0=tf.concat([lf1_up,feats_stage0],axis=1)

        return lf0, feats_stage0


def mlp_anchor_conv(sxyzs, feats, ifn, weights_dims, ofn, anchor_num, name,
                    nidxs, nlens, nbegs, ncens, reuse=None, l2_norm=True):
    with tf.name_scope(name):
        # [pn]
        edge_weights_feats = neighbor_ops.neighbor_scatter(feats, nidxs, nlens, nbegs, use_diff=True)
        edge_weights_feats = tf.concat([sxyzs,edge_weights_feats],axis=1)

        for idx,fd in enumerate(weights_dims):
            cfeats=tf.contrib.layers.fully_connected(edge_weights_feats, num_outputs=fd,
                                                     scope='{}_fc_weights_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            edge_weights_feats=tf.concat([cfeats,edge_weights_feats],axis=1)

        # [en,an]
        edge_weights=tf.contrib.layers.fully_connected(edge_weights_feats, num_outputs=anchor_num,
                                                       scope='{}_fc_weights_final'.format(name),
                                                       activation_fn=None, reuse=reuse)
        if l2_norm:
            norm=tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(edge_weights),axis=1)+1e-5),axis=1)
            edge_weights/=(norm+1e-5)
            with tf.variable_scope(name):
                weights_transformer = variable_on_cpu('edge_weights_trans', [1, anchor_num], tf.ones_initializer)
                edge_weights *= weights_transformer

        # [en,ifn]
        edge_feats = neighbor_ops.neighbor_scatter(feats, nidxs, nlens, nbegs, use_diff=False)

        # weight edge feats
        weighted_edge_feats=tf.expand_dims(edge_weights,axis=2)*tf.expand_dims(edge_feats,axis=1)  # [en,an,ed]
        weighted_edge_feats=tf.reshape(weighted_edge_feats,[-1,anchor_num*ifn])                    # [en,an*ed]

        # sum to points
        weighted_point_feats=neighbor_ops.neighbor_sum_feat_gather(weighted_edge_feats,ncens,nlens,nbegs) #[pn,an*ed]

        # normalize point number
        weighted_point_feats/=tf.expand_dims(tf.cast(nlens,tf.float32),axis=1)

        output_point_feats = tf.contrib.layers.fully_connected(
            weighted_point_feats, num_outputs=ofn, scope='{}_fc_out'.format(name),
            activation_fn=tf.nn.leaky_relu, reuse=reuse)

        return output_point_feats

def pgnet_13_embed(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        feats1=graph_avg_pool(feats,vlens[0],vbegs[0],vcens[0])
        feats2=graph_avg_pool(feats1,vlens[1],vbegs[1],vcens[1])
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15) # 29
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale

            feats_pn=pointnet_conv(sxyzs,feats,[8],8,'pointnet0',nidxs,nlens,nbegs,ncens,reuse)
            feats_pn=mlp_anchor_conv(sxyzs,feats_pn,8,[32],32,9,'anchor_conv0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)  # 12
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats_pn=mlp_anchor_conv(sxyzs,feats,35,[32],32,9,'anchor_conv1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=mlp_anchor_conv(sxyzs,feats,67,[32],32,9,'anchor_conv2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed3',32,reuse)
            feats_pn=mlp_anchor_conv(sxyzs,feats_ed,32,[32],32,9,'anchor_conv3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats_pn],axis=1)

            voxel_stage0_pool,voxel_stage0_fc=pointnet_pool(dxyzs[0],feats_stage0,[8,8,16],32,'pool0',vlens[0],vbegs[0],reuse)
            feats_pool=graph_pool(feats_stage0,vlens[0],vbegs[0])
            feats_stage0_pool=tf.concat([feats1,feats_pool,voxel_stage0_pool],axis=1)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.45) # 30
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale
            feats_ed=fc_embed(feats,'embed4',64,reuse)
            feats_pn=mlp_anchor_conv(sxyzs,feats_ed,64,[64],64,9,'anchor_conv4',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3) # 12
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.45  # rescale
            feats_ed=fc_embed(feats,'embed5',48,reuse)
            feats_pn=mlp_anchor_conv(sxyzs,feats_ed,48,[48],48,9,'anchor_conv5',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed6',48,reuse)
            feats_pn=mlp_anchor_conv(sxyzs,feats_ed,48,[48],48,9,'anchor_conv6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed7',64,reuse)
            feats_pn=mlp_anchor_conv(sxyzs,feats_ed,64,[48],48,9,'anchor_conv7',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed8',64,reuse)
            feats_pn=mlp_anchor_conv(sxyzs,feats_ed,64,[48],48,9,'anchor_conv8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed9',96,reuse)
            feats_pn=mlp_anchor_conv(sxyzs,feats_ed,96,[48],48,9,'anchor_conv9',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed10',96,reuse)
            feats_pn=mlp_anchor_conv(sxyzs,feats_ed,96,[48],48,9,'anchor_conv10',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats_pn],axis=1)

            voxel_stage1_pool,voxel_stage1_fc=pointnet_pool(dxyzs[1],feats_stage1,[16,16,16],48,'pool1',vlens[1],vbegs[1],reuse)
            feats_pool=graph_pool(feats_stage1,vlens[1],vbegs[1])
            feats_stage1_pool=tf.concat([feats2,feats_pool,voxel_stage1_pool],axis=1)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale

            feats_ed=fc_embed(feats,'embed11',128,reuse)
            feats_pn=mlp_anchor_conv(sxyzs,feats_ed,128,[64],64,9,'anchor_conv11',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_ed=fc_embed(feats,'embed12',128,reuse)
            feats_pn=mlp_anchor_conv(sxyzs,feats_ed,128,[64],64,9,'anchor_conv12',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            feats_stage2_fc=mlp(feats,[32,32,48],128,'global',reuse)

        with tf.name_scope('unpool'):
            lf2=tf.concat([feats_stage2,feats_stage2_fc],axis=1)
            lf2_up=unpool('1',lf2,vlens[1],vbegs[1],vcens[1])
            lf1=tf.concat([lf2_up,feats_stage1],axis=1)
            lf1_up=unpool('0',lf1,vlens[0],vbegs[0],vcens[0])
            lf0=tf.concat([lf1_up,feats_stage0],axis=1)

        return lf0, feats_stage0
