from tf_ops.graph_layer_new import neighbor_ops, pooling_ops, graph_concat_scatter, graph_pool, \
    search_neighborhood, graph_unpool, search_neighborhood_range, graph_avg_pool

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