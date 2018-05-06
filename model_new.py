from model_pooling import *
from tf_ops.graph_layer_new import *
from model import classifier_v2

def model_template(stage_fn,xyzs, dxyzs, feats, vlens, vbegs, vcens, voxel_sizes, block_size,
                   radius=(0.15,0.3,0.5), reuse=False):
    with tf.name_scope('base_graph_conv_edge_net'):
        with tf.variable_scope('base_graph_conv_edge_net',reuse=reuse):
            with tf.name_scope('conv_stage0'):
                fc0, lf0, = stage_fn(0, xyzs[0], dxyzs[0], feats, tf.shape(feats)[1], radius=radius[0],
                                     reuse=reuse, voxel_size=voxel_sizes[0],gxyz_dim=16, gc_dims=[16],
                                     gfc_dims=[8,8,8], final_dim=64)

                fc0_pool = graph_max_pool_stage(0, fc0, vlens[0], vbegs[0])             # 64
                lf0_avg = graph_avg_pool_stage(0, lf0, vlens[0], vbegs[0], vcens[0])    # 61
                ifeats_0 = tf.concat([fc0_pool,lf0_avg],axis=1)

            with tf.name_scope('conv_stage1'):
                fc1, lf1, = stage_fn(1, xyzs[1], xyzs[1], ifeats_0, tf.shape(ifeats_0)[1], radius=radius[1],
                                     reuse=reuse, voxel_size=voxel_sizes[1],gxyz_dim=32, gc_dims=[32],
                                     gfc_dims=[32,32,32], final_dim=128)

                fc1_pool = graph_max_pool_stage(1, fc1, vlens[1], vbegs[1])         # 256
                lf1_avg = graph_avg_pool_stage(1, lf1, vlens[1], vbegs[1], vcens[1])# 429
                ifeats_1 = tf.concat([fc1_pool,lf1_avg],axis=1)                     # 685

            with tf.name_scope('conv_stage2'):
                fc2, lf2, = stage_fn(2, xyzs[2], xyzs[2], ifeats_1, tf.shape(ifeats_1)[1], radius=radius[2],
                                     reuse=reuse, voxel_size=block_size,gxyz_dim=32, gc_dims=[32],
                                     gfc_dims=[32,32,32], final_dim=256)

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
            ops=[fc0,lf0,fc1,lf1,fc2,lf2]

    return upf0, lf, ops


# diff diffusion
def diff_pgnet_shallow(xyzs, dxyzs, feats, vlens, vbegs, vcens, voxel_sizes, block_size,
                  radius=(0.15,0.3,0.5), reuse=False):
    with tf.name_scope('base_graph_conv_edge_net'):
        with tf.variable_scope('base_graph_conv_edge_net',reuse=reuse):
            with tf.name_scope('conv_stage0'):
                fc0, lf0, ops0 = graph_conv_pool_stage_edge_simp(0, xyzs[0], dxyzs[0], feats, tf.shape(feats)[1], radius=radius[0],
                                                                 reuse=reuse, voxel_size=voxel_sizes[0],
                                                                 gxyz_dim=32, gc_dims=[32],
                                                                 gfc_dims=[8,8,8], final_dim=64)
                fc0_pool = graph_max_pool_stage(0, fc0, vlens[0], vbegs[0])             # 64
                lf0_avg = graph_avg_pool_stage(0, lf0, vlens[0], vbegs[0], vcens[0])    # 61
                ifeats_0 = tf.concat([fc0_pool,lf0_avg],axis=1)

            with tf.name_scope('conv_stage1'):
                fc1, lf1, ops1 = graph_conv_pool_stage_edge_simp(1, xyzs[1], xyzs[1], ifeats_0, tf.shape(ifeats_0)[1], radius=radius[1],
                                                                 reuse=reuse, voxel_size=voxel_sizes[1],
                                                                 gxyz_dim=32, gc_dims=[32],
                                                                 gfc_dims=[32,64,64], final_dim=256)
                fc1_pool = graph_max_pool_stage(1, fc1, vlens[1], vbegs[1])         # 256
                lf1_avg = graph_avg_pool_stage(1, lf1, vlens[1], vbegs[1], vcens[1])# 429
                ifeats_1 = tf.concat([fc1_pool,lf1_avg],axis=1)                     # 685

            with tf.name_scope('conv_stage2'):
                fc2, lf2, ops2 = graph_conv_pool_stage_edge_simp(2, xyzs[2], xyzs[2], ifeats_1, tf.shape(ifeats_1)[1], radius=radius[2],
                                                                 reuse=reuse, voxel_size=block_size,
                                                                 gxyz_dim=32, gc_dims=[32],
                                                                 gfc_dims=[32,64,64], final_dim=384)
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
            ops=[fc0,lf0,fc1,lf1,fc2,lf2]

    return upf0, lf, ops


# pointnet
def pointnet_stage(stage_idx, xyzs, dxyz, feats, feats_dim, gxyz_dim, gc_dims, gfc_dims, final_dim,
                   radius, voxel_size, reuse):
    with tf.name_scope('stage_{}'.format(stage_idx)):
        nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,radius)

        sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
        sxyzs /= radius

        xyz_gc=pointnet_conv(sxyzs,xyzs,[gxyz_dim/2,gxyz_dim/2],gxyz_dim,'{}_xyz'.format(stage_idx),
                             nidxs,nlens,nbegs,ncens,reuse)
        cfeats = tf.concat([xyz_gc, feats], axis=1)

        cdim = feats_dim + gxyz_dim
        conv_fn = partial(pointnet_conv, ncens=ncens, nidxs=nidxs, nlens=nlens, nbegs=nbegs, reuse=reuse)

        layer_idx = 1
        for gd in gc_dims:

            conv_feats = tf.contrib.layers.fully_connected(cfeats, num_outputs=gd, activation_fn=tf.nn.relu, reuse=reuse,
                                                           scope='{}_{}_embed'.format(stage_idx, layer_idx))
            conv_feats = conv_fn(sxyzs, conv_feats, [gd/2,gd/2], gd, '{}_{}_gc'.format(stage_idx,layer_idx))
            cfeats = tf.concat([cfeats, conv_feats], axis=1)
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

    return fc_final, cfeats  # cfeats: [pn,fc_dims+gxyz_dim+feats_dim]


# concat diffusion
def concat_diffusion_stage(stage_idx, xyzs, dxyz, feats, feats_dim, gxyz_dim, gc_dims, gfc_dims, final_dim,
                           radius, voxel_size, reuse):
    with tf.name_scope('stage_{}'.format(stage_idx)):
        nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,radius)

        sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
        sxyzs /= radius

        xyz_gc=concat_feats_ecd(sxyzs,xyzs,3,[gxyz_dim/2,gxyz_dim/2],[gxyz_dim/2,gxyz_dim/2],gxyz_dim,
                                '{}_xyz'.format(stage_idx),nidxs,nlens,nbegs,ncens,reuse)
        cfeats = tf.concat([xyz_gc, feats], axis=1)

        cdim = feats_dim + gxyz_dim
        conv_fn = partial(concat_feats_ecd, ncens=ncens, nidxs=nidxs, nlens=nlens, nbegs=nbegs, reuse=reuse)

        layer_idx = 1
        for gd in gc_dims:
            conv_feats = tf.contrib.layers.fully_connected(cfeats, num_outputs=gd, activation_fn=tf.nn.relu, reuse=reuse,
                                                           scope='{}_{}_embed'.format(stage_idx, layer_idx))
            conv_feats = conv_fn(sxyzs, conv_feats, cdim, [gd/2,gd/2], [gd/2,gd/2], gd,
                                 '{}_{}_gc'.format(stage_idx,layer_idx))
            cfeats = tf.concat([cfeats, conv_feats], axis=1)
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

    return fc_final, cfeats  # cfeats: [pn,fc_dims+gxyz_dim+feats_dim]


# anchor convolution
def anchor_conv_stage(stage_idx, xyzs, dxyz, feats, feats_dim, gxyz_dim, gc_dims, gfc_dims, final_dim,
                      radius, voxel_size, reuse):
    with tf.name_scope('stage_{}'.format(stage_idx)):
        nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,radius)

        sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
        sxyzs /= radius

        xyz_gc=anchor_conv(sxyzs, xyzs, 3, gxyz_dim, 16, '{}_xyz'.format(stage_idx),
                           nidxs,nlens,nbegs,ncens,reuse)
        cfeats = tf.concat([xyz_gc, feats], axis=1)

        cdim = feats_dim + gxyz_dim
        conv_fn = partial(anchor_conv, ncens=ncens, nidxs=nidxs, nlens=nlens, nbegs=nbegs, reuse=reuse)

        layer_idx = 1
        for gd in gc_dims:

            conv_feats = tf.contrib.layers.fully_connected(cfeats, num_outputs=gd, activation_fn=tf.nn.relu, reuse=reuse,
                                                           scope='{}_{}_embed'.format(stage_idx, layer_idx))
            conv_feats = conv_fn(sxyzs, conv_feats, cdim, gd, 16, '{}_{}_gc'.format(stage_idx,layer_idx))
            cfeats = tf.concat([cfeats, conv_feats], axis=1)
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

    return fc_final, cfeats  # cfeats: [pn,fc_dims+gxyz_dim+feats_dim]


# edge condition convolution
def edge_conv_stage():
    pass


def edge_condition_diffusion_anchor_stage(stage_idx, xyzs, dxyz, feats, feats_dim,
                                          gc_dims, anchor_nums, embed_dims, xyz_feats_dim,
                                          gfc_dims, final_dim, radius, voxel_size, reuse):

    with tf.name_scope('stage_{}'.format(stage_idx)):
        nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,radius)

        sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
        sxyzs /= radius

        xyz_gc=pointnet_conv(sxyzs,feats,[xyz_feats_dim/2,xyz_feats_dim/2],xyz_feats_dim,'{}_xyz'.format(stage_idx),
                             nidxs,nlens,nbegs,ncens,reuse)
        cfeats = tf.concat([xyz_gc, feats], axis=1)
        cdim = feats_dim+xyz_feats_dim

        conv_fn = partial(edge_condition_diffusion_anchor_v2, ncens=ncens, nidxs=nidxs, nlens=nlens, nbegs=nbegs, reuse=reuse)

        layer_idx = 1
        for gd,an,ed in zip(gc_dims,anchor_nums,embed_dims):
            conv_feats = conv_fn(sxyzs, cfeats, [an, an*2], gd, an, ed,
                                 '{}_{}_gc'.format(stage_idx,layer_idx))
            cfeats = tf.concat([cfeats, conv_feats], axis=1)
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

    return fc_final, cfeats


def edge_condition_diffusion_anchor_model(xyzs, dxyzs, feats, vlens, vbegs, vcens, voxel_sizes, block_size,
                                          radius=(0.15,0.3,0.5), reuse=False):
    with tf.name_scope('base_graph_conv_edge_net'):
        with tf.variable_scope('base_graph_conv_edge_net',reuse=reuse):
            with tf.name_scope('conv_stage0'):
                fc0, lf0, = edge_condition_diffusion_anchor_stage(
                    0,xyzs[0],dxyzs[0],feats,tf.shape(feats)[0],
                    [32],[9],[16],32,
                    [16,16,16],128,radius[0],voxel_sizes[0],reuse
                )

                fc0_pool = graph_max_pool_stage(0, fc0, vlens[0], vbegs[0])             # 64
                # lf0_avg = graph_avg_pool_stage(0, lf0, vlens[0], vbegs[0], vcens[0])    # 61
                # ifeats_0 = tf.concat([fc0_pool,lf0_avg],axis=1)

            with tf.name_scope('conv_stage1'):
                fc1, lf1, = edge_condition_diffusion_anchor_stage(
                    1,xyzs[1],dxyzs[1],fc0_pool,tf.shape(fc0_pool)[0],
                    [48,48,48],[12,12,12],[32,32,32],64,
                    [32,32,32],256,radius[1],voxel_sizes[1],reuse
                )

                fc1_pool = graph_max_pool_stage(1, fc1, vlens[1], vbegs[1])         # 256
                # lf1_avg = graph_avg_pool_stage(1, lf1, vlens[1], vbegs[1], vcens[1])# 429
                # ifeats_1 = tf.concat([fc1_pool,lf1_avg],axis=1)                     # 685

            with tf.name_scope('conv_stage2'):
                fc2, lf2, = edge_condition_diffusion_anchor_stage(
                    2,xyzs[2],xyzs[2],fc1_pool, tf.shape(fc1_pool)[1],
                    [64,64,64],[16,16,16],[48,48,48],128,
                    [64,64,64],384,radius[2],block_size,reuse
                )

                fc2_pool = tf.reduce_max(fc2, axis=0)
                # lf2_avg = tf.reduce_mean(lf2, axis=0)
                # ifeats_2 = tf.concat([fc2_pool,lf2_avg],axis=0)

            with tf.name_scope('unpool_stage2'):
                upfeats2 = tf.tile(tf.expand_dims(fc2_pool, axis=0), [tf.shape(fc2)[0], 1])
                upf2 = tf.concat([upfeats2, fc2, lf2], axis=1)

            with tf.name_scope('unpool_stage1'):
                upfeats1 = graph_unpool_stage(1, upf2, vlens[1], vbegs[1], vcens[1])
                upf1 = tf.concat([upfeats1, fc1, lf1], axis=1)

            with tf.name_scope('unpool_stage0'):
                upfeats0 = graph_unpool_stage(0, upf1, vlens[0], vbegs[0], vcens[0])
                upf0 = tf.concat([upfeats0, fc0, lf0], axis=1)

            lf = tf.concat([fc0, lf0], axis=1)
            ops=[fc0,lf0,fc1,lf1,fc2,lf2]

    return upf0, lf, ops


def edge_condition_diffusion_anchor_model_v2(xyzs, dxyzs, feats, vlens, vbegs, vcens, voxel_sizes,
                                             block_size, radius=(0.15,0.3,0.5), reuse=False):
    with tf.name_scope('base_graph_conv_edge_net'):
        with tf.variable_scope('base_graph_conv_edge_net',reuse=reuse):
            with tf.name_scope('conv_stage0'):
                fc0, lf0, = edge_condition_diffusion_anchor_stage(
                    0,xyzs[0],dxyzs[0],feats,tf.shape(feats)[0],
                    [ 8, 8,12,12,16,16],
                    [ 4, 4, 6, 6, 9, 9],
                    [ 4, 4,12,12,16,16],
                    16,[16,16,16],128,radius[0],voxel_sizes[0],reuse
                )

                fc0_pool = graph_max_pool_stage(0, fc0, vlens[0], vbegs[0])             # 64

            with tf.name_scope('conv_stage1'):
                fc1, lf1, = edge_condition_diffusion_anchor_stage(
                    1,xyzs[1],dxyzs[1],fc0_pool,tf.shape(fc0_pool)[0],
                    [16,16,16,24,24,24,32,32,32],
                    [ 9, 9, 9, 9, 9, 9, 9, 9, 9],
                    [16,16,16,24,24,24,32,32,32],
                    32,[32,32,32],256,radius[1],voxel_sizes[1],reuse
                )

                fc1_pool = graph_max_pool_stage(1, fc1, vlens[1], vbegs[1])         # 256

            with tf.name_scope('conv_stage2'):
                fc2, lf2, = edge_condition_diffusion_anchor_stage(
                    2,xyzs[2],xyzs[2],fc1_pool, tf.shape(fc1_pool)[1],
                    [32,32,32,32,32,32,32,32,32],
                    [ 9, 9, 9, 9, 9, 9, 9, 9, 9],
                    [32,32,32,40,40,40,48,48,48],
                    64,[64,64,64],384,radius[2],block_size,reuse
                )

                fc2_pool = tf.reduce_max(fc2, axis=0)

            with tf.name_scope('unpool_stage2'):
                upfeats2 = tf.tile(tf.expand_dims(fc2_pool, axis=0), [tf.shape(fc2)[0], 1])
                upf2 = tf.concat([upfeats2, fc2, lf2], axis=1)

            with tf.name_scope('unpool_stage1'):
                upfeats1 = graph_unpool_stage(1, upf2, vlens[1], vbegs[1], vcens[1])
                upf1 = tf.concat([upfeats1, fc1, lf1], axis=1)

            with tf.name_scope('unpool_stage0'):
                upfeats0 = graph_unpool_stage(0, upf1, vlens[0], vbegs[0], vcens[0])
                upf0 = tf.concat([upfeats0, fc0, lf0], axis=1)

            lf = tf.concat([fc0, lf0], axis=1)
            ops=[fc0,lf0,fc1,lf1,fc2,lf2]

    return upf0, lf, ops


def ecd_nse_stage(stage_idx, xyzs, feats, feats_dim, xyz_feats_dim, gc_dims, anchor_nums, embed_dims, radius, reuse):
    with tf.name_scope('stage_{}'.format(stage_idx)):
        nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,radius)
        sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)
        sxyzs /= radius

        xyz_gc=pointnet_conv(sxyzs,feats,[xyz_feats_dim/2,xyz_feats_dim/2],xyz_feats_dim,'{}_xyz'.format(stage_idx),
                             nidxs,nlens,nbegs,ncens,reuse)
        cfeats = tf.concat([xyz_gc, feats], axis=1)
        cdim = feats_dim+xyz_feats_dim

        conv_fn = partial(edge_condition_diffusion_anchor_v2, ncens=ncens, nidxs=nidxs, nlens=nlens, nbegs=nbegs, reuse=reuse)

        layer_idx = 1
        for gd,an,ed in zip(gc_dims,anchor_nums,embed_dims):
            conv_feats = conv_fn(sxyzs, cfeats, [an, an*2], gd, an, ed,
                                 '{}_{}_gc'.format(stage_idx,layer_idx))
            cfeats = tf.concat([cfeats, conv_feats], axis=1)
            layer_idx += 1
            cdim += gd

    return cfeats


def vanilla_pointnet(xyzs,feats,fc_dims,final_dim,name,reuse):
    feats=tf.concat([xyzs,feats],axis=1)
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        for i,fd in enumerate(fc_dims):
            cfeats = tf.contrib.layers.fully_connected(feats, num_outputs=fd,
                                                       scope='{}_{}_fc'.format(name, i))
            feats=tf.concat([feats,cfeats],axis=1)

        feats = tf.contrib.layers.fully_connected(feats, num_outputs=final_dim, activation_fn=None,
                                                  scope='{}_final_fc'.format(name))

    return feats




def pgnet_nse(xyzs, feats, vlens, vbegs, vcens, radius, reuse=False):
    with tf.name_scope('base_graph_conv_edge_net'):
        with tf.variable_scope('base_graph_conv_edge_net',reuse=reuse):

            with tf.name_scope('conv_stage0'):
                feats0 = ecd_nse_stage(0,xyzs[0],feats,tf.shape(feats)[1],
                                       32,[32],[9],[12],
                                       radius[0],reuse)
                feats0_pool = graph_max_pool_stage(0, feats0, vlens[0], vbegs[0])

            with tf.name_scope('conv_stage1'):
                feats1 = ecd_nse_stage(1,xyzs[1],feats0_pool,tf.shape(feats0_pool)[1],
                                       64,[48,48],[12,12],[16,16],
                                       radius[1],reuse)
                feats1_pool = graph_max_pool_stage(1, feats1, vlens[1], vbegs[1])

            with tf.name_scope('conv_stage2'):
                feats2 = ecd_nse_stage(2,xyzs[2],feats1_pool,tf.shape(feats1_pool)[1],
                                       96,[96,96,96],[16,16,16],[32,32,32],
                                       radius[2],reuse)
                feats2_pool = graph_max_pool_stage(2, feats2, vlens[2], vbegs[2])

            with tf.name_scope('conv_stage3'):
                feats3 = ecd_nse_stage(3,xyzs[3],feats2_pool,tf.shape(feats2_pool)[1],
                                       128,[128,128,128],[16,16,16],[64,64,64],
                                       radius[3],reuse)
                pointnet_feats3 = vanilla_pointnet(xyzs[3],feats3,[384,384],512,'global_embed',reuse)
                feat3_pool = tf.reduce_max(pointnet_feats3,axis=0)

            with tf.name_scope('unpool_stage3'):
                up3 = tf.tile(tf.expand_dims(feat3_pool, axis=0), [tf.shape(feats3)[0], 1])
                up3 = tf.concat([feats3,pointnet_feats3,up3],axis=1)

            with tf.name_scope('unpool_stage2'):
                up2 = graph_unpool_stage(2, up3, vlens[2], vbegs[2], vcens[2])
                up2 = tf.concat([feats2,up2],axis=1)

            with tf.name_scope('unpool_stage1'):
                up1 = graph_unpool_stage(1, up2, vlens[1], vbegs[1], vcens[1])
                up1 = tf.concat([feats1,up1],axis=1)

            with tf.name_scope('unpool_stage0'):
                up0 = graph_unpool_stage(0, up1, vlens[0], vbegs[0], vcens[0])
                up0 = tf.concat([feats0, up0],axis=1)

            return up0


def pgnet_nse_whole(xyzs,feats,labels,reuse,is_training,num_classes):

    with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
        xyzs, dxyzs, feats, labels, vlens, vbegs, vcens = \
            points_pooling_three_layers(xyzs,feats,labels,voxel_size1=0.1,voxel_size2=0.2,voxel_size3=0.4,block_size=3.0)
        graph_feats=pgnet_nse(xyzs,feats,vlens,vbegs,vcens,radius=[0.1,0.2,0.4,0.8],reuse=reuse)
        graph_feats=tf.expand_dims(graph_feats,axis=1)

        logits=classifier_v2(graph_feats, is_training, num_classes, reuse, use_bn=False)  # [1,pn,num_classes]

    return logits,xyzs,labels

