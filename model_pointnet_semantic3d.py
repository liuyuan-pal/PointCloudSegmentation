from model_pointnet import *


def pointnet_13_dilated_embed_semantic3d(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
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


def pointnet_10_concat_pre_embed_semantic3d(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        feats1=graph_avg_pool(feats,vlens[0],vbegs[0],vcens[0])
        feats2=graph_avg_pool(feats1,vlens[1],vbegs[1],vcens[1])

        with tf.name_scope('stage_pre'):
            nidxs_pre, nlens_pre, nbegs_pre, ncens_pre = search_neighborhood(xyzs[1], 0.6) # 16
            sxyzs_pre = neighbor_ops.neighbor_scatter(xyzs[1], nidxs_pre, nlens_pre, nbegs_pre, use_diff=True)  # [en,ifn]
            sxyzs_pre /= 0.6  # rescale

            feats_pre=pointnet_conv(sxyzs_pre,feats1,[16,16,16],32,'feats_pre',
                                    nidxs_pre, nlens_pre, nbegs_pre, ncens_pre, reuse)
            feats_pre=graph_unpool(feats_pre,vlens[0],vbegs[0],vcens[0])
            feats=tf.concat([feats_pre,feats],axis=1)

        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.3) # 16
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale

            feats_pn=fc_embed(feats,'embed0',16,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[4,4,8],16,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=fc_embed(feats,'embed1',16,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[4,4,8],16,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.2) # 8
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.2  # rescale

            feats_pn=fc_embed(feats,'embed2',32,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[8,8,16],32,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=fc_embed(feats,'embed3',32,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[8,8,16],32,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats_pn],axis=1)

            voxel_stage0_pool,voxel_stage0_fc=pointnet_pool(dxyzs[0],feats_stage0,[8,8,16],24,'pool0',vlens[0],vbegs[0],reuse)
            feats_pool=graph_pool(feats_stage0,vlens[0],vbegs[0])
            feats_stage0_pool=tf.concat([feats1,feats_pool,voxel_stage0_pool],axis=1)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            # nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.6) # 16
            # sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            # sxyzs /= 0.6  # rescale

            feats_pn=fc_embed(feats,'embed4',48,reuse)
            feats_pn=pointnet_conv(sxyzs_pre,feats_pn,[8,8,16],32,'feats4',nidxs_pre, nlens_pre, nbegs_pre, ncens_pre,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=fc_embed(feats,'embed5',48,reuse)
            feats_pn=pointnet_conv(sxyzs_pre,feats_pn,[8,8,16],32,'feats5',nidxs_pre, nlens_pre, nbegs_pre, ncens_pre,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.4) # 8
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.4  # rescale

            feats_pn=fc_embed(feats,'embed6',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[16,16,24],48,'feats6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=fc_embed(feats,'embed7',96,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[16,16,32],64,'feats7',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats_pn],axis=1)

            voxel_stage1_pool,voxel_stage1_fc=pointnet_pool(dxyzs[1],feats_stage1,[16,16,16],48,'pool1',vlens[1],vbegs[1],reuse)
            feats_pool=graph_pool(feats_stage1,vlens[1],vbegs[1])
            feats_stage1_pool=tf.concat([feats2,feats_pool,voxel_stage1_pool],axis=1)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 2.0) # 19
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 2.0  # rescale

            feats_pn=fc_embed(feats,'embed8',128,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[32,32,32],96,'feats8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=fc_embed(feats,'embed9',160,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[32,32,64],128,'feats9',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            feats_stage2_fc=mlp(feats,[32,32,64],128,'global',reuse)

        with tf.name_scope('unpool'):
            lf2=tf.concat([feats_stage2,feats_stage2_fc],axis=1)
            lf2_up=unpool('1',lf2,vlens[1],vbegs[1],vcens[1])
            lf1=tf.concat([lf2_up,feats_stage1],axis=1)
            lf1_up=unpool('0',lf1,vlens[0],vbegs[0],vcens[0])
            lf0=tf.concat([lf1_up,feats_stage0],axis=1)

        return lf0, feats_stage0

def pointnet_10_concat_embed_semantic3d(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        feats1=graph_avg_pool(feats,vlens[0],vbegs[0],vcens[0])
        feats2=graph_avg_pool(feats1,vlens[1],vbegs[1],vcens[1])

        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.3) # 22
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale

            feats_pn=fc_embed(feats,'embed0',16,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[4,4,8],16,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=fc_embed(feats,'embed1',16,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[4,4,8],16,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.25) # 16
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.25  # rescale

            feats_pn=fc_embed(feats,'embed2',32,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[8,8,16],32,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=fc_embed(feats,'embed3',32,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[8,8,16],32,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats_pn],axis=1)

            voxel_stage0_pool,voxel_stage0_fc=pointnet_pool(dxyzs[0],feats_stage0,[8,8,16],32,'pool0',vlens[0],vbegs[0],reuse)
            feats_pool=graph_pool(feats_stage0,vlens[0],vbegs[0])
            feats_stage0_pool=tf.concat([feats1,feats_pool,voxel_stage0_pool],axis=1)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 1.25) # 22
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 1.25  # rescale

            feats_pn=fc_embed(feats,'embed4',48,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[8,8,16],32,'feats4',nidxs, nlens, nbegs, ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=fc_embed(feats,'embed5',48,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[8,8,16],32,'feats5',nidxs, nlens, nbegs, ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 1.0) # 14
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 1.0  # rescale

            feats_pn=fc_embed(feats,'embed6',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[16,16,24],48,'feats6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=fc_embed(feats,'embed7',96,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[16,16,32],64,'feats7',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats_pn],axis=1)

            voxel_stage1_pool,voxel_stage1_fc=pointnet_pool(dxyzs[1],feats_stage1,[16,16,32],64,'pool1',vlens[1],vbegs[1],reuse)
            feats_pool=graph_pool(feats_stage1,vlens[1],vbegs[1])
            feats_stage1_pool=tf.concat([feats2,feats_pool,voxel_stage1_pool],axis=1)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 4.0) # 14
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 4.0  # rescale

            feats_pn=fc_embed(feats,'embed8',128,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[32,32,32],96,'feats8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=fc_embed(feats,'embed9',160,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[32,32,64],128,'feats9',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            feats_stage2_fc=mlp(feats,[32,32,64],128,'global',reuse)

        with tf.name_scope('unpool'):
            lf2=tf.concat([feats_stage2,feats_stage2_fc],axis=1)
            lf2_up=unpool('1',lf2,vlens[1],vbegs[1],vcens[1])
            lf1=tf.concat([lf2_up,feats_stage1],axis=1)
            lf1_up=unpool('0',lf1,vlens[0],vbegs[0],vcens[0])
            lf0=tf.concat([lf1_up,feats_stage0],axis=1)

        return lf0, feats_stage0


def dense_feats(xyzs,feats,lbls,idxs,nidxs,nlens,nbegs,ncens,reuse=False):
    cxyzs=tf.gather(xyzs,idxs)
    cfeats=tf.gather(feats,idxs)
    clbls=tf.gather(lbls,idxs)

    cxyzs_scatter=graph_unpool(cxyzs,nlens,nbegs,ncens)
    cfeats_scatter=graph_unpool(cfeats,nlens,nbegs,ncens)
    sxyzs=tf.gather(xyzs,nidxs)
    sfeats=tf.gather(feats,nidxs)
    dxyzs=sxyzs-cxyzs_scatter

    pfeats=tf.concat([dxyzs,cfeats_scatter,sfeats],axis=1)
    pfeats=mlp(pfeats,[16,16,16],48,'dense_feats',reuse)
    pfeats=graph_pool(pfeats,nlens,nbegs)

    cfeats=tf.concat([pfeats,cfeats],axis=1)

    return cxyzs,cfeats,clbls

def pointnet_13_dilate_embed_semantic3d(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        feats1=graph_avg_pool(feats,vlens[0],vbegs[0],vcens[0])
        feats2=graph_avg_pool(feats1,vlens[1],vbegs[1],vcens[1])

        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.3) # 22
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale
            feats_pn=fc_embed(feats,'embed0',32,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[8,8,16],32,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[0], 0.3, 0.4) # 20
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale
            feats_pn=fc_embed(feats,'embed1',32,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[8,8,16],32,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[0], 0.2, 0.3) # 16
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale
            feats_pn=fc_embed(feats,'embed2',32,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[8,8,16],32,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.2) # 18
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale
            feats_pn=fc_embed(feats,'embed3',32,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[8,8,16],32,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats_pn],axis=1)

            voxel_stage0_pool,voxel_stage0_fc=pointnet_pool(dxyzs[0],feats_stage0,[8,8,16],32,'pool0',vlens[0],vbegs[0],reuse)
            feats_pool=graph_pool(feats_stage0,vlens[0],vbegs[0])
            feats_stage0_pool=tf.concat([feats1,feats_pool,voxel_stage0_pool],axis=1)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 1.25) # 22
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 1.25  # rescale

            feats_pn=fc_embed(feats,'embed4',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[16,16,32],64,'feats4',nidxs, nlens, nbegs, ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[1], 1.25, 1.6) # 22
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 1.25  # rescale

            feats_pn=fc_embed(feats,'embed5',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[12,12,24],48,'feats5',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=fc_embed(feats,'embed6',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[12,12,24],48,'feats6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs[1], 0.9, 1.25) # 22
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 1.25  # rescale

            feats_pn=fc_embed(feats,'embed7',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[12,12,24],48,'feats7',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=fc_embed(feats,'embed8',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[12,12,24],48,'feats8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.9) # 22
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 1.25  # rescale

            feats_pn=fc_embed(feats,'embed9',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[12,12,24],48,'feats9',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=fc_embed(feats,'embed10',64,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[12,12,24],48,'feats10',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats_pn],axis=1)

            voxel_stage1_pool,voxel_stage1_fc=pointnet_pool(dxyzs[1],feats_stage1,[16,16,32],64,'pool1',vlens[1],vbegs[1],reuse)
            feats_pool=graph_pool(feats_stage1,vlens[1],vbegs[1])
            feats_stage1_pool=tf.concat([feats2,feats_pool,voxel_stage1_pool],axis=1)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 4.0) # 14
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 4.0  # rescale

            feats_pn=fc_embed(feats,'embed11',128,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[32,32,32],96,'feats11',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats_pn],axis=1)

            feats_pn=fc_embed(feats,'embed12',160,reuse)
            feats_pn=pointnet_conv(sxyzs,feats_pn,[32,32,64],128,'feats12',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            feats_stage2_fc=mlp(feats,[32,32,64],128,'global',reuse)

        with tf.name_scope('unpool'):
            lf2=tf.concat([feats_stage2,feats_stage2_fc],axis=1)
            lf2_up=unpool('1',lf2,vlens[1],vbegs[1],vcens[1])
            lf1=tf.concat([lf2_up,feats_stage1],axis=1)
            lf1_up=unpool('0',lf1,vlens[0],vbegs[0],vcens[0])
            lf0=tf.concat([lf1_up,feats_stage0],axis=1)

        return lf0, feats_stage0



def test_zero_neighborhood():
    import numpy as np
    xyzs_pl = tf.placeholder(tf.float32, [None, 3], 'xyzs')
    feats_pl = tf.placeholder(tf.float32, [None, 4], 'feats')

    nidxs, nlens, nbegs, ncens = search_neighborhood_range(xyzs_pl, 2.0, 3.0)  # 22
    sxyzs = neighbor_ops.neighbor_scatter(xyzs_pl, nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
    sxyzs /= 1.25  # rescale
    feats_pn=pointnet_conv(sxyzs,feats_pl,[12,12,24],48,'feats9',nidxs,nlens,nbegs,ncens,False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess=tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    pts=np.random.uniform(-0.5,0.5,[512,7])

    ofeats,olens,osxyzs,onidxs=sess.run([feats_pn,nlens,sxyzs,nidxs],feed_dict={xyzs_pl:pts[:,:3],feats_pl:pts[:,3:]})
    print olens
    print ofeats
    print osxyzs
    print onidxs


if __name__=="__main__":
    test_zero_neighborhood()