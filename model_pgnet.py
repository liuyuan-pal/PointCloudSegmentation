from model_new import *


def ecd_feats(edge_coord, point_feats, input_dim, phi_dims, g_dims, doutput_dim,
              nidxs, nlens, nbegs, ncens, name, reuse=None):
    with tf.name_scope(name):
        diff_edge_feats = neighbor_ops.neighbor_scatter(point_feats, nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
        phi_edge_feats = tf.concat([diff_edge_feats, edge_coord], axis=1)

        for idx,fd in enumerate(phi_dims):
            phi_out_feats = tf.contrib.layers.fully_connected(phi_edge_feats, num_outputs=fd, scope='{}_ifc_{}'.format(name,idx),
                                                              activation_fn=tf.nn.relu, reuse=reuse)
            phi_edge_feats=tf.concat([phi_out_feats,phi_edge_feats],axis=1)

        edge_weights = tf.contrib.layers.fully_connected(phi_edge_feats, num_outputs=input_dim, scope='{}_fc_ew'.format(name),
                                                         activation_fn=tf.nn.tanh, reuse=reuse)

        edge_feats=neighbor_ops.neighbor_scatter(point_feats, nidxs, nlens, nbegs, use_diff=False)      # [en,ifn]

        edge_feats= edge_weights * edge_feats
        for idx,fd in enumerate(g_dims):
            g_out_feats=tf.contrib.layers.fully_connected(edge_feats, num_outputs=fd, scope='{}_ofc_{}'.format(name, idx),
                                                           activation_fn=tf.nn.relu, reuse=reuse)
            edge_feats=tf.concat([g_out_feats, edge_feats], axis=1)

        eps=1e-3
        neighbor_size_inv = tf.expand_dims((1.0+eps) / (tf.cast(nlens, tf.float32) + eps), axis=1)                 # [pn]
        point_feats= neighbor_size_inv * neighbor_ops.neighbor_sum_feat_gather(edge_feats, ncens, nlens, nbegs)  # [pn,ofn]
        point_feats=tf.contrib.layers.fully_connected(point_feats, num_outputs=doutput_dim, scope='{}_fc_out'.format(name),
                                                      activation_fn=tf.nn.relu, reuse=reuse)

        return point_feats


def ecd_xyz(edge_coorid, phi_dims, g_dims, out_dim, nlens, nbegs, ncens, name, reuse=None):
    with tf.name_scope(name):
        phi_edge_feats=edge_coorid
        dim_sum=3
        for idx,fd in enumerate(phi_dims):
            phi_out_feats=tf.contrib.layers.fully_connected(
                phi_edge_feats, num_outputs=fd,scope='{}_ifc_{}'.format(name,idx),activation_fn=tf.nn.relu, reuse=reuse)

            phi_edge_feats=tf.concat([phi_out_feats,phi_edge_feats],axis=1)
            dim_sum+=fd

        edge_weights=tf.contrib.layers.fully_connected(phi_edge_feats, num_outputs=dim_sum, scope='{}_fc_ew'.format(name),
                                                       activation_fn=tf.nn.tanh, reuse=reuse)

        edge_feats=edge_weights*phi_edge_feats
        for idx,fd in enumerate(g_dims):
            g_out_feats=tf.contrib.layers.fully_connected(edge_feats, num_outputs=fd, scope='{}_ofc_{}'.format(name,idx),
                                                          activation_fn=tf.nn.relu, reuse=reuse)
            edge_feats=tf.concat([g_out_feats,edge_feats],axis=1)

        eps=1e-3
        neighbor_size_inv=tf.expand_dims((1.0+eps) / (tf.cast(nlens, tf.float32) + eps), axis=1)              # [pn]
        point_feats=neighbor_size_inv*neighbor_ops.neighbor_sum_feat_gather(edge_feats, ncens, nlens, nbegs)  # [pn,ofn]

        point_feats=tf.contrib.layers.fully_connected(point_feats, num_outputs=out_dim, scope='{}_fc_out'.format(name),
                                                      activation_fn=tf.nn.relu, reuse=reuse)

        return point_feats


def ecd_stage(stage_idx, xyzs, dxyz, feats,
              xyz_dim, feats_dims, embed_dims, final_dim,
              radius, voxel_size, reuse, name=''):
    with tf.name_scope('stage_{}'.format(stage_idx)):
        nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,radius)

        sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
        sxyzs /= radius   # rescale

        xyz_phi_dims=[xyz_dim/2,xyz_dim/2]
        xyz_g_dims=[xyz_dim/2,xyz_dim/2]
        xyz_gc=ecd_xyz(sxyzs,xyz_phi_dims,xyz_g_dims,xyz_dim,nlens,nbegs,ncens,'{}{}_xyz_gc'.format(name,stage_idx),reuse)
        cfeats = tf.concat([xyz_gc, feats], axis=1)

        # cdim = feats_dim + xyz_dim
        layer_idx = 1
        for fdim in feats_dims:
            conv_feats=tf.contrib.layers.fully_connected(
                cfeats, num_outputs=fdim, scope='{}{}_{}_fc'.format(name, stage_idx, layer_idx),
                activation_fn=tf.nn.relu, reuse=reuse)
            conv_feats = ecd_feats(
                sxyzs, conv_feats, fdim, [fdim/2,fdim/2], [fdim/2,fdim/2], fdim, nidxs, nlens, nbegs, ncens,
                '{}{}_{}_gc'.format(name,stage_idx,layer_idx), reuse)

            cfeats = tf.concat([cfeats, conv_feats], axis=1)
            layer_idx += 1
            # cdim += fdim

        with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
            with tf.name_scope('fc_global{}'.format(stage_idx)):
                dxyz= dxyz / voxel_size
                fc_feats = tf.concat([cfeats, dxyz], axis=1)
                for i, gfd in enumerate(embed_dims):
                    fc = tf.contrib.layers.fully_connected(fc_feats, num_outputs=gfd,
                                                           scope='{}{}_{}_gfc'.format(name,stage_idx, i))
                    fc_feats=tf.concat([fc,fc_feats],axis=1)

                fc_final = tf.contrib.layers.fully_connected(fc_feats, num_outputs=final_dim, activation_fn=None,
                                                             scope='{}{}_final_gfc'.format(name,stage_idx))

    return fc_final, cfeats


def pgnet_model_v3_bug(xyzs, dxyzs, feats, vlens, vbegs, vcens, voxel_sizes, block_size, radius=(0.15,0.3,0.5), reuse=False):
    with tf.name_scope('base_graph_conv_edge_net'):
        with tf.variable_scope('base_graph_conv_edge_net',reuse=reuse):
            with tf.name_scope('conv_stage0'):
                fc0, lf0 = ecd_stage(0,xyzs[0],dxyzs[0],feats,
                                     16,[16,16],[8,8,8],64,
                                     radius[0],voxel_sizes[0],reuse)

                fc0_pool = graph_max_pool_stage(0, fc0, vlens[0], vbegs[0])             # 64
                lf0_avg = graph_avg_pool_stage(0, lf0, vlens[0], vbegs[0], vcens[0])    # 61
                ifeats_0 = tf.concat([fc0_pool,lf0_avg],axis=1)

            with tf.name_scope('conv_stage1'):
                fc1, lf1 = ecd_stage(1,xyzs[1],xyzs[1],ifeats_0,    # !!! bug dxyzs to xyzs
                                     16,[32,32,32,32,32,32,32,32,32],[32,32,32],256,
                                     radius[1],voxel_sizes[1],reuse)
                fc1_pool = graph_max_pool_stage(1, fc1, vlens[1], vbegs[1])         # 256
                lf1_avg = graph_avg_pool_stage(1, lf1, vlens[1], vbegs[1], vcens[1])# 429
                ifeats_1 = tf.concat([fc1_pool,lf1_avg],axis=1)                     # 685

            with tf.name_scope('conv_stage2'):
                fc2, lf2 = ecd_stage(2, xyzs[2],xyzs[2], ifeats_1,
                                     16,[32,32,32,32,32,32,32,32,32],[32,32,32],512,
                                     radius[2],block_size,reuse)
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


def pgnet_model_v3(xyzs, dxyzs, feats, vlens, vbegs, vcens, voxel_sizes, block_size, radius=(0.15,0.3,0.5), reuse=False):
    with tf.name_scope('base_graph_conv_edge_net'):
        with tf.variable_scope('base_graph_conv_edge_net',reuse=reuse):
            with tf.name_scope('conv_stage0'):
                fc0, lf0 = ecd_stage(0,xyzs[0],dxyzs[0],feats,
                                     16,[16,16],[8,8,8],64,
                                     radius[0],voxel_sizes[0],reuse)

                fc0_pool = graph_max_pool_stage(0, fc0, vlens[0], vbegs[0])             # 64
                lf0_avg = graph_avg_pool_stage(0, lf0, vlens[0], vbegs[0], vcens[0])    # 61
                ifeats_0 = tf.concat([fc0_pool,lf0_avg],axis=1)

            with tf.name_scope('conv_stage1'):
                fc1, lf1 = ecd_stage(1,xyzs[1],dxyzs[1],ifeats_0,    # !!! bug dxyzs to xyzs
                                     16,[32,32,32,32,32,32,32,32,32],[32,32,32],256,
                                     radius[1],voxel_sizes[1],reuse)
                fc1_pool = graph_max_pool_stage(1, fc1, vlens[1], vbegs[1])         # 256
                lf1_avg = graph_avg_pool_stage(1, lf1, vlens[1], vbegs[1], vcens[1])# 429
                ifeats_1 = tf.concat([fc1_pool,lf1_avg],axis=1)                     # 685

            with tf.name_scope('conv_stage2'):
                fc2, lf2 = ecd_stage(2, xyzs[2],xyzs[2], ifeats_1,
                                     16,[32,32,32,32,32,32,32,32,32],[32,32,32],512,
                                     radius[2],block_size,reuse)
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


def pgnet_model_v4(xyzs, dxyzs, feats, vlens, vbegs, vcens, voxel_sizes, block_size, radius=(0.15,0.3,0.5), reuse=False):
    with tf.name_scope('base_graph_conv_edge_net'):
        with tf.variable_scope('base_graph_conv_edge_net',reuse=reuse):
            with tf.name_scope('conv_stage0'):
                fc0, lf0 = ecd_stage(0,xyzs[0],dxyzs[0],feats,
                                     16,[8,8,8,8],[8,8,8],64,
                                     radius[0],voxel_sizes[0],reuse)

                fc0_pool = graph_max_pool_stage(0, fc0, vlens[0], vbegs[0])             # 64
                lf0_avg = graph_avg_pool_stage(0, lf0, vlens[0], vbegs[0], vcens[0])    # 61
                ifeats_0 = tf.concat([fc0_pool,lf0_avg],axis=1)

            with tf.name_scope('conv_stage1'):
                fc1, lf1 = ecd_stage(1,xyzs[1],dxyzs[1],ifeats_0,
                                     16,
                                     [16,16,16,
                                     16,16,16,
                                     16,16,16,
                                     16,16,16,
                                     16,16,16,
                                     16,16,16],
                                     [16,16,16,
                                      16,16,16],256,
                                     radius[1],voxel_sizes[1],reuse)
                fc1_pool = graph_max_pool_stage(1, fc1, vlens[1], vbegs[1])         # 256
                lf1_avg = graph_avg_pool_stage(1, lf1, vlens[1], vbegs[1], vcens[1])# 429
                ifeats_1 = tf.concat([fc1_pool,lf1_avg],axis=1)                     # 685

            with tf.name_scope('conv_stage2'):
                fc2, lf2 = ecd_stage(2, xyzs[2],xyzs[2], ifeats_1,
                                     16,
                                     [16,16,16,
                                     16,16,16,
                                     16,16,16,
                                     16,16,16,
                                     16,16,16,
                                     16,16,16],
                                     [16,16,16,
                                      16,16,16],512,
                                     radius[2],block_size,reuse)
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


def pgnet_model_v5(xyzs, dxyzs, feats, vlens, vbegs, vcens, voxel_sizes, block_size, radius=(0.15,0.3,0.5), reuse=False):
    with tf.name_scope('base_graph_conv_edge_net'):
        with tf.variable_scope('base_graph_conv_edge_net',reuse=reuse):
            with tf.name_scope('conv_stage0'):
                fc0, lf0 = ecd_stage(0,xyzs[0],dxyzs[0],feats,
                                     16,[16],[8,8,8],64,
                                     radius[0],voxel_sizes[0],reuse)

                fc0_pool = graph_max_pool_stage(0, fc0, vlens[0], vbegs[0])             # 64
                lf0_avg = graph_avg_pool_stage(0, lf0, vlens[0], vbegs[0], vcens[0])    # 61
                ifeats_0 = tf.concat([fc0_pool,lf0_avg],axis=1)

            with tf.name_scope('conv_stage1'):
                fc1, lf1 = ecd_stage(1,xyzs[1],dxyzs[1],ifeats_0,
                                     16,[32,32,32],[32,32,32],256,
                                     radius[1],voxel_sizes[1],reuse)
                fc1_pool = graph_max_pool_stage(1, fc1, vlens[1], vbegs[1])         # 256
                lf1_avg = graph_avg_pool_stage(1, lf1, vlens[1], vbegs[1], vcens[1])# 429
                ifeats_1 = tf.concat([fc1_pool,lf1_avg],axis=1)                     # 685

            with tf.name_scope('conv_stage2'):
                fc2, lf2 = ecd_stage(2, xyzs[2],xyzs[2], ifeats_1,
                                     16,[32,32,32],[32,32,32],512,
                                     radius[2],block_size,reuse)
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


def inter_layer_feats(dxyzs, feats, vlens, vbegs, vcens,
                      feats_dims, diffusion_dims, embed_dims,
                      name='', reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        # mean feats
        mfeats=neighbor_ops.neighbor_sum_feat_gather(feats,vcens,vlens,vbegs) # [vn,3]
        mfeats=mfeats/tf.expand_dims(tf.cast(vlens,tf.float32),axis=1)

        # dfeats
        upmfeats = graph_unpool_stage('{}_mean_feats'.format(name), mfeats, vlens, vbegs, vcens)
        dfeats=feats-upmfeats

        # feats
        points_feats=tf.concat([dxyzs,feats],axis=1)
        for fi,fd in enumerate(feats_dims[:-1]):
            fc_feats = tf.contrib.layers.fully_connected(points_feats, num_outputs=fd,
                                                         scope='{}{}_feats_fc'.format(name,fi))
            points_feats=tf.concat([points_feats,fc_feats],axis=1)
        points_feats = tf.contrib.layers.fully_connected(points_feats, num_outputs=feats_dims[-1],activation_fn=None,
                                                         scope='{}final_feats_fc'.format(name))

        # diffusion
        diff_feats=tf.concat([dxyzs,dfeats],axis=1)
        for di,dd in enumerate(diffusion_dims):
            fc_feats = tf.contrib.layers.fully_connected(diff_feats, num_outputs=dd,
                                                         scope='{}{}_diffusion_fc'.format(name,di))
            diff_feats=tf.concat([diff_feats,fc_feats],axis=1)
        diff_weights = tf.contrib.layers.fully_connected(diff_feats, num_outputs=feats_dims[-1],activation_fn=tf.nn.tanh,
                                                         scope='{}final_diffusion_fc'.format(name))

        weighted_points_feats=diff_weights*points_feats

        # embed
        embed_points_feats=weighted_points_feats
        for ei,ed in enumerate(embed_dims[:-1]):
            fc_feats = tf.contrib.layers.fully_connected(embed_points_feats, num_outputs=ed,
                                                         scope='{}{}_embed_fc'.format(name,ei))
            embed_points_feats=tf.concat([embed_points_feats,fc_feats],axis=1)
        embed_points_feats = tf.contrib.layers.fully_connected(embed_points_feats, num_outputs=embed_dims[-1],
                                                               scope='{}final_embed_fc'.format(name))

        eps=1e-3
        neighbor_size_inv=tf.expand_dims((1.0+eps) / (tf.cast(vlens, tf.float32) + eps), axis=1)
        voxel_feats=neighbor_size_inv*neighbor_ops.neighbor_sum_feat_gather(
            embed_points_feats, vcens, vlens, vbegs)

        voxel_feats=tf.concat([mfeats,voxel_feats],axis=1)
        points_feats=tf.concat([points_feats,feats],axis=1)
        return points_feats,voxel_feats


def ecd_xyz_v2(dxyzs, feats_dims, final_feats_dim, diffusion_dims, trans_dims, out_dim,
               nlens, nbegs, ncens, name, is_training, reuse=None):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope(name):
            # extract feats
            edge_feats = dxyzs
            for fi,fd in enumerate(feats_dims):
                fc_feats=tf.contrib.layers.fully_connected(edge_feats,num_outputs=fd,scope='{}{}_feats_fc'.format(name,fi))
                edge_feats=tf.concat([edge_feats,fc_feats],axis=1)
            edge_feats = tf.contrib.layers.fully_connected(edge_feats, num_outputs=final_feats_dim,activation_fn=None,
                                                           scope='{}final_feats_fc'.format(name))
            # edge_feats = tf.contrib.layers.batch_norm(edge_feats,scale=True,is_training=is_training,
            #                                           scope='{}_feats_bn'.format(name))

            # diffusion
            edge_diff_feats = dxyzs
            for di,dd in enumerate(diffusion_dims):
                fc_feats = tf.contrib.layers.fully_connected(edge_diff_feats, num_outputs=dd,
                                                             scope='{}{}_diffusion_fc'.format(name,di))
                edge_diff_feats=tf.concat([edge_diff_feats,fc_feats],axis=1)
            edge_weights = tf.contrib.layers.fully_connected(edge_diff_feats, num_outputs=final_feats_dim,
                                                             activation_fn=tf.nn.tanh,
                                                             scope='{}final_diffusion_fc'.format(name))
            # trans
            edge_feats=edge_weights*edge_feats
            # edge_feats = tf.contrib.layers.batch_norm(edge_feats,scale=True,is_training=is_training,
            #                                           scope='{}_diffusion_bn'.format(name))
            for ti,td in enumerate(trans_dims):
                fc_feats=tf.contrib.layers.fully_connected(edge_feats, num_outputs=td,
                                                           scope='{}{}_embed_fc'.format(name,ti),
                                                           activation_fn=tf.nn.relu, reuse=reuse)
                edge_feats=tf.concat([edge_feats,fc_feats],axis=1)

            # divide by neighbor size
            eps=1e-3
            neighbor_size_inv=tf.expand_dims((1.0+eps) / (tf.cast(nlens, tf.float32) + eps), axis=1)
            point_feats=neighbor_size_inv*neighbor_ops.neighbor_sum_feat_gather(edge_feats, ncens, nlens, nbegs)

            point_feats=tf.contrib.layers.fully_connected(point_feats, num_outputs=out_dim, scope='{}out_embed_fc'.format(name),
                                                          activation_fn=tf.nn.relu, reuse=reuse)
            point_feats = tf.contrib.layers.batch_norm(point_feats,scale=True,is_training=is_training,
                                                       scope='{}_out_bn'.format(name),decay=0.9)

            return point_feats


def ecd_feats_v2(dxyzs, feats, embed_dim, diffusion_dims, trans_dims, out_dim,
                 nidxs, nlens, nbegs, ncens, name, is_training, reuse=None):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope(name):
            # embed lowering dimensions
            feats=tf.contrib.layers.fully_connected(feats, num_outputs=embed_dim, scope='{}in_embed_fc'.format(name),
                                                    activation_fn=None, reuse=reuse)
            # feats = tf.contrib.layers.batch_norm(feats,scale=True,is_training=is_training,
            #                                      scope='{}_embed_bn'.format(name))

            # diffusion
            edge_diff_feats = neighbor_ops.neighbor_scatter(feats, nidxs, nlens, nbegs, use_diff=True)
            edge_diff_feats = tf.concat([edge_diff_feats,dxyzs],axis=1)
            for di,dd in enumerate(diffusion_dims):
                fc_feats = tf.contrib.layers.fully_connected(edge_diff_feats, num_outputs=dd,
                                                             scope='{}{}_diffusion_fc'.format(name,di))
                edge_diff_feats=tf.concat([edge_diff_feats,fc_feats],axis=1)
            edge_weights = tf.contrib.layers.fully_connected(edge_diff_feats, num_outputs=embed_dim,
                                                             activation_fn=tf.nn.tanh,
                                                             scope='{}final_diffusion_fc'.format(name))
            # trans
            edge_feats = neighbor_ops.neighbor_scatter(feats, nidxs, nlens, nbegs, use_diff=False)
            edge_feats = edge_weights * edge_feats

            # edge_feats = tf.contrib.layers.batch_norm(edge_feats,scale=True,is_training=is_training,
            #                                           scope='{}_diffused_bn'.format(name))

            for ti,td in enumerate(trans_dims):
                fc_feats=tf.contrib.layers.fully_connected(edge_feats, num_outputs=td,
                                                           scope='{}{}_embed_fc'.format(name,ti),
                                                           activation_fn=tf.nn.relu, reuse=reuse)
                edge_feats=tf.concat([edge_feats,fc_feats],axis=1)

            # divide by neighbor size
            eps=1e-3
            neighbor_size_inv=tf.expand_dims((1.0+eps) / (tf.cast(nlens, tf.float32) + eps), axis=1)
            point_feats=neighbor_size_inv*neighbor_ops.neighbor_sum_feat_gather(edge_feats, ncens, nlens, nbegs)

            point_feats=tf.contrib.layers.fully_connected(point_feats, num_outputs=out_dim, scope='{}out_embed_fc'.format(name),
                                                          activation_fn=tf.nn.relu, reuse=reuse)
            point_feats = tf.contrib.layers.batch_norm(point_feats,scale=True,is_training=is_training,
                                                       scope='{}_out_bn'.format(name),decay=0.9)

            return point_feats


def ecd_stage_v2(stage_idx, xyzs, dxyzs, feats, xyz_param, feats_params, embed_dims, final_dim,
                 radius, sxyz_scale, dxyz_scale, is_training, reuse):
    with tf.name_scope('stage_{}'.format(stage_idx)):
        nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,radius)
        sxyzs=neighbor_ops.neighbor_scatter(xyzs,nidxs,nlens,nbegs,use_diff=True)
        sxyzs*=sxyz_scale
        xyz_feats=ecd_xyz_v2(sxyzs, xyz_param[0], xyz_param[1], xyz_param[2], xyz_param[3], xyz_param[4],
                             nlens, nbegs, ncens, '{}_xyz'.format(stage_idx),is_training, reuse)
        cfeats=tf.concat([feats,xyz_feats],axis=1)
        for fi,fp in enumerate(feats_params):
            ecd_feats_val=ecd_feats_v2(sxyzs,cfeats,fp[0],fp[1],fp[2],fp[3],nidxs,nlens,nbegs,ncens,
                                       '{}_{}_feats'.format(stage_idx,fi),is_training,reuse)
            cfeats=tf.concat([cfeats,ecd_feats_val],axis=1)

        with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
            with tf.name_scope('{}_global'.format(stage_idx)):
                dxyzs*=dxyz_scale
                fc_feats = tf.concat([cfeats, dxyzs], axis=1)
                for i, gfd in enumerate(embed_dims):
                    fc = tf.contrib.layers.fully_connected(fc_feats, num_outputs=gfd,
                                                           scope='{}_{}_global'.format(stage_idx, i))
                    fc_feats=tf.concat([fc,fc_feats],axis=1)

                fc_final = tf.contrib.layers.fully_connected(fc_feats, num_outputs=final_dim, activation_fn=None,
                                                             scope='{}_final_global'.format(stage_idx))


        return cfeats,fc_final


def pgnet_model_v6(xyzs, dxyzs, feats, vlens, vbegs, vcens, is_training, radius=(0.15,0.3,0.5), reuse=False):
    with tf.name_scope('base_graph_conv_edge_net'):
        with tf.variable_scope('base_graph_conv_edge_net',reuse=reuse):
            with tf.name_scope('conv_stage0'):
                fc0,lf0=ecd_stage_v2(0,xyzs[0],dxyzs[0],feats,
                                     # feats_dims, final_feats_dim, diffusion_dims, trans_dims, out_dim
                                     [[8,8],16,[8,8],[8,8],32],
                                     [
                                         # embed_dim, diffusion_dims, trans_dims, out_dim
                                         [16,[8,8],[8,8],32],
                                         [16,[8,8],[8,8],32],
                                     ],
                                     [16,16,16], 128,radius[0],3.0/0.15,3.0/0.15,is_training,reuse)

                fc0_pool = graph_max_pool_stage(0, fc0, vlens[0], vbegs[0])
                lf0_avg = graph_avg_pool_stage(0, feats, vlens[0], vbegs[0], vcens[0])
                ifeats_0 = tf.concat([lf0_avg, fc0_pool], axis=1)

            with tf.name_scope('conv_stage1'):
                fc1,lf1=ecd_stage_v2(1, xyzs[1], dxyzs[1], ifeats_0,
                                     # feats_dims, final_feats_dim, diffusion_dims, trans_dims, out_dim
                                     [[16,16],32,[16,16],[16,16],32],
                                     [
                                         # embed_dim, diffusion_dims, trans_dims, out_dim
                                         [32,[16,16],[16,16],32],
                                         [32,[16,16],[16,16],32],
                                         [32,[16,16],[16,16],32],
                                     ],
                                     [32,32,32], 256,radius[1],3.0/0.3,3.0/0.45,is_training,reuse)

                fc1_pool = graph_max_pool_stage(1, fc1, vlens[1], vbegs[1])
                lf1_avg = graph_avg_pool_stage(1, lf0_avg, vlens[1], vbegs[1], vcens[1])
                ifeats_1 = tf.concat([fc1_pool,lf1_avg],axis=1)

            with tf.name_scope('conv_stage2'):
                fc2,lf2=ecd_stage_v2(2, xyzs[2], xyzs[2], ifeats_1,
                                     # feats_dims, final_feats_dim, diffusion_dims, trans_dims, out_dim
                                     [[16,16],32,[16,16],[16,16],32],
                                     [
                                         # embed_dim, diffusion_dims, trans_dims, out_dim
                                         [48,[16,16],[16,16],48],
                                         [48,[16,16],[16,16],48],
                                         [48,[16,16],[16,16],48],
                                     ],
                                     [64,64,64,128], 512,radius[2],3.0/0.9,3.0/3.0,is_training,reuse)
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

            lf = lf0

            ops=[lf0,fc0,lf1,fc1,lf2,fc2]

    return upf0, lf, ops


def pointnet_model(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats0=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats0],axis=1)
            feats1=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats1],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.1  # rescale
            feats2=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats2],axis=1)
            feats3=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats3],axis=1)

            feats_stage0_pool=pointnet_pool(dxyzs[0],feats_stage0,[16,16],64,'pool0',vlens[0],vbegs[0],reuse)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.6)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.8  # rescale

            feats5=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats5',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats5],axis=1)
            feats6=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats6],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale
            feats7=pointnet_conv(sxyzs,feats,[16,16,16],32,'feats7',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats7],axis=1)
            feats8=pointnet_conv(sxyzs,feats,[16,16,24],48,'feats8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats8],axis=1)
            feats9=pointnet_conv(sxyzs,feats,[16,16,32],64,'feats9',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats9],axis=1)

            feats_stage1_pool=pointnet_pool(dxyzs[1],feats_stage1,[32,32],128,'pool1',vlens[1],vbegs[1],reuse)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale
            feats10=pointnet_conv(sxyzs,feats,[32,32,32],64,'feats10',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats10],axis=1)
            feats11=pointnet_conv(sxyzs,feats,[32,32,48],96,'feats11',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats11],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            # for idx, fd in enumerate([64,64,64,128]):
            for idx, fd in enumerate([64,64,128]):
                cfeats = tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='global_{}'.format(idx),
                                                           activation_fn=tf.nn.relu, reuse=reuse)
                feats = tf.concat([feats, cfeats], axis=1)

            feats_stage2_global = tf.contrib.layers.fully_connected(feats, num_outputs=256, scope='global_out',
                                                             activation_fn=None, reuse=reuse)
            # feats_stage2_global = tf.contrib.layers.fully_connected(feats, num_outputs=384, scope='global_out',
            #                                                  activation_fn=None, reuse=reuse)
        with tf.name_scope('unpool'):
            lf2=tf.concat([feats_stage2_global,feats_stage2],axis=1)
            lf2_up=graph_unpool_stage(1,lf2,vlens[1],vbegs[1],vcens[1])

            lf1=tf.concat([lf2_up,feats_stage1],axis=1)
            lf1_up=graph_unpool_stage(0,lf1,vlens[0],vbegs[0],vcens[0])

            lf0=tf.concat([lf1_up,feats_stage0],axis=1)

        return lf0, feats_stage0

def pointnet_baseline_model(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats0=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats0],axis=1)
            feats1=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats1],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.1  # rescale
            feats2=pointnet_conv(sxyzs,feats,[],32,'feats2-0',nidxs,nlens,nbegs,ncens,reuse)
            feats2=pointnet_conv(sxyzs,feats2,[8,8,16],32,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats2],axis=1)

            feats3=pointnet_conv(sxyzs,feats,[],32,'feats3-0',nidxs,nlens,nbegs,ncens,reuse)
            feats3=pointnet_conv(sxyzs,feats3,[8,8,16],32,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats3],axis=1)

            feats_stage0_pool=pointnet_pool(dxyzs[0],feats_stage0,[16,16],64,'pool0',vlens[0],vbegs[0],reuse)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.6)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.6  # rescale

            feats5=pointnet_conv(sxyzs,feats,[],32,'feats5-0',nidxs,nlens,nbegs,ncens,reuse)
            feats5=pointnet_conv(sxyzs,feats5,[8,8,16],32,'feats5',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats5],axis=1)

            feats6=pointnet_conv(sxyzs,feats,[],32,'feats6-0',nidxs,nlens,nbegs,ncens,reuse)
            feats6=pointnet_conv(sxyzs,feats6,[8,8,16],32,'feats6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats6],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale

            feats7=pointnet_conv(sxyzs,feats,[],32,'feats7-0',nidxs,nlens,nbegs,ncens,reuse)
            feats7=pointnet_conv(sxyzs,feats7,[16,16,16],32,'feats7',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats7],axis=1)

            feats8=pointnet_conv(sxyzs,feats,[],48,'feats8-0',nidxs,nlens,nbegs,ncens,reuse)
            feats8=pointnet_conv(sxyzs,feats8,[16,16,24],48,'feats8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats8],axis=1)

            feats9=pointnet_conv(sxyzs,feats,[],64,'feats9-0',nidxs,nlens,nbegs,ncens,reuse)
            feats9=pointnet_conv(sxyzs,feats9,[16,16,32],64,'feats9',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats9],axis=1)

            feats_stage1_pool=pointnet_pool(dxyzs[1],feats_stage1,[32,32],128,'pool1',vlens[1],vbegs[1],reuse)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale
            feats10=pointnet_conv(sxyzs,feats,[],128,'feats10-0',nidxs,nlens,nbegs,ncens,reuse)
            feats10=pointnet_conv(sxyzs,feats10,[32,32,32],64,'feats10',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats10],axis=1)

            feats11=pointnet_conv(sxyzs,feats,[],128,'feats11-0',nidxs,nlens,nbegs,ncens,reuse)
            feats11=pointnet_conv(sxyzs,feats11,[32,32,48],96,'feats11',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats11],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            # for idx, fd in enumerate([64,64,64,128]):
            for idx, fd in enumerate([64,64,128]):
                cfeats = tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='global_{}'.format(idx),
                                                           activation_fn=tf.nn.relu, reuse=reuse)
                feats = tf.concat([feats, cfeats], axis=1)

            feats_stage2_global = tf.contrib.layers.fully_connected(feats, num_outputs=256, scope='global_out',
                                                             activation_fn=None, reuse=reuse)
            # feats_stage2_global = tf.contrib.layers.fully_connected(feats, num_outputs=384, scope='global_out',
            #                                                  activation_fn=None, reuse=reuse)
        with tf.name_scope('unpool'):
            lf2=tf.concat([feats_stage2_global,feats_stage2],axis=1)
            lf2_up=graph_unpool_stage(1,lf2,vlens[1],vbegs[1],vcens[1])

            lf1=tf.concat([lf2_up,feats_stage1],axis=1)
            lf1_up=graph_unpool_stage(0,lf1,vlens[0],vbegs[0],vcens[0])

            lf0=tf.concat([lf1_up,feats_stage0],axis=1)

        return lf0, feats_stage0


def ecd_xyz_v3(edge_coord, phi_dims, g_dims, out_dim, nlens, nbegs, ncens, is_training, name, reuse=None):
        phi_edge_feats=edge_coord
        dim_sum=3
        for idx,fd in enumerate(phi_dims):
            phi_out_feats=tf.contrib.layers.fully_connected(
                phi_edge_feats, num_outputs=fd,scope='{}_ifc_{}'.format(name,idx),
                weights_initializer=tf.truncated_normal_initializer(stddev=np.sqrt(8.0/fd)),
                activation_fn=tf.nn.relu, reuse=reuse)

            phi_edge_feats=tf.concat([phi_edge_feats,phi_out_feats],axis=1)
            dim_sum+=fd

        edge_weights=tf.contrib.layers.fully_connected(phi_edge_feats, num_outputs=dim_sum, scope='{}_fc_ew'.format(name),
                                                       weights_initializer=tf.truncated_normal_initializer(stddev=np.sqrt(8.0/dim_sum)),
                                                       activation_fn=tf.nn.tanh, reuse=reuse)

        edge_feats=edge_weights*phi_edge_feats
        for idx,fd in enumerate(g_dims):
            g_out_feats=tf.contrib.layers.fully_connected(edge_feats, num_outputs=fd, scope='{}_ofc_{}'.format(name,idx),
                                                          weights_initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0/fd)),
                                                          activation_fn=tf.nn.relu, reuse=reuse)
            edge_feats=tf.concat([edge_feats,g_out_feats],axis=1)
            dim_sum+=fd

        eps=1e-3
        neighbor_size_inv=tf.expand_dims((1.0+eps) / (tf.cast(nlens, tf.float32) + eps), axis=1)              # [pn]
        point_feats=neighbor_size_inv*neighbor_ops.neighbor_sum_feat_gather(edge_feats, ncens, nlens, nbegs)  # [pn,ofn]

        point_feats=tf.contrib.layers.fully_connected(point_feats, num_outputs=out_dim, scope='{}_fc_out'.format(name),
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={"is_training":is_training,"scale":True,
                                                                         "scope":'{}_fc_out_bn'.format(name)},
                                                      activation_fn=tf.nn.relu, reuse=reuse)
        return point_feats


def ecd_feats_v3(edge_coord, point_feats, input_dim, phi_dims, g_dims, output_dim,
                 nidxs, nlens, nbegs, ncens, is_training, name, reuse=None):
    with tf.name_scope(name):
        diff_edge_feats = neighbor_ops.neighbor_scatter(point_feats, nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
        phi_edge_feats = tf.concat([edge_coord,diff_edge_feats], axis=1)

        for idx,fd in enumerate(phi_dims):
            phi_out_feats = tf.contrib.layers.fully_connected(phi_edge_feats, num_outputs=fd, scope='{}_ifc_{}'.format(name,idx),
                                                              weights_initializer=tf.truncated_normal_initializer(stddev=np.sqrt(3.0/fd)),
                                                              activation_fn=tf.nn.relu, reuse=reuse)
            phi_edge_feats=tf.concat([phi_edge_feats,phi_out_feats],axis=1)

        edge_weights = tf.contrib.layers.fully_connected(phi_edge_feats, num_outputs=input_dim, scope='{}_fc_ew'.format(name),
                                                         weights_initializer=tf.truncated_normal_initializer(stddev=np.sqrt(3.0/input_dim)),
                                                         activation_fn=tf.nn.tanh, reuse=reuse)

        edge_feats=neighbor_ops.neighbor_scatter(point_feats, nidxs, nlens, nbegs, use_diff=False)      # [en,ifn]
        edge_feats= edge_weights * edge_feats
        for idx,fd in enumerate(g_dims):
            g_out_feats=tf.contrib.layers.fully_connected(edge_feats, num_outputs=fd, scope='{}_ofc_{}'.format(name, idx),
                                                          weights_initializer=tf.truncated_normal_initializer(stddev=np.sqrt(2.0/fd)),
                                                          activation_fn=tf.nn.relu, reuse=reuse)
            edge_feats=tf.concat([edge_feats,g_out_feats], axis=1)

        eps=1e-3
        neighbor_size_inv = tf.expand_dims((1.0+eps) / (tf.cast(nlens, tf.float32) + eps), axis=1)                 # [pn]
        point_feats= neighbor_size_inv * neighbor_ops.neighbor_sum_feat_gather(edge_feats, ncens, nlens, nbegs)  # [pn,ofn]
        point_feats=tf.contrib.layers.fully_connected(point_feats, num_outputs=output_dim, scope='{}_fc_out'.format(name),
                                                      normalizer_fn=tf.contrib.layers.batch_norm,
                                                      normalizer_params={"is_training":is_training,"scale":True,
                                                                         "scope":'{}_fc_out_bn'.format(name)},
                                                      activation_fn=tf.nn.relu, reuse=reuse)

        return point_feats



def ecd_stage_v3(stage_idx, xyzs, dxyz, feats,
                 xyz_dim, feats_dims, embed_dims, final_dim,
                 radius, voxel_size, is_training, reuse, name=''):
    with tf.name_scope('stage_{}'.format(stage_idx)):
        nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,radius)

        sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
        sxyzs /= radius   # rescale

        xyz_phi_dims=[xyz_dim/2,xyz_dim/2]
        xyz_g_dims=[xyz_dim/2,xyz_dim/2]
        xyz_gc=ecd_xyz_v3(sxyzs,xyz_phi_dims,xyz_g_dims,xyz_dim,nlens,nbegs,ncens,
                          is_training,'{}{}_xyz_gc'.format(name,stage_idx),reuse)
        cfeats = tf.concat([xyz_gc, feats], axis=1)

        # cdim = feats_dim + xyz_dim
        layer_idx = 1
        for fdim in feats_dims:
            conv_feats=tf.contrib.layers.fully_connected(
                cfeats, num_outputs=fdim, scope='{}{}_{}_fc'.format(name, stage_idx, layer_idx),
                activation_fn=tf.nn.relu, reuse=reuse)
            conv_feats = ecd_feats_v3(
                sxyzs, conv_feats, fdim, [fdim/2,fdim/2], [fdim/2,fdim/2], fdim, nidxs, nlens, nbegs, ncens,
                is_training, '{}{}_{}_gc'.format(name,stage_idx,layer_idx), reuse)

            cfeats = tf.concat([cfeats, conv_feats], axis=1)
            layer_idx += 1
            # cdim += fdim

        with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
            with tf.name_scope('fc_global{}'.format(stage_idx)):
                dxyz= dxyz / voxel_size
                fc_feats = tf.concat([cfeats, dxyz], axis=1)
                for i, gfd in enumerate(embed_dims):
                    fc = tf.contrib.layers.fully_connected(fc_feats, num_outputs=gfd,
                                                           scope='{}{}_{}_gfc'.format(name,stage_idx, i))
                    fc_feats=tf.concat([fc,fc_feats],axis=1)

                fc_final = tf.contrib.layers.fully_connected(fc_feats, num_outputs=final_dim, activation_fn=None,
                                                             scope='{}{}_final_gfc'.format(name,stage_idx))

    return fc_final, cfeats





def pgnet_model_v3_bn(xyzs, dxyzs, feats, vlens, vbegs, vcens, voxel_sizes, block_size,
                      is_training, radius=(0.15,0.3,0.5), reuse=False):
    with tf.name_scope('base_graph_conv_edge_net'):
        with tf.variable_scope('base_graph_conv_edge_net',reuse=reuse):
            with tf.name_scope('conv_stage0'):
                fc0, lf0 = ecd_stage_v3(0,xyzs[0],dxyzs[0],feats,
                                        16,[16,16],[8,8,8],64,
                                        radius[0],voxel_sizes[0],is_training,reuse)

                fc0_pool = graph_max_pool_stage(0, fc0, vlens[0], vbegs[0])             # 64
                lf0_avg = graph_avg_pool_stage(0, lf0, vlens[0], vbegs[0], vcens[0])    # 61
                ifeats_0 = tf.concat([fc0_pool,lf0_avg],axis=1)

            with tf.name_scope('conv_stage1'):
                fc1, lf1 = ecd_stage_v3(1,xyzs[1],dxyzs[1],ifeats_0,    # !!! bug dxyzs to xyzs
                                        16,[32,32,32,32,32,32,32,32,32],[32,32,32],256,
                                        radius[1],voxel_sizes[1],is_training,reuse)
                fc1_pool = graph_max_pool_stage(1, fc1, vlens[1], vbegs[1])         # 256
                lf1_avg = graph_avg_pool_stage(1, lf1, vlens[1], vbegs[1], vcens[1])# 429
                ifeats_1 = tf.concat([fc1_pool,lf1_avg],axis=1)                     # 685

            with tf.name_scope('conv_stage2'):
                fc2, lf2 = ecd_stage_v3(2, xyzs[2],xyzs[2], ifeats_1,
                                     16,[32,32,32,32,32,32,32,32,32],[32,32,32],512,
                                     radius[2],block_size,is_training,reuse)
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


def ecd_feats_v4(sxyzs, feats, ifn, ifc_dims, ofn, nidxs, nidxs_lens, nidxs_bgs, cidxs, name, reuse=None):
    with tf.name_scope(name):
        sfeats = neighbor_ops.neighbor_scatter(feats, nidxs, nidxs_lens, nidxs_bgs, use_diff=True)  # [en,ifn]
        sfeats = tf.concat([sfeats, sxyzs], axis=1)

        for idx, fd in enumerate(ifc_dims):
            cfeats = tf.contrib.layers.fully_connected(sfeats, num_outputs=fd, scope='{}_ifc_{}'.format(name, idx),
                                                       activation_fn=tf.nn.relu, reuse=reuse)
            sfeats = tf.concat([cfeats, sfeats], axis=1)

        ew = tf.contrib.layers.fully_connected(sfeats, num_outputs=ifn, scope='{}_fc_ew'.format(name),
                                               activation_fn=None, reuse=reuse)

        norm = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(ew), axis=1) + 1e-5), axis=1)
        ew /= (norm + 1e-5)
        with tf.variable_scope(name):
            weights_transformer = variable_on_cpu('edge_weights_trans', [1, ifn], tf.ones_initializer)
            ew *= weights_transformer

        feats = neighbor_ops.neighbor_scatter(feats, nidxs, nidxs_lens, nidxs_bgs, use_diff=False)  # [en,ifn]
        feats = ew * feats

        eps = 1e-3
        weights_inv = tf.expand_dims((1.0 + eps) / (tf.cast(nidxs_lens, tf.float32) + eps), axis=1)  # [pn]
        feats = weights_inv * neighbor_ops.neighbor_sum_feat_gather(feats, cidxs, nidxs_lens, nidxs_bgs)  # [pn,ofn]
        feats = tf.contrib.layers.fully_connected(feats, num_outputs=ofn, scope='{}_fc_out'.format(name),
                                                  activation_fn=None, reuse=reuse)

        return feats


def pgnet_model_v7(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale

            feats0=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats0],axis=1)

            feats1=ecd_feats_v4(sxyzs,feats,19,[16],16,nidxs,nlens,nbegs,ncens,'ecd1',reuse)
            feats=tf.concat([feats,feats1],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.1  # rescale

            feats2=ecd_feats_v4(sxyzs,feats,35,[32],32,nidxs,nlens,nbegs,ncens,'ecd2',reuse)
            feats=tf.concat([feats,feats2],axis=1)

            feats3=ecd_feats_v4(sxyzs,feats,67,[32],32,nidxs,nlens,nbegs,ncens,'ecd3',reuse)
            feats_stage0=tf.concat([feats,feats3],axis=1)

            feats_stage0_pool=pointnet_pool(dxyzs[0],feats_stage0,[16,16],64,'pool0',vlens[0],vbegs[0],reuse)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.6)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.6

            feats4=pointnet_conv(sxyzs,feats,[8,8,16],32,'feats4',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats4],axis=1)
            feats5=ecd_feats_v4(sxyzs,feats,96,[32],32,nidxs,nlens,nbegs,ncens,'ecd5',reuse)
            feats=tf.concat([feats,feats5],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale
            feats6=ecd_feats_v4(sxyzs,feats,128,[32],32,nidxs,nlens,nbegs,ncens,'ecd6',reuse)
            feats=tf.concat([feats,feats6],axis=1)
            feats7=ecd_feats_v4(sxyzs,feats,160,[48],48,nidxs,nlens,nbegs,ncens,'ecd7',reuse)
            feats=tf.concat([feats,feats7],axis=1)
            feats8=ecd_feats_v4(sxyzs,feats,208,[64],64,nidxs,nlens,nbegs,ncens,'ecd8',reuse)
            feats_stage1=tf.concat([feats,feats8],axis=1)

            feats_stage1_pool=pointnet_pool(dxyzs[1],feats_stage1,[32,32],128,'pool1',vlens[1],vbegs[1],reuse)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale
            feats9=ecd_feats_v4(sxyzs,feats,128,[64],64,nidxs,nlens,nbegs,ncens,'ecd9',reuse)
            feats=tf.concat([feats,feats9],axis=1)
            feats10=ecd_feats_v4(sxyzs,feats,192,[96],96,nidxs,nlens,nbegs,ncens,'ecd10',reuse)
            feats_stage2=tf.concat([feats,feats10],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            for idx, fd in enumerate([64,64,64,128]):
                cfeats = tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='global_{}'.format(idx),
                                                           activation_fn=tf.nn.relu, reuse=reuse)
                feats = tf.concat([feats, cfeats], axis=1)

            feats_stage2_global = tf.contrib.layers.fully_connected(feats, num_outputs=384, scope='global_out',
                                                                    activation_fn=None, reuse=reuse)
        with tf.name_scope('unpool'):
            lf2=tf.concat([feats_stage2_global,feats_stage2],axis=1)
            lf2_up=graph_unpool_stage(1,lf2,vlens[1],vbegs[1],vcens[1])

            lf1=tf.concat([lf2_up,feats_stage1],axis=1)
            lf1_up=graph_unpool_stage(0,lf1,vlens[0],vbegs[0],vcens[0])

            lf0=tf.concat([lf1_up,feats_stage0],axis=1)

        return lf0, feats_stage0

def mlp_anchor_conv(sxyzs, feats, ifn, weights_dims, ofn, anchor_num, name,
                    nidxs, nlens, nbegs, ncens, reuse=None, l2_norm=False):
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
            activation_fn=None, reuse=reuse)

        return output_point_feats


def pgnet_model_v8(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale

            feats0_pn=pointnet_conv(sxyzs,feats,[8],8,'pointnet0',nidxs,nlens,nbegs,ncens,reuse)
            feats0=mlp_anchor_conv(sxyzs,feats0_pn,8,[16],16,9,'anchor_conv0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats0,feats0_pn],axis=1)

            feats1_pn=pointnet_conv(sxyzs,feats,[8],8,'pointnet1',nidxs,nlens,nbegs,ncens,reuse)
            feats1=mlp_anchor_conv(sxyzs,feats1_pn,8,[16],16,9,'anchor_conv1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats1,feats1_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.1  # rescale

            feats2_pn=pointnet_conv(sxyzs,feats,[16],16,'pointnet2',nidxs,nlens,nbegs,ncens,reuse)
            feats2=mlp_anchor_conv(sxyzs,feats2_pn,16,[32],32,9,'anchor_conv2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats2,feats2_pn],axis=1)

            feats3_pn=pointnet_conv(sxyzs,feats,[16],16,'pointnet3',nidxs,nlens,nbegs,ncens,reuse)
            feats3=mlp_anchor_conv(sxyzs,feats3_pn,16,[32],32,9,'anchor_conv3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats3,feats3_pn],axis=1)

            feats_stage0_pool=pointnet_pool(dxyzs[0],feats_stage0,[16,16],64,'pool0',vlens[0],vbegs[0],reuse)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.6)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.6

            feats4_pn=pointnet_conv(sxyzs,feats,[16],16,'pointnet4',nidxs,nlens,nbegs,ncens,reuse)
            feats4=mlp_anchor_conv(sxyzs,feats4_pn,16,[32],32,9,'anchor_conv4',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats4,feats4_pn],axis=1)

            feats5_pn=pointnet_conv(sxyzs,feats,[16],16,'pointnet5',nidxs,nlens,nbegs,ncens,reuse)
            feats5=mlp_anchor_conv(sxyzs,feats5_pn,16,[32],32,9,'anchor_conv5',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats5,feats5_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale

            feats6_pn=pointnet_conv(sxyzs,feats,[16],16,'pointnet6',nidxs,nlens,nbegs,ncens,reuse)
            feats6=mlp_anchor_conv(sxyzs,feats6_pn,16,[24],48,12,'anchor_conv6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats6,feats6_pn],axis=1)

            feats7_pn=pointnet_conv(sxyzs,feats,[20],20,'pointnet7',nidxs,nlens,nbegs,ncens,reuse)
            feats7=mlp_anchor_conv(sxyzs,feats7_pn,20,[32],64,12,'anchor_conv7',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats7,feats7_pn],axis=1)

            feats_stage1_pool=pointnet_pool(dxyzs[1],feats_stage1,[32,32],128,'pool1',vlens[1],vbegs[1],reuse)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale

            feats8_pn=pointnet_conv(sxyzs,feats,[24],24,'pointnet8',nidxs,nlens,nbegs,ncens,reuse)
            feats8=mlp_anchor_conv(sxyzs,feats8_pn,24,[32],64,12,'anchor_conv8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats8,feats8_pn],axis=1)

            feats9_pn=pointnet_conv(sxyzs,feats,[24],24,'pointnet9',nidxs,nlens,nbegs,ncens,reuse)
            feats9=mlp_anchor_conv(sxyzs,feats9_pn,24,[48],96,16,'anchor_conv9',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats9,feats9_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            for idx, fd in enumerate([64,64,128]):
                cfeats = tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='global_{}'.format(idx),
                                                           activation_fn=tf.nn.relu, reuse=reuse)
                feats = tf.concat([feats, cfeats], axis=1)

            feats_stage2_global = tf.contrib.layers.fully_connected(feats, num_outputs=256, scope='global_out',
                                                                    activation_fn=None, reuse=reuse)
        with tf.name_scope('unpool'):
            lf2=tf.concat([feats_stage2_global,feats_stage2],axis=1)
            lf2_up=graph_unpool_stage(1,lf2,vlens[1],vbegs[1],vcens[1])

            lf1=tf.concat([lf2_up,feats_stage1],axis=1)
            lf1_up=graph_unpool_stage(0,lf1,vlens[0],vbegs[0],vcens[0])

            lf0=tf.concat([lf1_up,feats_stage0],axis=1)

        return lf0, feats_stage0


def pointnet2_v2(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale

            feats0_pn=pointnet_conv(sxyzs,feats,[8],8,'pointnet0',nidxs,nlens,nbegs,ncens,reuse)
            feats0=pointnet_conv(sxyzs,feats0_pn,[8,16],16,'pointnet0-1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats0,feats0_pn],axis=1)

            feats1_pn=pointnet_conv(sxyzs,feats,[8],8,'pointnet1',nidxs,nlens,nbegs,ncens,reuse)
            feats1=pointnet_conv(sxyzs,feats1_pn,[8,16],16,'pointnet1-1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats1,feats1_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.1  # rescale

            feats2_pn=pointnet_conv(sxyzs,feats,[16],16,'pointnet2',nidxs,nlens,nbegs,ncens,reuse)
            feats2=pointnet_conv(sxyzs,feats2_pn,[16,32],32,'pointnet2-1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats2,feats2_pn],axis=1)

            feats3_pn=pointnet_conv(sxyzs,feats,[16],16,'pointnet3',nidxs,nlens,nbegs,ncens,reuse)
            feats3=pointnet_conv(sxyzs,feats3_pn,[16,32],32,'pointnet3-1',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats3,feats3_pn],axis=1)

            feats_stage0_pool=pointnet_pool(dxyzs[0],feats_stage0,[16,16],64,'pool0',vlens[0],vbegs[0],reuse)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.6)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.6

            feats4_pn=pointnet_conv(sxyzs,feats,[16],16,'pointnet4',nidxs,nlens,nbegs,ncens,reuse)
            feats4=pointnet_conv(sxyzs,feats4_pn,[16,32],32,'pointnet4-1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats4,feats4_pn],axis=1)

            feats5_pn=pointnet_conv(sxyzs,feats,[16],16,'pointnet5',nidxs,nlens,nbegs,ncens,reuse)
            feats5=pointnet_conv(sxyzs,feats5_pn,[16,32],32,'pointnet5-1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats5,feats5_pn],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale

            feats6_pn=pointnet_conv(sxyzs,feats,[16],16,'pointnet6',nidxs,nlens,nbegs,ncens,reuse)
            feats6=pointnet_conv(sxyzs,feats6_pn,[24,48],48,'pointnet6-1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats6,feats6_pn],axis=1)

            feats7_pn=pointnet_conv(sxyzs,feats,[20],20,'pointnet7',nidxs,nlens,nbegs,ncens,reuse)
            feats7=pointnet_conv(sxyzs,feats7_pn,[32,64],64,'pointnet7-1',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats7,feats7_pn],axis=1)

            feats_stage1_pool=pointnet_pool(dxyzs[1],feats_stage1,[32,32],128,'pool1',vlens[1],vbegs[1],reuse)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale

            feats8_pn=pointnet_conv(sxyzs,feats,[24],24,'pointnet8',nidxs,nlens,nbegs,ncens,reuse)
            feats8=mlp_anchor_conv(sxyzs,feats8_pn,24,[32],64,12,'anchor_conv8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats8,feats8_pn],axis=1)

            feats9_pn=pointnet_conv(sxyzs,feats,[24],24,'pointnet9',nidxs,nlens,nbegs,ncens,reuse)
            feats9=mlp_anchor_conv(sxyzs,feats9_pn,24,[48],96,16,'anchor_conv9',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats9,feats9_pn],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            for idx, fd in enumerate([64,64,128]):
                cfeats = tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='global_{}'.format(idx),
                                                           activation_fn=tf.nn.relu, reuse=reuse)
                feats = tf.concat([feats, cfeats], axis=1)

            feats_stage2_global = tf.contrib.layers.fully_connected(feats, num_outputs=256, scope='global_out',
                                                                    activation_fn=None, reuse=reuse)
        with tf.name_scope('unpool'):
            lf2=tf.concat([feats_stage2_global,feats_stage2],axis=1)
            lf2_up=graph_unpool_stage(1,lf2,vlens[1],vbegs[1],vcens[1])

            lf1=tf.concat([lf2_up,feats_stage1],axis=1)
            lf1_up=graph_unpool_stage(0,lf1,vlens[0],vbegs[0],vcens[0])

            lf0=tf.concat([lf1_up,feats_stage0],axis=1)

        return lf0, feats_stage0


def mlp_anchor_conv_baseline(sxyzs, feats, ifn, weights_dims, ofn, anchor_num, name,
                             nidxs, nlens, nbegs, ncens, reuse=None):
    with tf.name_scope(name):
        # [pn]
        edge_weights_feats = graph_concat_scatter(feats,nidxs,nlens,nbegs,ncens)
        edge_weights_feats = tf.concat([sxyzs,edge_weights_feats],axis=1)

        for idx,fd in enumerate(weights_dims):
            cfeats=tf.contrib.layers.fully_connected(edge_weights_feats, num_outputs=fd,
                                                     scope='{}_fc_weights_{}'.format(name,idx),
                                                     activation_fn=tf.nn.relu, reuse=reuse)
            edge_weights_feats=tf.concat([cfeats,edge_weights_feats],axis=1)

        # [en,an]
        edge_weights=tf.contrib.layers.fully_connected(edge_weights_feats, num_outputs=anchor_num,
                                                       scope='{}_fc_weights_final'.format(name),
                                                       activation_fn=tf.exp, reuse=reuse)
        # [en,ifn]
        edge_feats = neighbor_ops.neighbor_scatter(feats, nidxs, nlens, nbegs, use_diff=False)

        # weight edge feats
        weighted_edge_feats=tf.expand_dims(edge_weights,axis=2)*tf.expand_dims(edge_feats,axis=1)  # [en,an,ifn]
        weighted_edge_feats=tf.reshape(weighted_edge_feats,[-1,anchor_num*ifn])                    # [en,an*ifn]

        # sum to points
        weighted_point_feats=neighbor_ops.neighbor_sum_feat_gather(weighted_edge_feats,ncens,nlens,nbegs) #[pn,an*ifn]
        point_weights=neighbor_ops.neighbor_sum_feat_gather(edge_weights,ncens,nlens,nbegs) #[pn,an]
        weighted_point_feats=tf.reshape(weighted_point_feats,[-1,anchor_num,ifn])

        # normalize point number
        weighted_point_feats/=tf.expand_dims(point_weights,axis=2)
        weighted_point_feats=tf.reshape(weighted_point_feats,[-1,anchor_num*ifn])


        output_point_feats = tf.contrib.layers.fully_connected(
            weighted_point_feats, num_outputs=ofn, scope='{}_fc_out'.format(name),
            activation_fn=tf.nn.relu, reuse=reuse)

        return output_point_feats


def mlp_anchor_conv_revise(sxyzs, feats, ifn, weights_dims, ofn, anchor_num, name,
                             nidxs, nlens, nbegs, ncens, reuse=None):
    with tf.name_scope(name):
        # [pn]
        edge_weights_feats = graph_concat_scatter(feats,nidxs,nlens,nbegs,ncens)
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

        norm = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(edge_weights), axis=1) + 1e-5), axis=1)
        edge_weights /= (norm + 1e-5)
        with tf.variable_scope(name):
            weights_transformer = variable_on_cpu('edge_weights_trans', [1, anchor_num], tf.ones_initializer)
        edge_weights *= weights_transformer

        # [en,ifn]
        edge_feats = neighbor_ops.neighbor_scatter(feats, nidxs, nlens, nbegs, use_diff=False)

        # weight edge feats
        weighted_edge_feats=tf.expand_dims(edge_weights,axis=2)*tf.expand_dims(edge_feats,axis=1)  # [en,an,ifn]
        weighted_edge_feats=tf.reshape(weighted_edge_feats,[-1,anchor_num*ifn])                    # [en,an*ifn]

        # sum to points
        weighted_point_feats=neighbor_ops.neighbor_sum_feat_gather(weighted_edge_feats,ncens,nlens,nbegs) #[pn,an*ifn]

        # normalize point number
        weighted_point_feats/=tf.expand_dims(tf.cast(nlens,tf.float32),axis=1)

        output_point_feats = tf.contrib.layers.fully_connected(
            weighted_point_feats, num_outputs=ofn, scope='{}_fc_out'.format(name),
            activation_fn=tf.nn.leaky_relu, reuse=reuse)

        return output_point_feats


def mlp_anchor_conv_revise_v2(sxyzs, feats, ifn, weights_dims, ofn, anchor_num, name,
                              nidxs, nlens, nbegs, ncens, reuse=None,final_act=tf.nn.leaky_relu,l2_norm=False):
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
            norm = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(edge_weights), axis=1) + 1e-5), axis=1)
            edge_weights /= (norm + 1e-5)
            with tf.variable_scope(name):
                weights_transformer = variable_on_cpu('edge_weights_trans', [1, anchor_num], tf.ones_initializer)
            edge_weights *= weights_transformer

        # [en,ifn]
        edge_feats = neighbor_ops.neighbor_scatter(feats, nidxs, nlens, nbegs, use_diff=False)

        # weight edge feats
        weighted_edge_feats=tf.expand_dims(edge_weights,axis=2)*tf.expand_dims(edge_feats,axis=1)  # [en,an,ifn]
        weighted_edge_feats=tf.reshape(weighted_edge_feats,[-1,anchor_num*ifn])                    # [en,an*ifn]

        # sum to points
        weighted_point_feats=neighbor_ops.neighbor_sum_feat_gather(weighted_edge_feats,ncens,nlens,nbegs) #[pn,an*ifn]

        # normalize point number
        weighted_point_feats/=tf.expand_dims(tf.cast(nlens,tf.float32),axis=1)

        output_point_feats = tf.contrib.layers.fully_connected(
            weighted_point_feats, num_outputs=ofn, scope='{}_fc_out'.format(name),
            activation_fn=final_act, reuse=reuse)

        return output_point_feats


def df_conv_baseline_model(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats0=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats0],axis=1)
            feats1=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats1],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.1  # rescale
            feats2=pointnet_conv(sxyzs,feats,[],16,'feats2-0',nidxs,nlens,nbegs,ncens,reuse)
            feats2=mlp_anchor_conv_baseline(sxyzs,feats2,16,[16],32,9,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats2],axis=1)

            feats3=pointnet_conv(sxyzs,feats,[],16,'feats3-0',nidxs,nlens,nbegs,ncens,reuse)
            feats3=mlp_anchor_conv_baseline(sxyzs,feats3,16,[16],32,9,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats3],axis=1)

            feats_stage0_pool=pointnet_pool(dxyzs[0],feats_stage0,[16,16],64,'pool0',vlens[0],vbegs[0],reuse)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.6)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.6  # rescale

            feats5=pointnet_conv(sxyzs,feats,[],32,'feats5-0',nidxs,nlens,nbegs,ncens,reuse)
            feats5=mlp_anchor_conv_baseline(sxyzs,feats5,32,[16],32,9,'feats5',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats5],axis=1)

            feats6=pointnet_conv(sxyzs,feats,[],32,'feats6-0',nidxs,nlens,nbegs,ncens,reuse)
            feats6=mlp_anchor_conv_baseline(sxyzs,feats6,32,[16],32,9,'feats6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats6],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale

            feats7=pointnet_conv(sxyzs,feats,[],32,'feats7-0',nidxs,nlens,nbegs,ncens,reuse)
            feats7=mlp_anchor_conv_baseline(sxyzs,feats7,32,[16],32,9,'feats7',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats7],axis=1)

            feats8=pointnet_conv(sxyzs,feats,[],48,'feats8-0',nidxs,nlens,nbegs,ncens,reuse)
            feats8=mlp_anchor_conv_baseline(sxyzs,feats8,48,[25],48,9,'feats8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats8],axis=1)

            feats9=pointnet_conv(sxyzs,feats,[],64,'feats9-0',nidxs,nlens,nbegs,ncens,reuse)
            feats9=mlp_anchor_conv_baseline(sxyzs,feats9,64,[32],64,9,'feats9',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats9],axis=1)

            feats_stage1_pool=pointnet_pool(dxyzs[1],feats_stage1,[32,32],128,'pool1',vlens[1],vbegs[1],reuse)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale
            feats10=pointnet_conv(sxyzs,feats,[],64,'feats10-0',nidxs,nlens,nbegs,ncens,reuse)
            feats10=mlp_anchor_conv_baseline(sxyzs,feats10,64,[64],64,9,'feats10',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats10],axis=1)

            feats11=pointnet_conv(sxyzs,feats,[],64,'feats11-0',nidxs,nlens,nbegs,ncens,reuse)
            feats11=mlp_anchor_conv_baseline(sxyzs,feats11,64,[64],96,9,'feats11',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats11],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            # for idx, fd in enumerate([64,64,64,128]):
            for idx, fd in enumerate([64,64,128]):
                cfeats = tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='global_{}'.format(idx),
                                                           activation_fn=tf.nn.relu, reuse=reuse)
                feats = tf.concat([feats, cfeats], axis=1)

            feats_stage2_global = tf.contrib.layers.fully_connected(feats, num_outputs=256, scope='global_out',
                                                             activation_fn=None, reuse=reuse)
            # feats_stage2_global = tf.contrib.layers.fully_connected(feats, num_outputs=384, scope='global_out',
            #                                                  activation_fn=None, reuse=reuse)
        with tf.name_scope('unpool'):
            lf2=tf.concat([feats_stage2_global,feats_stage2],axis=1)
            lf2_up=graph_unpool_stage(1,lf2,vlens[1],vbegs[1],vcens[1])

            lf1=tf.concat([lf2_up,feats_stage1],axis=1)
            lf1_up=graph_unpool_stage(0,lf1,vlens[0],vbegs[0],vcens[0])

            lf0=tf.concat([lf1_up,feats_stage0],axis=1)

        return lf0, feats_stage0


def df_conv_revise_model(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats0=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats0],axis=1)
            feats1=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats1],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.1  # rescale
            feats2=pointnet_conv(sxyzs,feats,[],16,'feats2-0',nidxs,nlens,nbegs,ncens,reuse)
            feats2=mlp_anchor_conv_revise(sxyzs,feats2,16,[16],32,9,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats2],axis=1)

            feats3=pointnet_conv(sxyzs,feats,[],16,'feats3-0',nidxs,nlens,nbegs,ncens,reuse)
            feats3=mlp_anchor_conv_revise(sxyzs,feats3,16,[16],32,9,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats3],axis=1)

            feats_stage0_pool=pointnet_pool(dxyzs[0],feats_stage0,[16,16],64,'pool0',vlens[0],vbegs[0],reuse)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.6)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.6  # rescale

            feats5=pointnet_conv(sxyzs,feats,[],32,'feats5-0',nidxs,nlens,nbegs,ncens,reuse)
            feats5=mlp_anchor_conv_revise(sxyzs,feats5,32,[16],32,9,'feats5',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats5],axis=1)

            feats6=pointnet_conv(sxyzs,feats,[],32,'feats6-0',nidxs,nlens,nbegs,ncens,reuse)
            feats6=mlp_anchor_conv_revise(sxyzs,feats6,32,[16],32,9,'feats6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats6],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale

            feats7=pointnet_conv(sxyzs,feats,[],32,'feats7-0',nidxs,nlens,nbegs,ncens,reuse)
            feats7=mlp_anchor_conv_revise(sxyzs,feats7,32,[16],32,9,'feats7',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats7],axis=1)

            feats8=pointnet_conv(sxyzs,feats,[],48,'feats8-0',nidxs,nlens,nbegs,ncens,reuse)
            feats8=mlp_anchor_conv_revise(sxyzs,feats8,48,[25],48,9,'feats8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats8],axis=1)

            feats9=pointnet_conv(sxyzs,feats,[],64,'feats9-0',nidxs,nlens,nbegs,ncens,reuse)
            feats9=mlp_anchor_conv_revise(sxyzs,feats9,64,[32],64,9,'feats9',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats9],axis=1)

            feats_stage1_pool=pointnet_pool(dxyzs[1],feats_stage1,[32,32],128,'pool1',vlens[1],vbegs[1],reuse)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale
            feats10=pointnet_conv(sxyzs,feats,[],64,'feats10-0',nidxs,nlens,nbegs,ncens,reuse)
            feats10=mlp_anchor_conv_revise(sxyzs,feats10,64,[64],64,9,'feats10',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats10],axis=1)

            feats11=pointnet_conv(sxyzs,feats,[],64,'feats11-0',nidxs,nlens,nbegs,ncens,reuse)
            feats11=mlp_anchor_conv_revise(sxyzs,feats11,64,[64],96,9,'feats11',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats11],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            for idx, fd in enumerate([64,64,128]):
                cfeats = tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='global_{}'.format(idx),
                                                           activation_fn=tf.nn.relu, reuse=reuse)
                feats = tf.concat([feats, cfeats], axis=1)

            feats_stage2_global = tf.contrib.layers.fully_connected(feats, num_outputs=256, scope='global_out',
                                                                    activation_fn=None, reuse=reuse)
        with tf.name_scope('unpool'):
            lf2=tf.concat([feats_stage2_global,feats_stage2],axis=1)
            lf2_up=graph_unpool_stage(1,lf2,vlens[1],vbegs[1],vcens[1])

            lf1=tf.concat([lf2_up,feats_stage1],axis=1)
            lf1_up=graph_unpool_stage(0,lf1,vlens[0],vbegs[0],vcens[0])

            lf0=tf.concat([lf1_up,feats_stage0],axis=1)

        return lf0, feats_stage0


def df_conv_revise_model_v2(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope('stage0'):
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.15)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.15  # rescale
            feats0=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats0',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats0],axis=1)
            feats1=pointnet_conv(sxyzs,feats,[4,4,8],16,'feats1',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats1],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[0], 0.1)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[0], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.1  # rescale
            feats2=pointnet_conv(sxyzs,feats,[],16,'feats2-0',nidxs,nlens,nbegs,ncens,reuse)
            feats2=mlp_anchor_conv_revise_v2(sxyzs,feats2,16,[16],32,9,'feats2',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats2],axis=1)

            feats3=pointnet_conv(sxyzs,feats,[],16,'feats3-0',nidxs,nlens,nbegs,ncens,reuse)
            feats3=mlp_anchor_conv_revise_v2(sxyzs,feats3,16,[16],32,9,'feats3',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage0=tf.concat([feats,feats3],axis=1)

            feats_stage0_pool=pointnet_pool(dxyzs[0],feats_stage0,[16,16],64,'pool0',vlens[0],vbegs[0],reuse)

        with tf.name_scope('stage1'):
            feats=feats_stage0_pool
            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.6)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.6  # rescale

            feats5=pointnet_conv(sxyzs,feats,[],32,'feats5-0',nidxs,nlens,nbegs,ncens,reuse)
            feats5=mlp_anchor_conv_revise_v2(sxyzs,feats5,32,[16],32,9,'feats5',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats5],axis=1)

            feats6=pointnet_conv(sxyzs,feats,[],32,'feats6-0',nidxs,nlens,nbegs,ncens,reuse)
            feats6=mlp_anchor_conv_revise_v2(sxyzs,feats6,32,[16],32,9,'feats6',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats6],axis=1)

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[1], 0.3)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[1], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.3  # rescale

            feats7=pointnet_conv(sxyzs,feats,[],32,'feats7-0',nidxs,nlens,nbegs,ncens,reuse)
            feats7=mlp_anchor_conv_revise_v2(sxyzs,feats7,32,[16],32,9,'feats7',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats7],axis=1)

            feats8=pointnet_conv(sxyzs,feats,[],48,'feats8-0',nidxs,nlens,nbegs,ncens,reuse)
            feats8=mlp_anchor_conv_revise_v2(sxyzs,feats8,48,[25],48,9,'feats8',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats8],axis=1)

            feats9=pointnet_conv(sxyzs,feats,[],64,'feats9-0',nidxs,nlens,nbegs,ncens,reuse)
            feats9=mlp_anchor_conv_revise_v2(sxyzs,feats9,64,[32],64,9,'feats9',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage1=tf.concat([feats,feats9],axis=1)

            feats_stage1_pool=pointnet_pool(dxyzs[1],feats_stage1,[32,32],128,'pool1',vlens[1],vbegs[1],reuse)

        with tf.name_scope('stage2'):
            feats=feats_stage1_pool

            nidxs, nlens, nbegs, ncens = search_neighborhood(xyzs[2], 0.9)
            sxyzs = neighbor_ops.neighbor_scatter(xyzs[2], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
            sxyzs /= 0.9  # rescale
            feats10=pointnet_conv(sxyzs,feats,[],64,'feats10-0',nidxs,nlens,nbegs,ncens,reuse)
            feats10=mlp_anchor_conv_revise_v2(sxyzs,feats10,64,[64],64,9,'feats10',nidxs,nlens,nbegs,ncens,reuse)
            feats=tf.concat([feats,feats10],axis=1)

            feats11=pointnet_conv(sxyzs,feats,[],64,'feats11-0',nidxs,nlens,nbegs,ncens,reuse)
            feats11=mlp_anchor_conv_revise_v2(sxyzs,feats11,64,[64],96,9,'feats11',nidxs,nlens,nbegs,ncens,reuse)
            feats_stage2=tf.concat([feats,feats11],axis=1)

            feats=tf.concat([xyzs[2],feats],axis=1)
            for idx, fd in enumerate([64,64,128]):
                cfeats = tf.contrib.layers.fully_connected(feats, num_outputs=fd, scope='global_{}'.format(idx),
                                                           activation_fn=tf.nn.relu, reuse=reuse)
                feats = tf.concat([feats, cfeats], axis=1)

            feats_stage2_global = tf.contrib.layers.fully_connected(feats, num_outputs=256, scope='global_out',
                                                                    activation_fn=None, reuse=reuse)
        with tf.name_scope('unpool'):
            lf2=tf.concat([feats_stage2_global,feats_stage2],axis=1)
            lf2_up=graph_unpool_stage(1,lf2,vlens[1],vbegs[1],vcens[1])

            lf1=tf.concat([lf2_up,feats_stage1],axis=1)
            lf1_up=graph_unpool_stage(0,lf1,vlens[0],vbegs[0],vcens[0])

            lf0=tf.concat([lf1_up,feats_stage0],axis=1)

        return lf0, feats_stage0
