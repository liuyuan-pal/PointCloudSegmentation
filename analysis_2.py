from model_new import *
from io_util import *

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random

ops={}
def ecd_feats(edge_coord, point_feats, input_dim, phi_dims, g_dims, doutput_dim,
              nidxs, nlens, nbegs, ncens, name, reuse=None):
    with tf.name_scope(name):
        ops['{}_feats'.format(name)]=point_feats
        diff_edge_feats = neighbor_ops.neighbor_scatter(point_feats, nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
        phi_edge_feats = tf.concat([diff_edge_feats, edge_coord], axis=1)

        for idx,fd in enumerate(phi_dims):
            phi_out_feats = tf.contrib.layers.fully_connected(phi_edge_feats, num_outputs=fd, scope='{}_ifc_{}'.format(name,idx),
                                                              activation_fn=tf.nn.relu, reuse=reuse)
            phi_edge_feats=tf.concat([phi_out_feats,phi_edge_feats],axis=1)

        edge_weights = tf.contrib.layers.fully_connected(phi_edge_feats, num_outputs=input_dim, scope='{}_fc_ew'.format(name),
                                                         activation_fn=tf.nn.tanh, reuse=reuse)

        ops['{}_weights'.format(name)]=edge_weights
        edge_feats=neighbor_ops.neighbor_scatter(point_feats, nidxs, nlens, nbegs, use_diff=False)      # [en,ifn]

        edge_feats= edge_weights * edge_feats
        ops['{}_wfeats'.format(name)]=edge_feats
        for idx,fd in enumerate(g_dims):
            g_out_feats=tf.contrib.layers.fully_connected(edge_feats, num_outputs=fd, scope='{}_ofc_{}'.format(name, idx),
                                                           activation_fn=tf.nn.relu, reuse=reuse)
            edge_feats=tf.concat([g_out_feats, edge_feats], axis=1)

        eps=1e-3
        neighbor_size_inv = tf.expand_dims((1.0+eps) / (tf.cast(nlens, tf.float32) + eps), axis=1)                 # [pn]
        point_feats= neighbor_size_inv * neighbor_ops.neighbor_sum_feat_gather(edge_feats, ncens, nlens, nbegs)  # [pn,ofn]
        point_feats=tf.contrib.layers.fully_connected(point_feats, num_outputs=doutput_dim, scope='{}_fc_out'.format(name),
                                                      activation_fn=tf.nn.relu, reuse=reuse)
        ops['{}_ofeats'.format(name)]=point_feats

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
        ops['{}_feats'.format(name)]=phi_edge_feats

        edge_weights=tf.contrib.layers.fully_connected(phi_edge_feats, num_outputs=dim_sum, scope='{}_fc_ew'.format(name),
                                                       activation_fn=tf.nn.tanh, reuse=reuse)
        ops['{}_weights'.format(name)]=edge_weights

        edge_feats=edge_weights*phi_edge_feats
        ops['{}_wfeats'.format(name)]=edge_weights
        for idx,fd in enumerate(g_dims):
            g_out_feats=tf.contrib.layers.fully_connected(edge_feats, num_outputs=fd, scope='{}_ofc_{}'.format(name,idx),
                                                          activation_fn=tf.nn.relu, reuse=reuse)
            edge_feats=tf.concat([g_out_feats,edge_feats],axis=1)

        eps=1e-3
        neighbor_size_inv=tf.expand_dims((1.0+eps) / (tf.cast(nlens, tf.float32) + eps), axis=1)              # [pn]
        point_feats=neighbor_size_inv*neighbor_ops.neighbor_sum_feat_gather(edge_feats, ncens, nlens, nbegs)  # [pn,ofn]

        point_feats=tf.contrib.layers.fully_connected(point_feats, num_outputs=out_dim, scope='{}_fc_out'.format(name),
                                                      activation_fn=tf.nn.relu, reuse=reuse)
        ops['{}_ofeats'.format(name)]=point_feats

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

    return upf0, lf,


def ecd_xyz_v3(edge_coord, phi_dims, g_dims, out_dim, nlens, nbegs, ncens, is_training, name, reuse=None):
        ops['{}_sxyz'.format(name)]=edge_coord
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
        ops['{}_ofeats'.format(name)]=point_feats
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
        ops['{}_ofeats'.format(name)]=point_feats

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
                                        32,[32,32,32,32,32,32,32,32,32],[32,32,32],256,
                                        radius[1],voxel_sizes[1],is_training,reuse)
                fc1_pool = graph_max_pool_stage(1, fc1, vlens[1], vbegs[1])         # 256
                lf1_avg = graph_avg_pool_stage(1, lf1, vlens[1], vbegs[1], vcens[1])# 429
                ifeats_1 = tf.concat([fc1_pool,lf1_avg],axis=1)                     # 685

            with tf.name_scope('conv_stage2'):
                fc2, lf2 = ecd_stage_v3(2, xyzs[2],xyzs[2], ifeats_1,
                                        32,[32,32,32,32,32,32,32,32,32],[32,32,32],512,
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


    return upf0, lf

def build_model():
    pls={}
    pls['xyzs'] = tf.placeholder(tf.float32, [None, 3], 'xyzs')
    pls['feats'] = tf.placeholder(tf.float32, [None, 3], 'feats')
    pls['logits_grad'] = tf.placeholder(tf.float32, [None, 13], 'logits_grad')
    pls['labels'] = tf.placeholder(tf.int32, [None], 'labels')
    pls['is_training'] = tf.placeholder(tf.bool, name='is_training')


    xyzs, dxyzs, feats, labels, vlens, vbegs, vcens = \
        points_pooling_two_layers(pls['xyzs'],pls['feats'],pls['labels'],voxel_size1=0.15,voxel_size2=0.3,block_size=3.0)
    global_feats, local_feats= pgnet_model_v3_bn(xyzs, dxyzs, feats, vlens, vbegs, vcens,[0.15,0.3],3.0,
                                                 pls['is_training'], [0.15, 0.3, 0.5], False)

    global_feats = tf.expand_dims(global_feats, axis=0)
    local_feats = tf.expand_dims(local_feats, axis=0)
    logits = classifier_v3(global_feats, local_feats, pls['is_training'], 13, False, use_bn=False)  # [1,pn,num_classes]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    ops['global_feats']=global_feats
    ops['logits']=logits

    return sess,pls

def sample_feats(name,sess,pls,fns):
    feats=[]
    random.shuffle(fns)
    for fn in fns[:3]:
        xyzs,rgbs,covars,labels,block_mins=read_pkl(fn)
        for i in xrange(len(xyzs[:10])):
            feats_vals=sess.run(ops[name],feed_dict={
                pls['xyzs']:xyzs[i],pls['feats']:rgbs[i],pls['labels']:labels[i],pls['is_training']:False
            })

            feats.append(feats_vals)

    feats=np.concatenate(feats,axis=0)

    return feats


def sample_one_feats(name,sess,pls,fns):
    feats=[]
    random.shuffle(fns)
    for fn in fns[:3]:
        xyzs,rgbs,covars,labels,block_mins=read_pkl(fn)
        for i in xrange(len(xyzs[:1])):
            feats_vals=sess.run(ops[name],feed_dict={
                pls['xyzs']:xyzs[i],pls['feats']:rgbs[i],pls['labels']:labels[i],pls['is_training']:False
            })

            feats.append(feats_vals)

    feats=np.concatenate(feats,axis=0)

    return feats

def cluster_feats(xyzs,feats,name):
    pass


def draw_hist(feats,name):
    plt.figure(0)
    plt.hist(feats)
    plt.savefig('test_result/{}_hist.png'.format(name))
    plt.close()


def draw_line(feats,name):
    plt.figure(0)
    plt.plot(np.arange(feats.shape[0]),feats,'-')
    plt.savefig('test_result/{}_line.png'.format(name))
    plt.close()

prefix=['{}_xyz_gc'.format(i) for i in xrange(3)]
prefix+=['0_{}_gc'.format(i) for i in xrange(1,3)]
prefix+=['1_{}_gc'.format(i) for i in xrange(1,10)]
prefix+=['2_{}_gc'.format(i) for i in xrange(1,10)]

if __name__=="__main__":
    sess,pls=build_model()
    train_list,test_list=get_block_train_test_split()
    random.shuffle(test_list)
    test_list=['data/S3DIS/sampled_test_new/'+fn for fn in test_list]

    saver=tf.train.Saver()
    saver.restore(sess,'model/pgnet_v3_bn/model10.ckpt')
    for name in ops.keys():
        print name

    names=[ '0_xyz_gc_ofeats','0_1_gc_ofeats','0_2_gc_ofeats',
            '1_xyz_gc_ofeats', '1_1_gc_ofeats', '1_3_gc_ofeats','1_5_gc_ofeats','1_9_gc_ofeats',
            '2_xyz_gc_ofeats', '2_1_gc_ofeats', '2_3_gc_ofeats','2_5_gc_ofeats','2_9_gc_ofeats']

    for name in names:
        feats=sample_feats(name,sess,pls,test_list)
        var=np.var(feats, axis=0)
        for k in xrange(feats.shape[1]):
            draw_hist(feats[:,k],name+'_{}'.format(k))
            print '{} {} {}'.format(name,k,var[k])

