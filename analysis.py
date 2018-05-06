from model_pgnet import *
from io_util import *

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random

ops={}

def ecd_xyz_test(dxyzs, feats_dims, final_feats_dim, diffusion_dims, trans_dims, out_dim,
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
            ops['{}_feats'.format(name)]=edge_feats
            # diffusion
            edge_diff_feats = dxyzs
            for di,dd in enumerate(diffusion_dims):
                fc_feats = tf.contrib.layers.fully_connected(edge_diff_feats, num_outputs=dd,
                                                             scope='{}{}_diffusion_fc'.format(name,di))
                edge_diff_feats=tf.concat([edge_diff_feats,fc_feats],axis=1)
            edge_weights = tf.contrib.layers.fully_connected(edge_diff_feats, num_outputs=final_feats_dim,
                                                             activation_fn=tf.nn.tanh,
                                                             scope='{}final_diffusion_fc'.format(name))
            ops['{}_weights'.format(name)] = edge_weights
            # trans
            edge_feats=edge_weights*edge_feats
            for ti,td in enumerate(trans_dims):
                fc_feats=tf.contrib.layers.fully_connected(edge_feats, num_outputs=td,
                                                           scope='{}{}_embed_fc'.format(name,ti),
                                                           activation_fn=tf.nn.relu, reuse=reuse)
                edge_feats=tf.concat([edge_feats,fc_feats],axis=1)
            ops['{}_trans'.format(name)] = edge_feats

            # divide by neighbor size
            eps=1e-3
            neighbor_size_inv=tf.expand_dims((1.0+eps) / (tf.cast(nlens, tf.float32) + eps), axis=1)
            point_feats=neighbor_size_inv*neighbor_ops.neighbor_sum_feat_gather(edge_feats, ncens, nlens, nbegs)

            point_feats=tf.contrib.layers.fully_connected(point_feats, num_outputs=out_dim, scope='{}out_embed_fc'.format(name),
                                                          activation_fn=tf.nn.relu, reuse=reuse)
            point_feats = tf.contrib.layers.batch_norm(point_feats,scale=True,is_training=is_training,
                                                       scope='{}_out_bn'.format(name))
            ops['{}_ofeats'.format(name)]=point_feats

            return point_feats


def ecd_feats_test(dxyzs, feats, embed_dim, diffusion_dims, trans_dims, out_dim,
                   nidxs, nlens, nbegs, ncens, name, is_training, reuse=None):
    with framework.arg_scope([tf.contrib.layers.fully_connected], activation_fn=tf.nn.relu, reuse=reuse):
        with tf.name_scope(name):
            # embed lowering dimensions
            feats=tf.contrib.layers.fully_connected(feats, num_outputs=embed_dim, scope='{}in_embed_fc'.format(name),
                                                    activation_fn=None, reuse=reuse)

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
            ops['{}_weights'.format(name)] = edge_weights
            # trans
            edge_feats = neighbor_ops.neighbor_scatter(feats, nidxs, nlens, nbegs, use_diff=False)
            ops['{}_feats'.format(name)]=edge_feats
            edge_feats = edge_weights * edge_feats
            for ti,td in enumerate(trans_dims):
                fc_feats=tf.contrib.layers.fully_connected(edge_feats, num_outputs=td,
                                                           scope='{}{}_embed_fc'.format(name,ti),
                                                           activation_fn=tf.nn.relu, reuse=reuse)
                edge_feats=tf.concat([edge_feats,fc_feats],axis=1)
            ops['{}_trans'.format(name)] = edge_feats

            # divide by neighbor size
            eps=1e-3
            neighbor_size_inv=tf.expand_dims((1.0+eps) / (tf.cast(nlens, tf.float32) + eps), axis=1)
            point_feats=neighbor_size_inv*neighbor_ops.neighbor_sum_feat_gather(edge_feats, ncens, nlens, nbegs)

            point_feats=tf.contrib.layers.fully_connected(point_feats, num_outputs=out_dim, scope='{}out_embed_fc'.format(name),
                                                          activation_fn=tf.nn.relu, reuse=reuse)
            point_feats = tf.contrib.layers.batch_norm(point_feats,scale=True,is_training=is_training,
                                                       scope='{}_out_bn'.format(name))
            ops['{}_ofeats'.format(name)]=point_feats

            return point_feats


def ecd_stage_test(stage_idx, xyzs, dxyzs, feats, xyz_param, feats_params, embed_dims, final_dim, radius,
                   sxyz_scale, dxyz_scale, is_training, reuse):
    with tf.name_scope('stage_{}'.format(stage_idx)):
        nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,radius)
        sxyzs=neighbor_ops.neighbor_scatter(xyzs,nidxs,nlens,nbegs,use_diff=True)
        sxyzs*=sxyz_scale
        xyz_feats=ecd_xyz_test(sxyzs, xyz_param[0], xyz_param[1], xyz_param[2], xyz_param[3], xyz_param[4],
                               nlens, nbegs, ncens, '{}_xyz'.format(stage_idx), is_training, reuse)
        cfeats=tf.concat([feats,xyz_feats],axis=1)
        for fi,fp in enumerate(feats_params):
            ecd_feats_val=ecd_feats_test(sxyzs,cfeats,fp[0],fp[1],fp[2],fp[3],nidxs,nlens,nbegs,ncens,
                                         '{}_{}_feats'.format(stage_idx,fi),is_training, reuse)
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


def pgnet_model_v6_test(xyzs, dxyzs, feats, vlens, vbegs, vcens, is_training, radius=(0.15,0.3,0.5), reuse=False):
    with tf.name_scope('base_graph_conv_edge_net'):
        with tf.variable_scope('base_graph_conv_edge_net',reuse=reuse):
            with tf.name_scope('conv_stage0'):
                fc0,lf0=ecd_stage_test(0,xyzs[0],dxyzs[0],feats,
                                     # feats_dims, final_feats_dim, diffusion_dims, trans_dims, out_dim
                                     [[8,8],16,[8,8],[8,8],32],
                                     [
                                         # embed_dim, diffusion_dims, trans_dims, out_dim
                                         [16,[8,8],[8,8],32],
                                         [16,[8,8],[8,8],32],
                                     ],
                                     [16,16,16], 128,radius[0], 3.0/0.15,3.0/0.15, is_training, reuse)

                fc0_pool = graph_max_pool_stage(0, fc0, vlens[0], vbegs[0])
                lf0_avg = graph_avg_pool_stage(0, feats, vlens[0], vbegs[0], vcens[0])
                ifeats_0 = tf.concat([lf0_avg, fc0_pool], axis=1)

            with tf.name_scope('conv_stage1'):
                fc1,lf1=ecd_stage_test(1, xyzs[1], dxyzs[1], ifeats_0,
                                     # feats_dims, final_feats_dim, diffusion_dims, trans_dims, out_dim
                                     [[16,16],32,[16,16],[16,16],32],
                                     [
                                         # embed_dim, diffusion_dims, trans_dims, out_dim
                                         [32,[16,16],[16,16],32],
                                         [32,[16,16],[16,16],32],
                                         [32,[16,16],[16,16],32],
                                     ],
                                     [32,32,32], 256,radius[1], 3.0/0.3,3.0/0.45, is_training, reuse)

                fc1_pool = graph_max_pool_stage(1, fc1, vlens[1], vbegs[1])
                lf1_avg = graph_avg_pool_stage(1, lf0_avg, vlens[1], vbegs[1], vcens[1])
                ifeats_1 = tf.concat([fc1_pool,lf1_avg],axis=1)

            with tf.name_scope('conv_stage2'):
                fc2,lf2=ecd_stage_test(2, xyzs[2], xyzs[2], ifeats_1,
                                     # feats_dims, final_feats_dim, diffusion_dims, trans_dims, out_dim
                                     [[16,16],32,[16,16],[16,16],32],
                                     [
                                         # embed_dim, diffusion_dims, trans_dims, out_dim
                                         [48,[16,16],[16,16],48],
                                         [48,[16,16],[16,16],48],
                                         [48,[16,16],[16,16],48],
                                     ],
                                     [64,64,64,128], 512,radius[2], 1.0, 1.0, is_training, reuse)
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

            # ops=[lf0,fc0,lf1,fc1,lf2,fc2]

    return upf0, lf, #ops


def build_model():

    pls={}
    pls['xyzs'] = tf.placeholder(tf.float32, [None, 3], 'xyzs')
    pls['feats'] = tf.placeholder(tf.float32, [None, 3], 'feats')
    pls['logits_grad'] = tf.placeholder(tf.float32, [None, 13], 'logits_grad')
    pls['labels'] = tf.placeholder(tf.int32, [None], 'labels')
    pls['is_training'] = tf.placeholder(tf.bool, name='is_training')


    xyzs, dxyzs, feats, labels, vlens, vbegs, vcens = \
        points_pooling_two_layers(pls['xyzs'],pls['feats'],pls['labels'],voxel_size1=0.15,voxel_size2=0.45,block_size=3.0)
    global_feats, local_feats = pgnet_model_v6_test(xyzs, dxyzs, feats, vlens, vbegs, vcens, pls['is_training'], [0.15, 0.3, 0.9], False)

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

def sample_feats(name,sess,pls,fns,ops,fs_num=3,block_num=10):
    feats=[]
    random.shuffle(fns)
    for fn in fns[:fs_num]:
        xyzs,rgbs,covars,labels,block_mins=read_pkl(fn)
        for i in xrange(len(xyzs[:block_num])):
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


if __name__=="__main__":
    sess,pls=build_model()
    train_list,test_list=get_block_train_test_split()
    random.shuffle(test_list)
    test_list=['data/S3DIS/sampled_test_new/'+fn for fn in test_list]

    saver=tf.train.Saver()
    saver.restore(sess,'model/pgnet_v6/model92.ckpt')

    names=['0_xyz_ofeats','0_0_feats_ofeats','0_1_feats_ofeats',
           '1_xyz_ofeats','1_0_feats_ofeats','1_1_feats_ofeats','1_2_feats_ofeats',
           '2_xyz_ofeats', '2_0_feats_ofeats', '2_1_feats_ofeats', '2_2_feats_ofeats',]

    l1_internal_names=['0_xyz_weights','0_0_feats_weights','0_1_feats_weights',
                       '0_xyz_feats', '0_0_feats_feats', '0_1_feats_feats',]
    l2_internal_names=['1_xyz_weights','1_0_feats_weights','1_1_feats_weights',
                       '1_xyz_feats', '1_0_feats_feats', '1_1_feats_feats',]
    for name in names:
        feats=sample_feats(name,sess,pls,test_list,ops)
        var=np.var(feats, axis=0)
        for k in xrange(feats.shape[1]):
            draw_hist(feats[:,k],name+'_{}'.format(k))
            print '{} {} {}'.format(name,k,var[k])

