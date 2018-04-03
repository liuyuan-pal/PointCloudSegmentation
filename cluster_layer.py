from model_pooling import *
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def graph_conv_pool_block_edge_xyz_cluster(sxyzs, stage_idx, gxyz_dim, ncens, nidxs, nlens, nbegs, reuse):
    # feats = tf.contrib.layers.fully_connected(sxyzs, num_outputs=gxyz_dim, scope='{}_xyz_fc'.format(stage_idx),
    #                                           activation_fn=tf.nn.relu, reuse=reuse)
    xyz_gc=graph_conv_edge_xyz(sxyzs, 3, [gxyz_dim/2, gxyz_dim/2], gxyz_dim, nidxs, nlens, nbegs, ncens,
                               '{}_xyz_gc'.format(stage_idx),reuse=reuse)
    return xyz_gc

def graph_conv_pool_stage_cluster(stage_idx, xyzs, dxyz, feats, feats_dim, gxyz_dim, gc_dims, gfc_dims, final_dim, radius, reuse):
    ops=[]
    with tf.name_scope('stage_{}'.format(stage_idx)):
        nidxs,nlens,nbegs,ncens=search_neighborhood(xyzs,radius)

        sxyzs = neighbor_ops.neighbor_scatter(xyzs, nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
        xyz_gc=graph_conv_pool_block_edge_xyz_cluster(sxyzs,stage_idx,gxyz_dim,ncens,nidxs,nlens,nbegs,reuse)
        ops.append(xyz_gc)

        cfeats = tf.concat([xyz_gc, feats], axis=1)

        cdim = feats_dim + gxyz_dim
        conv_fn = partial(graph_conv_pool_block_edge_new, ncens=ncens, nidxs=nidxs, nlens=nlens, nbegs=nbegs, reuse=reuse)

        layer_idx = 1
        for gd in gc_dims:
            conv_feats = conv_fn(sxyzs, cfeats, stage_idx, layer_idx, gd)
            ops.append(conv_feats)

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

    return fc_final, cfeats, ops  # cfeats: [pn,fc_dims+gxyz_dim+feats_dim]

def graph_conv_model(xyzs, dxyzs, pxyzs, feats, vlens, vbegs, vcens, reuse=False):
    with tf.name_scope('base_graph_conv_edge_net'):
        with tf.variable_scope('base_graph_conv_edge_net',reuse=reuse):
            with tf.name_scope('conv_stage0'):
                # 8 64 64*2
                fc0, lf0, ops1 = graph_conv_pool_stage_cluster(0,xyzs,dxyzs,feats,tf.shape(feats)[1],radius=0.1,reuse=reuse,
                                                               gxyz_dim=8,gc_dims=[16,16,16,16],gfc_dims=[64,64,64],final_dim=64)
                fc0_pool = graph_pool_stage(0, fc0, vlens, vbegs)

            with tf.name_scope('conv_stage1'):
                # 16 288 512*2
                fc1, lf1, ops2 = graph_conv_pool_stage_cluster(1,pxyzs,pxyzs,fc0_pool,64,radius=0.5,reuse=reuse,
                                                               gxyz_dim=16,gc_dims=[32,32,32,64,64,64],gfc_dims=[256,256,256],final_dim=512)
                fc1_pool = tf.reduce_max(fc1, axis=0)

            with tf.name_scope('unpool_stage1'):
                upfeats1 = tf.tile(tf.expand_dims(fc1_pool, axis=0), [tf.shape(fc1)[0], 1])
                upf1 = tf.concat([upfeats1, fc1, lf1], axis=1)
            with tf.name_scope('unpool_stage0'):
                upfeats0 = graph_unpool_stage(0, upf1, vlens, vbegs, vcens)
                upf0 = tf.concat([upfeats0, fc0, lf0], axis=1)

            lf = tf.concat([fc0, lf0], axis=1)


            ops2=[graph_unpool_stage(1+idx, op, vlens, vbegs, vcens) for idx,op in enumerate(ops2)]
            ops1+=ops2
    # 1528 + 132
    return upf0, lf, ops1


def build_model(model=graph_conv_model):
    xyzs=tf.placeholder(tf.float32,[None,3],'xyzs')
    feats=tf.placeholder(tf.float32,[None,12],'feats')
    labels=tf.placeholder(tf.int32,[None],'labels')
    logits_grad=tf.placeholder(tf.float32,[None,13],'logits_grad')
    pls={'xyzs':xyzs,'feats':feats,'labels':labels,'logits_grad':logits_grad}
    xyzs, pxyzs, dxyzs, feats, labels, vlens, vbegs, vcens = \
        points_pooling(xyzs, feats, labels, voxel_size=0.3, block_size=3.0)
    global_feats,local_feats,ops=model(xyzs, dxyzs, pxyzs, feats, vlens, vbegs, vcens, voxel_size=0.3, block_size=3.0)

    global_feats_exp = tf.expand_dims(global_feats, axis=0)
    local_feats_exp = tf.expand_dims(local_feats, axis=0)
    logits=classifier_v3(global_feats_exp, local_feats_exp, tf.Variable(False,False,dtype=tf.bool), 13)
    feats_grad=tf.gradients(logits,global_feats,logits_grad)[0]

    return ops,pls,feats_grad

from io_util import read_pkl,get_block_train_test_split
import numpy as np
import random
def load_data():
    train_list,test_list=get_block_train_test_split()
    train_list=['data/S3DIS/sampled_train/'+fn for fn in train_list]

    def fn(model,filename):
        data=read_pkl(filename)
        return data[0],data[2],data[3],data[4],data[12]

    random.shuffle(train_list)
    all_xyzs,all_feats,all_labels=[],[],[]
    for fs in train_list[:20]:
        xyzs,rgbs,covars,labels,block_mins=fn('',fs)
        for i in xrange(3):
            all_xyzs.append(xyzs[i][0])
            feats=np.concatenate([rgbs[i],covars[i]],axis=1)
            all_feats.append(feats)
            all_labels.append(labels[i])

    return all_xyzs,all_feats,all_labels

def cluster():
    from sklearn.cluster import KMeans
    from draw_util import output_points
    ops,pls,feats_grad=build_model()
    xyzs,feats,labels=load_data()

    all_feats=tf.concat(ops,axis=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    saver=tf.train.Saver(tf.trainable_variables())
    saver.restore(sess,'model/gpn_edge_new_v2/model23.ckpt')

    kmeans=KMeans(n_clusters=5,n_jobs=-1)
    colors=np.random.randint(0,256,[5,3])
    for i in xrange(1):
        all_layer_feats=[]
        for k in xrange(len(xyzs)):
            layer_feats=sess.run(all_feats,feed_dict={
                pls['xyzs']:xyzs[k],
                pls['feats']:feats[k],
                pls['labels']:labels[k],
            })
            all_layer_feats.append(layer_feats)

        all_layer_feats=np.concatenate(all_layer_feats,axis=0)
        print i,np.max(all_layer_feats,axis=0),np.mean(all_layer_feats)
        preds=kmeans.fit_predict(all_layer_feats)
        cur_loc=0
        for t in xrange(2):
            output_points('test_result/{}_{}_preds.txt'.format(i,t),xyzs[t],colors[preds[cur_loc:cur_loc+len(xyzs[t])],:])
            output_points('test_result/{}_{}_colors.txt'.format(i,t),xyzs[t],feats[t][:,:3]*127+128)

def gradient(model):
    # [fc1pool,fc1,lf1,fc0,lf0]
    # lf1:[fc0pool,gxyz1,gc1_0,...]
    # lf2:[feats,gxyz0,gc0_0,...]
    dims=[512,  # fc1pool
          512,  # fc1
          # lf1
          16,   # gxyz1
          128,   # fc0pool
          32,32,32,32,32,32, # gc1
          128,   # fc0
          # lf0
          16,    # gxyz0
          12,   # feats
          16,16,16,16,16,16]      # gc0
    names=['gfc1','fc1','gxyz1','gfc0']
    names+=['gc1_{}'.format(k) for k in range(6)]
    names+=['fc0','gxyz0','feats']
    names+=['gc0_{}'.format(k) for k in range(6)]

    ops,pls,feats_grad=build_model(model)
    xyzs,feats,labels=load_data()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    saver=tf.train.Saver(tf.trainable_variables())
    saver.restore(sess,'model/gpn_edge_simp/model25.ckpt')

    for i in xrange(13):
        all_feats_grad_val=[]
        for k in xrange(len(xyzs)):
            grads = np.zeros([len(xyzs[k]),13],np.float32)
            grads[:,i] = 1.0
            feats_grad_val = sess.run(feats_grad, feed_dict={
                pls['xyzs']: xyzs[k],
                pls['feats']: feats[k],
                pls['labels']: labels[k],
                pls['logits_grad']:grads,
            })
            all_feats_grad_val.append(feats_grad_val)

        all_feats_grad_val=np.concatenate(all_feats_grad_val,axis=0)
        # print 'class {} grads distribution'.format(i)

        mean_val=np.mean(np.abs(all_feats_grad_val),axis=0)
        cur=0
        seg_val=[]
        seg_sum=[]
        for d in dims:
            seg_val.append(np.mean(mean_val[cur:cur+d]))
            seg_sum.append(np.sum(mean_val[cur:cur+d]))
            cur+=d
        seg_val=np.asarray(seg_val)
        seg_sum=np.asarray(seg_sum)

        print len(mean_val),np.sum(dims)
        assert len(mean_val)==np.sum(dims)

        # plt.figure(0)
        # plt.plot(np.arange(mean_val.shape[0]),mean_val,'-')
        # plt.savefig('test_result/class_{}.png'.format(i))
        # plt.close()

        plt.figure(0,figsize=(10,5))
        x=np.arange(len(dims))
        plt.xticks(x, names)
        plt.bar(x, seg_val)
        plt.savefig('test_result/class_{}_seg.png'.format(i))
        # plt.tight_layout()
        plt.close()
        plt.figure(0,figsize=(10,5))
        x=np.arange(len(dims))
        plt.xticks(x, names)
        plt.bar(x, seg_sum)
        plt.savefig('test_result/class_{}_seg_sum.png'.format(i))
        # plt.tight_layout()
        plt.close()

if __name__=="__main__":
    gradient(graph_conv_pool_edge_simp)