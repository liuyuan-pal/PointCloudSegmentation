import argparse
import time
import numpy as np
import os

import tensorflow as tf
from model_pointnet_semantic3d import *
from model_pooling import *
from train_util import *
from io_util import get_semantic3d_block_train_list,read_pkl,get_semantic3d_class_names,read_fn_hierarchy,\
    save_pkl,get_semantic3d_block_train_test_list
from provider import Provider,default_unpack_feats_labels
from draw_util import output_points,get_semantic3d_class_colors
from functools import partial

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=4, help='')
    parser.add_argument('--batch_size', type=int, default=1, help='')

    parser.add_argument('--lr_init', type=float, default=1e-3, help='')
    parser.add_argument('--lr_clip', type=float, default=1e-5, help='')
    parser.add_argument('--decay_rate', type=float, default=0.5, help='')
    parser.add_argument('--decay_epoch', type=int, default=20, help='')
    parser.add_argument('--num_classes', type=int, default=8, help='')

    parser.add_argument('--restore',type=bool, default=False, help='')
    parser.add_argument('--restore_epoch', type=int, default=0, help='')
    parser.add_argument('--restore_model', type=str, default='', help='')

    parser.add_argument('--log_step', type=int, default=240, help='')
    parser.add_argument('--train_dir', type=str, default='train/pointnet_13_dilate_embed_semantic3d_dense', help='')
    parser.add_argument('--save_dir', type=str, default='model/pointnet_13_dilate_embed_semantic3d_dense', help='')
    parser.add_argument('--log_file', type=str, default='pointnet_13_dilate_embed_semantic3d_dense.log', help='')

    parser.add_argument('--eval',type=bool, default=False, help='')
    parser.add_argument('--eval_model',type=str, default='model/label/unsupervise80.ckpt',help='')
    parser.add_argument('--eval_output',type=bool, default=False,help='')

    parser.add_argument('--train_epoch_num', type=int, default=500, help='')

    FLAGS = parser.parse_args()

    return FLAGS
# training set labels count
# [ 69821678.  38039379.  52260419.  54975958.  10229433.  67044610.
#    7620115.   3612462.   2362445.]


train_weights=np.asarray([0.0,3.5545287,3.211912,3.0997152,4.7834935,2.7914784,4.982284,5.260409,5.321266])

def tower_loss(xyzs, feats, labels, nidxs, nlens, nbegs, ncens, idxs, is_training,reuse=False):

    with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
        xyzs, feats, labels = dense_feats(xyzs,feats,labels,idxs,nidxs,nlens,nbegs,ncens,reuse)
        feats_ops['dense_feats']=feats
        xyzs, dxyzs, feats, labels, vlens, vbegs, vcens = \
            points_pooling_two_layers(xyzs,feats,labels, 0.45, 1.5, 10.0)
        # global_feats,local_feats=pointnet_10_concat_embed_semantic3d(xyzs,dxyzs,feats,vlens,vbegs,vcens,reuse)
        global_feats,local_feats=pointnet_13_dilate_embed_semantic3d(xyzs,dxyzs,feats,vlens,vbegs,vcens,reuse)

        global_feats_exp=tf.expand_dims(global_feats,axis=0)
        local_feats_exp=tf.expand_dims(local_feats,axis=0)
        logits=classifier_v3(global_feats_exp, local_feats_exp, is_training,
                             FLAGS.num_classes, reuse, use_bn=False)  # [1,pn,num_classes]

    flatten_logits=tf.reshape(logits,[-1,FLAGS.num_classes])  # [pn,num_classes]
    feats_ops['logits']=flatten_logits
    labels_flatten=tf.reshape(labels,[-1,1])                  # [pn,1]
    labels_flatten=tf.squeeze(labels_flatten,axis=1)          # [pn]
    train_weights_tf=tf.constant(train_weights,dtype=tf.float32,name='train_weights')
    weights=tf.gather(train_weights_tf,labels_flatten)        # assign 0 weights for unknown

    comparison=tf.equal(labels_flatten,tf.constant(0,dtype=tf.int64,name='zero_constant'))
    labels_flatten=tf.where(comparison, tf.ones_like(labels_flatten), labels_flatten)
    labels_flatten-=1
    loss=tf.losses.sparse_softmax_cross_entropy(labels_flatten,flatten_logits,weights=weights)
    tf.summary.scalar(loss.op.name,loss)

    _,rgbs=tf.split(feats,[48,4],axis=1)

    return loss,flatten_logits,labels_flatten,comparison,xyzs[0],rgbs


def train_ops(xyzs, feats, labels, nidxs, nlens, nbegs, ncens, idxs, is_training, epoch_batch_num):
    ops={}
    with tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        decay_steps=epoch_batch_num*FLAGS.decay_epoch
        lr=tf.train.exponential_decay(FLAGS.lr_init,global_step,decay_steps,FLAGS.decay_rate,staircase=True)
        lr=tf.maximum(FLAGS.lr_clip,lr)
        tf.summary.scalar('learning rate',lr)

        opt=tf.train.AdamOptimizer(lr)

        reuse=False
        tower_grads=[]
        tower_losses=[]
        tower_logits=[]
        tower_labels=[]
        tower_masks=[]
        tower_xyzs=[]
        tower_rgbs=[]
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:{}'.format(i%4)):
                with tf.name_scope('tower_{}'.format(i)):
                    loss,logits,label,mask,coords,rgbs=tower_loss(
                        xyzs[i], feats[i], labels[i], nidxs[i], nlens[i], nbegs[i], ncens[i], idxs[i], is_training, reuse)

                    grad=opt.compute_gradients(loss)
                    tower_grads.append(grad)
                    tower_losses.append(loss)
                    tower_logits.append(logits)
                    tower_labels.append(label)
                    tower_masks.append(mask)
                    tower_xyzs.append(coords)
                    tower_rgbs.append(rgbs)
                    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
                    reuse=True

        avg_grad=average_gradients(tower_grads)

        with tf.control_dependencies(update_op):
            apply_grad_op=tf.group(opt.apply_gradients(avg_grad,global_step=global_step))
        summary_op = tf.summary.merge(summaries)

        total_loss_op=tf.add_n(tower_losses)/FLAGS.num_gpus
        tower_labels=tf.concat(tower_labels,axis=0)
        tower_masks=tf.concat(tower_masks,axis=0)
        tower_xyzs=tf.concat(tower_xyzs,axis=0)
        tower_rgbs=tf.concat(tower_rgbs,axis=0)

        logits_op=tf.concat(tower_logits,axis=0)
        preds_op=tf.argmax(logits_op,axis=1)
        correct_num_op=tf.reduce_sum(tf.cast(tf.equal(preds_op,tower_labels),tf.float32))

        ops['total_loss']=total_loss_op
        ops['apply_grad']=apply_grad_op
        ops['logits']=logits_op
        ops['preds']=preds_op
        ops['correct_num']=correct_num_op
        ops['summary']=summary_op
        ops['global_step']=global_step
        ops['labels']=tower_labels
        ops['learning_rate']=lr
        ops['masks']=tower_masks
        ops['xyzs']=tower_xyzs
        ops['rgbs']=tower_rgbs

    return ops


def train_one_epoch(ops,pls,sess,summary_writer,trainset,epoch_num,feed_dict):
    epoch_begin=time.time()
    total_correct,total_block,total_points=0,0,0
    begin_time=time.time()
    total_losses=[]
    for i,feed_in in enumerate(trainset):
        fill_feed_dict(feed_in,feed_dict,pls,FLAGS.num_gpus)

        feed_dict[pls['is_training']]=True
        total_block+=FLAGS.num_gpus

        _,loss_val,logits,labels,masks,lr=\
            sess.run([ops['apply_grad'],ops['total_loss'],ops['logits'],ops['labels'],ops['masks'],ops['learning_rate']],feed_dict)

        masks=masks==0
        total_losses.append(loss_val)
        preds=np.argmax(logits,axis=1)[masks]
        labels=labels[masks]
        total_correct+=np.sum(labels==preds)
        total_points+=np.sum(masks)

        if i % FLAGS.log_step==0:
            summary,global_step=sess.run(
                [ops['summary'],ops['global_step']],feed_dict)

            log_str('epoch {} step {} loss {:.5} acc {:.5} | {:.5} examples/s lr {:.5}'.format(
                epoch_num,i,np.mean(np.asarray(total_losses)),
                float(total_correct+1)/(total_points+1),
                float(total_block)/(time.time()-begin_time),
                lr
            ),FLAGS.log_file)

            summary_writer.add_summary(summary,global_step)
            total_correct,total_block,total_points=0,0,0
            begin_time=time.time()
            total_losses=[]

        # the generated training set is too large
        # so every 20000 examples we test once and save the model
        if i*FLAGS.num_gpus>12800:
            break

    log_str('epoch {} cost {} s'.format(epoch_num, time.time()-epoch_begin), FLAGS.log_file)

colors_ground=[
    [128,128,128],# other
    [0,255,0],   # correct 0
    [0,255,255], # correct 1
    [255,0,0],   # error 1_0
    [255,165,0]  # error 0_1
]
colors_ground=np.asarray(colors_ground)
def test_one_epoch(ops,pls,sess,saver,testset,epoch_num,feed_dict,summary_writer=None):
    begin_time=time.time()
    test_loss=[]
    all_preds,all_labels=[],[]
    for i,feed_in in enumerate(testset):
        _,lens,block_mins=fill_feed_dict(feed_in,feed_dict,pls,FLAGS.num_gpus)

        feed_dict[pls['is_training']] = False

        loss,logits,labels,masks,xyzs,rgbs=sess.run(
            [ops['total_loss'],ops['logits'],ops['labels'],
             ops['masks'],ops['xyzs'],ops['rgbs']],feed_dict)

        masks=masks==0
        all_labels.append(labels[masks])
        preds=np.argmax(logits,axis=1)
        all_preds.append(preds[masks])
        test_loss.append(loss)

        if FLAGS.eval and FLAGS.eval_output:
            cur_loc=0
            for j in xrange(FLAGS.num_gpus):
                cur_labels=labels[cur_loc:cur_loc+lens[j]]
                cur_xyzs=xyzs[cur_loc:cur_loc+lens[j]]
                cur_preds=preds[cur_loc:cur_loc+lens[j]]
                cur_masks=masks[cur_loc:cur_loc+lens[j]]
                cur_rgbs=rgbs[cur_loc:cur_loc+lens[j]]*127+128

                cur_masks=np.asarray(cur_masks, dtype=np.bool)
                error_0_1 = np.logical_and(np.logical_and(cur_labels==0, cur_preds==1),cur_masks)
                error_1_0 = np.logical_and(np.logical_and(cur_labels==1, cur_preds==0),cur_masks)
                correct_0 = np.logical_and(np.logical_and(cur_labels==0, cur_preds==0),cur_masks)
                correct_1 = np.logical_and(np.logical_and(cur_labels==1, cur_preds==1),cur_masks)
                if np.sum(np.logical_or(error_0_1, error_1_0))==0:
                    cur_loc+=lens[j]
                    continue

                error_num=np.sum(np.logical_or(error_0_1, error_1_0))
                cur_colors=np.zeros(lens[j],dtype=np.int32)
                cur_colors[error_1_0]=3
                cur_colors[error_0_1]=4
                cur_colors[correct_0]=1
                cur_colors[correct_1]=2

                output_points('test_result/all/{}_{}.txt'.format(i,j),
                              cur_xyzs+block_mins[j],get_semantic3d_class_colors()[cur_preds+1])
                output_points('test_result/lbl/{}_{}_{}.txt'.format(i,j,error_num),
                              cur_xyzs+block_mins[j],colors_ground[cur_colors])
                output_points('test_result/rgb/{}_{}.txt'.format(i,j),
                              cur_xyzs+block_mins[j],cur_rgbs[:,:3])
                cur_loc += lens[j]


    all_preds=np.concatenate(all_preds,axis=0)
    all_labels=np.concatenate(all_labels,axis=0)
    iou, miou, oiou, acc, macc, oacc = compute_iou(all_labels,all_preds,8)

    test_loss=np.mean(np.asarray(test_loss))
    log_str('mean iou {:.5} overall iou {:5} loss {:5} \n mean acc {:5} overall acc {:5} cost {:3} s'.format(
        miou, oiou, test_loss, macc, oacc, time.time()-begin_time
    ),FLAGS.log_file)

    if not FLAGS.eval:
        checkpoint_path = os.path.join(FLAGS.save_dir, 'model{}.ckpt'.format(epoch_num))
        saver.save(sess,checkpoint_path)
    else:
        names=get_semantic3d_class_names()[1:]
        for i in xrange(len(names)):
            print '{} iou {} acc {}'.format(names[i],iou[i],acc[i])

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
def draw_hist(feats,name):
    plt.figure(0)
    plt.hist(feats)
    plt.savefig('test_result/{}_hist.png'.format(name))
    plt.close()

def sample_hist(pls,sess,testset,feed_dict,sample_num=20):
    ops,names=[],[] #[feats_ops['logits']],['logits']
    for k,v in feats_ops.iteritems():
        ops.append(v)
        names.append(k)

    feats_count=len(names)
    feats=[[] for _ in xrange(feats_count)]

    for i,feed_in in enumerate(testset):
        fill_feed_dict(feed_in,feed_dict,pls,FLAGS.num_gpus)
        feed_dict[pls['is_training']] = False
        cur_feats=sess.run(ops,feed_dict)

        for k in xrange(feats_count):
            feats[k].append(cur_feats[k])

        if i>sample_num:
            break

    for i in xrange(feats_count):
        cur_feats=np.concatenate(feats[i],axis=0)
        for k in xrange(cur_feats.shape[1]):
            draw_hist(cur_feats[:,k],'{}_{}'.format(names[i],k))

def fill_feed_dict(feed_in,feed_dict,pls,num_gpus):
    xyzs, rgbs, lbls, nidxs, nlens, nbegs, ncens, idxs, block_mins = \
        default_unpack_feats_labels(feed_in, num_gpus)

    batch_pt_num=0
    for k in xrange(num_gpus):
        feed_dict[pls['xyzs'][k]]=xyzs[k]
        feed_dict[pls['feats'][k]]=rgbs[k]
        feed_dict[pls['lbls'][k]]=lbls[k]

        feed_dict[pls['nidxs'][k]]=nidxs[k]
        feed_dict[pls['nlens'][k]]=nlens[k]
        feed_dict[pls['nbegs'][k]]=nbegs[k]
        feed_dict[pls['ncens'][k]]=ncens[k]
        feed_dict[pls['idxs'][k]]=idxs[k]

        batch_pt_num += idxs[k].shape[0]

    lens=[len(idxs[k]) for k in xrange(num_gpus)]

    return batch_pt_num,lens,block_mins

def build_placeholder(num_gpus):
    pls = {}
    pls['xyzs'], pls['feats'], pls['lbls'],=[],[],[]
    pls['nidxs'],pls['nlens'],pls['nbegs'],pls['ncens'],pls['idxs']=[],[],[],[],[]
    for i in xrange(num_gpus):
        pls['xyzs'].append(tf.placeholder(tf.float32,[None,3],'xyzs{}'.format(i)))
        pls['feats'].append(tf.placeholder(tf.float32,[None,4],'feats{}'.format(i)))
        pls['lbls'].append(tf.placeholder(tf.int32,[None],'lbls{}'.format(i)))

        pls['nidxs'].append(tf.placeholder(tf.int32,[None],'nidxs{}'.format(i)))
        pls['nlens'].append(tf.placeholder(tf.int32,[None],'nlens{}'.format(i)))
        pls['nbegs'].append(tf.placeholder(tf.int32,[None],'nbegs{}'.format(i)))
        pls['ncens'].append(tf.placeholder(tf.int32,[None],'ncens{}'.format(i)))
        pls['idxs'].append(tf.placeholder(tf.int32,[None],'idxs{}'.format(i)))

    pls['is_training'] = tf.placeholder(tf.bool, name='is_training')
    return pls


from aug_util import swap_xy,flip,compute_nidxs_bgs,compute_cidxs
import random
import libPointUtil
def train_fn(model,fn):
    xyzs, rgbs, lbls, block_mins, block_nidxs, block_nlens, block_nbegs =read_pkl(fn)

    nidxs,nlens,nbegs,ncens=[],[],[],[]
    ds_idxs=[]
    in_xyzs, in_rgbs, in_lbls=[],[],[]
    for i in xrange(len(xyzs)):
        # center idxs
        idxs=libPointUtil.gridDownsampleGPU(xyzs[i], 0.15, False)
        pt_num=len(idxs)
        if pt_num<256 and model=='train': continue

        if pt_num>4096:
            np.random.shuffle(idxs)
            ratio=np.random.uniform(0.8,1.0) if model=='train' else 1.0
            idxs=idxs[:min(int(pt_num*ratio),20480)]

        ds_idxs.append(idxs)

        # neighbor idxs
        cur_nidxs=[]
        cur_nlens=[]
        for id in idxs:
            cur_nidxs.append(block_nidxs[i][block_nbegs[i][id]:block_nbegs[i][id]+block_nlens[i][id]])
            cur_nlens.append(block_nlens[i][id])

        cur_nbegs=compute_nidxs_bgs(cur_nlens)
        cur_ncens=compute_cidxs(cur_nlens)

        cur_nidxs=np.concatenate(cur_nidxs,axis=0)
        nidxs.append(cur_nidxs)
        nlens.append(cur_nlens)
        nbegs.append(cur_nbegs)
        ncens.append(cur_ncens)

        # data aug
        if model=='train':
            if random.random()<0.5:
                xyzs[i]=flip(xyzs[i],axis=0)

            if random.random()<0.5:
                xyzs[i]=flip(xyzs[i],axis=1)

            if random.random()<0.5:
                xyzs[i]=swap_xy(xyzs[i])

            jitter_color=np.random.uniform(-0.02,0.02,rgbs[i].shape)
            rgbs[i]+=jitter_color
            rgbs[i][:,:3][rgbs[i][:,:3]>1.0]=1.0
            rgbs[i][:,:3][rgbs[i][:,:3]<-1.0]=-1.0

        in_xyzs.append(xyzs[i])
        in_rgbs.append(rgbs[i])
        in_lbls.append(lbls[i])

    return in_xyzs, in_rgbs, in_lbls, nidxs, nlens, nbegs, ncens, ds_idxs, block_mins

def train():
    test_set=['sg27_station4_intensity_rgb','bildstein_station1_xyz_intensity_rgb']
    train_list,test_list=get_semantic3d_block_train_test_list(test_set)
    train_list=['/data/Semantic3D.Net/'+fn for fn in train_list]
    test_list=['/data/Semantic3D.Net/'+fn for fn in test_list]

    train_provider = Provider(train_list,'train',FLAGS.batch_size*FLAGS.num_gpus,train_fn)
    test_provider = Provider(test_list,'test',FLAGS.batch_size*FLAGS.num_gpus,train_fn)

    try:
        pls=build_placeholder(FLAGS.num_gpus)

        batch_num_per_epoch=5000/FLAGS.num_gpus
        ops=train_ops(pls['xyzs'],pls['feats'],pls['lbls'],pls['nidxs'],
                      pls['nlens'],pls['nbegs'],pls['ncens'],pls['idxs'],
                      pls['is_training'],batch_num_per_epoch)

        feed_dict={}
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        saver = tf.train.Saver(max_to_keep=500)
        sess.run(tf.global_variables_initializer())
        if FLAGS.restore:
              saver.restore(sess,FLAGS.restore_model)
        else:
            sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir,graph=sess.graph)

        for epoch_num in xrange(FLAGS.restore_epoch,FLAGS.train_epoch_num):
            train_one_epoch(ops,pls,sess,summary_writer,train_provider,epoch_num,feed_dict)
            test_one_epoch(ops,pls,sess,saver,test_provider,epoch_num,feed_dict)

    finally:
        train_provider.close()
        test_provider.close()

def sample():
    test_set=['sg27_station4_intensity_rgb','bildstein_station1_xyz_intensity_rgb']
    train_list,test_list=get_semantic3d_block_train_test_list(test_set)
    train_list=['/data/Semantic3D.Net/'+fn for fn in train_list]
    test_list=['/data/Semantic3D.Net/'+fn for fn in test_list]

    train_provider = Provider(train_list,'train',FLAGS.batch_size*FLAGS.num_gpus,train_fn)
    test_provider = Provider(test_list,'test',FLAGS.batch_size*FLAGS.num_gpus,train_fn)

    try:
        pls=build_placeholder(FLAGS.num_gpus)

        batch_num_per_epoch=5000/FLAGS.num_gpus
        ops=train_ops(pls['xyzs'],pls['feats'],pls['lbls'],pls['nidxs'],
                      pls['nlens'],pls['nbegs'],pls['ncens'],pls['idxs'],
                      pls['is_training'],batch_num_per_epoch)

        feed_dict={}
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        saver = tf.train.Saver(max_to_keep=500)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,FLAGS.restore_model)
        sample_hist(pls,sess,test_provider,feed_dict,50)

    finally:
        train_provider.close()
        test_provider.close()


def eval():
    test_set=['sg27_station4_intensity_rgb','bildstein_station1_xyz_intensity_rgb']
    train_list,test_list=get_semantic3d_block_train_test_list(test_set)
    test_list=['/data/Semantic3D.Net/'+fn for fn in test_list]

    test_provider = Provider(test_list,'test',FLAGS.batch_size*FLAGS.num_gpus,train_fn)

    try:
        pls=build_placeholder(FLAGS.num_gpus)

        batch_num_per_epoch=5000/FLAGS.num_gpus
        ops=train_ops(pls['xyzs'],pls['feats'],pls['lbls'],pls['nidxs'],
                      pls['nlens'],pls['nbegs'],pls['ncens'],pls['idxs'],
                      pls['is_training'],batch_num_per_epoch)

        feed_dict={}
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        saver = tf.train.Saver(max_to_keep=500)
        saver.restore(sess,FLAGS.eval_model)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir,graph=sess.graph)
        test_one_epoch(ops,pls,sess,saver,test_provider,0,feed_dict,summary_writer)

    finally:
        test_provider.close()

def test_io():
    test_set=['sg27_station4_intensity_rgb','bildstein_station1_xyz_intensity_rgb']
    train_list,test_list=get_semantic3d_block_train_test_list(test_set)
    train_list=['data/Semantic3D.Net/block/sampled_dense/'+fn for fn in train_list]
    test_list=['data/Semantic3D.Net/block/sampled_dense/'+fn for fn in test_list]

    for fs in train_list[:1]:
        xyzs, rgbs, lbls, nidxs, nlens, nbegs, ncens, ds_idxs = train_fn('train',fs)
        for i in xrange(len(xyzs)):
            assert nlens[i][-1]+nbegs[i][-1]==len(nidxs[i])
            for j in xrange(len(ds_idxs[i])):
                idxs=nidxs[i][nbegs[i][j]:nbegs[i][j]+nlens[i][j]]
                diff=np.expand_dims(xyzs[i][ds_idxs[i][j]],axis=0)-xyzs[i][idxs,:]
                dist=np.max(np.sqrt(np.sum(diff**2,axis=1)))
                assert np.sum(dist>0.25)==0



if __name__=="__main__":
    FLAGS=parser()
    if FLAGS.eval: eval()
    else: train()
    # sample()