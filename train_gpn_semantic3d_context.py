import argparse
import time
import numpy as np
import os

import tensorflow as tf
from model_pooling import context_points_pooling_two_layers,graph_conv_pool_edge_simp_2layers_s3d,\
    graph_conv_pool_context,classifier_v3,graph_conv_pool_context_with_pool,context_points_pooling
from semantic3d_context_util import get_context_train_test
from train_util import *
from io_util import get_semantic3d_block_train_list,read_pkl,get_semantic3d_class_names,read_fn_hierarchy,\
    save_pkl,get_semantic3d_block_train_test_list
from provider import Provider,default_unpack_feats_labels
from draw_util import output_points,get_semantic3d_class_colors
from functools import partial

train_weights=np.asarray([0.0,3.5545287,3.211912,3.0997152,4.7834935,2.7914784,4.982284,5.260409,5.321266])

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, default=4, help='')
    parser.add_argument('--batch_size', type=int, default=1, help='')

    parser.add_argument('--lr_init', type=float, default=1e-3, help='')
    parser.add_argument('--lr_clip', type=float, default=1e-5, help='')
    parser.add_argument('--decay_rate', type=float, default=0.5, help='')
    parser.add_argument('--decay_epoch', type=int, default=50, help='')
    parser.add_argument('--num_classes', type=int, default=9, help='')

    parser.add_argument('--restore',type=bool, default=False, help='')
    parser.add_argument('--restore_epoch', type=int, default=0, help='')
    parser.add_argument('--restore_model', type=str, default='', help='')

    parser.add_argument('--log_step', type=int, default=240, help='')
    parser.add_argument('--train_dir', type=str, default='train/semantic_context_avg', help='')
    parser.add_argument('--save_dir', type=str, default='model/semantic_context_avg', help='')
    parser.add_argument('--log_file', type=str, default='semantic_context_avg.log', help='')

    parser.add_argument('--eval',type=bool, default=False, help='')
    parser.add_argument('--eval_model',type=str, default='model/label/unsupervise80.ckpt',help='')
    parser.add_argument('--eval_output',type=bool, default=False,help='')

    parser.add_argument('--train_epoch_num', type=int, default=500, help='')

    FLAGS = parser.parse_args()

    return FLAGS


def tower_loss(xyzs, feats, ctx_pts, ctx_idxs, labels, is_training,reuse=False):

    with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
        xyzs, dxyzs, feats, labels, vlens, vbegs, vcens, ctx_idxs = \
            context_points_pooling_two_layers(xyzs,feats,labels,ctx_idxs,voxel_size1=0.25,voxel_size2=1.0,block_size=10.0)
        global_feats,local_feats,ops=graph_conv_pool_edge_simp_2layers_s3d(xyzs, dxyzs, feats, vlens, vbegs, vcens,
                                                                           [0.25, 1.0], 10.0, [0.25,0.5,2.0], reuse)
        ctx_xyzs,ctx_feats=tf.split(ctx_pts,[3,13],axis=1)

        print ctx_xyzs,ctx_feats
        # ctx_graph_feats=graph_conv_pool_context(ctx_xyzs,ctx_feats,block_size=50,radius=4.0,reuse=reuse)
        ctx_xyzs,ctx_pxyzs,ctx_dxyzs,ctx_feats,ctx_vlens,ctx_vbegs,ctx_vcens,ctx_idxs=\
            context_points_pooling(ctx_xyzs,ctx_feats,ctx_idxs,5,300)
        ctx_graph_feats=graph_conv_pool_context_with_pool(ctx_xyzs,ctx_dxyzs,ctx_pxyzs,ctx_feats,ctx_vlens,ctx_vbegs,ctx_vcens,
                                                          voxel_size=5.0,block_size=50.0,radius1=5.0,radius2=15.0,reuse=reuse)
        ctx_graph_feats=tf.gather(ctx_graph_feats,ctx_idxs)

        global_feats=tf.concat([global_feats,ctx_graph_feats],axis=1)
        global_feats_exp=tf.expand_dims(global_feats,axis=0)
        local_feats_exp=tf.expand_dims(local_feats,axis=0)
        logits=classifier_v3(global_feats_exp, local_feats_exp, is_training,
                             FLAGS.num_classes, reuse, use_bn=False)  # [1,pn,num_classes]

    flatten_logits=tf.reshape(logits,[-1,FLAGS.num_classes])  # [pn,num_classes]
    labels_flatten=tf.reshape(labels,[-1,1])                  # [pn,1]
    labels_flatten=tf.squeeze(labels_flatten,axis=1)          # [pn]
    # weights=tf.cast(tf.not_equal(labels_flatten,0),tf.float32)# label 0 is unknown
    train_weights_tf=tf.Variable(train_weights,False,name='train_weights')
    weights=tf.gather(train_weights_tf,labels_flatten)
    loss=tf.losses.sparse_softmax_cross_entropy(labels_flatten,flatten_logits,weights=weights)
    tf.summary.scalar(loss.op.name,loss)

    return loss,flatten_logits,labels_flatten


def train_ops(xyzs, feats, ctx_pts, ctx_idxs, labels, is_training, epoch_batch_num):
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
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:{}'.format(i%4)):
                with tf.name_scope('tower_{}'.format(i)):
                    loss,logits,label=tower_loss(xyzs[i], feats[i], ctx_pts[i], ctx_idxs[i],
                                                 labels[i], is_training, reuse)

                    grad=opt.compute_gradients(loss)
                    tower_grads.append(grad)
                    tower_losses.append(loss)
                    tower_logits.append(logits)
                    tower_labels.append(label)
                    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
                    reuse=True

        avg_grad=average_gradients(tower_grads)

        with tf.control_dependencies(update_op):
            apply_grad_op=tf.group(opt.apply_gradients(avg_grad,global_step=global_step))
        summary_op = tf.summary.merge(summaries)

        total_loss_op=tf.add_n(tower_losses)/FLAGS.num_gpus
        tower_labels=tf.concat(tower_labels,axis=0)

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

    return ops


def fill_feed_dict(feed_in,feed_dict,pls,num_gpus):
    xyzs, rgbs, covars, lbls, ctx_xyzs, ctx_idxs, block_mins = \
        default_unpack_feats_labels(feed_in, num_gpus)

    batch_pt_num=0
    for k in xrange(num_gpus):
        feed_dict[pls['xyzs'][k]]=xyzs[k]
        feats=np.concatenate([rgbs[k],covars[k]],axis=1)
        feed_dict[pls['feats'][k]]=feats
        feed_dict[pls['lbls'][k]]=lbls[k]
        feed_dict[pls['lbls'][k]]=lbls[k]
        feed_dict[pls['ctx_pts'][k]]=ctx_xyzs[k]
        feed_dict[pls['ctx_idxs'][k]]=ctx_idxs[k]

        batch_pt_num += lbls[k].shape[0]

    return batch_pt_num


def train_one_epoch(ops,pls,sess,summary_writer,trainset,epoch_num,feed_dict):
    epoch_begin=time.time()
    total_correct,total_block,total_points=0,0,0
    begin_time=time.time()
    total_losses=[]
    for i,feed_in in enumerate(trainset):
        batch_pt_num=fill_feed_dict(feed_in,feed_dict,pls,FLAGS.num_gpus)

        feed_dict[pls['is_training']]=True
        total_block+=FLAGS.num_gpus

        _,loss_val,logits,batch_labels,lr=\
            sess.run([ops['apply_grad'],ops['total_loss'],ops['logits'],ops['labels'],ops['learning_rate']],feed_dict)
        total_losses.append(loss_val)
        preds=np.argmax(logits[:,1:],axis=1)+1
        total_correct+=np.sum((batch_labels!=0)&(batch_labels==preds))
        total_points+=batch_pt_num-np.sum(batch_labels==0)

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
        if i>(20000/FLAGS.num_gpus):
            break

    log_str('epoch {} cost {} s'.format(epoch_num, time.time()-epoch_begin), FLAGS.log_file)


def test_one_epoch(ops,pls,sess,saver,testset,epoch_num,feed_dict,summary_writer=None):
    begin_time=time.time()
    test_loss=[]
    all_preds,all_labels=[],[]
    for i,feed_in in enumerate(testset):
        fill_feed_dict(feed_in,feed_dict,pls,FLAGS.num_gpus)

        feed_dict[pls['is_training']] = False

        loss,logits,batch_labels=sess.run([ops['total_loss'],ops['logits'],ops['labels']],feed_dict)
        all_labels.append(batch_labels)
        preds=np.argmax(logits[:,1:],axis=1)+1
        test_loss.append(loss)
        all_preds.append(preds)

    all_preds=np.concatenate(all_preds,axis=0)
    all_labels=np.concatenate(all_labels,axis=0)
    mask=all_labels!=0
    all_preds=all_preds[mask]
    all_labels=all_labels[mask]
    iou, miou, oiou, acc, macc, oacc = compute_iou(all_labels-1,all_preds-1,8)

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


def build_placeholder(num_gpus):
    pls = {}
    pls['xyzs'], pls['feats'], pls['lbls'],pls['weights'],pls['ctx_pts'],pls['ctx_idxs']=[],[],[],[],[],[]
    for i in xrange(num_gpus):
        pls['xyzs'].append(tf.placeholder(tf.float32,[None,3],'xyzs{}'.format(i)))
        pls['feats'].append(tf.placeholder(tf.float32,[None,13],'feats{}'.format(i)))
        pls['lbls'].append(tf.placeholder(tf.int32,[None],'lbls{}'.format(i)))
        pls['weights'].append(tf.placeholder(tf.float32,[None],'weights{}'.format(i)))
        pls['ctx_pts'].append(tf.placeholder(tf.float32,[None,16],'ctx_pts{}'.format(i)))
        pls['ctx_idxs'].append(tf.placeholder(tf.int32,[None],'ctx_idxs{}'.format(i)))

    pls['is_training'] = tf.placeholder(tf.bool, name='is_training')
    return pls


def train():
    test_set=['sg27_station4_intensity_rgb','bildstein_station1_xyz_intensity_rgb']
    train_list,test_list=get_context_train_test(test_set)
    train_list=['data/Semantic3D.Net/context/block_avg/'+fn for fn in train_list]
    test_list=['data/Semantic3D.Net/context/block_avg/'+fn for fn in test_list]
    read_fn=lambda model,filename: read_pkl(filename)
    train_provider = Provider(train_list,'train',FLAGS.batch_size*FLAGS.num_gpus,read_fn)
    test_provider = Provider(test_list,'test',FLAGS.batch_size*FLAGS.num_gpus,read_fn)

    try:
        pls=build_placeholder(FLAGS.num_gpus)

        batch_num_per_epoch=5000/FLAGS.num_gpus
        ops=train_ops(pls['xyzs'],pls['feats'],pls['ctx_pts'],pls['ctx_idxs'],
                      pls['lbls'],pls['is_training'],batch_num_per_epoch)

        feed_dict={}
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        saver = tf.train.Saver(max_to_keep=500)
        sess.run(tf.global_variables_initializer())
        if FLAGS.restore:
            all_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            all_vars=[var for var in all_vars if not var.name.startswith('tower')]
            restore_saver=tf.train.Saver(var_list=all_vars)
            restore_saver.restore(sess,FLAGS.restore_model)
        else:
            sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir,graph=sess.graph)

        for epoch_num in xrange(FLAGS.restore_epoch,FLAGS.train_epoch_num):
            train_one_epoch(ops,pls,sess,summary_writer,train_provider,epoch_num,feed_dict)
            test_one_epoch(ops,pls,sess,saver,test_provider,epoch_num,feed_dict)

    finally:
        train_provider.close()
        test_provider.close()


def eval():
    test_set=['sg27_station4_intensity_rgb','bildstein_station1_xyz_intensity_rgb']
    train_list,test_list=get_context_train_test(test_set)
    train_list=['data/Semantic3D.Net/context/block/'+fn for fn in train_list]
    test_list=['data/Semantic3D.Net/context/block/'+fn for fn in test_list]
    read_fn=lambda model,filename: read_pkl(filename)

    test_provider = Provider(test_list,'test',FLAGS.batch_size*FLAGS.num_gpus,read_fn)

    try:
        pls=build_placeholder(FLAGS.num_gpus)

        batch_num_per_epoch=5000/FLAGS.num_gpus
        ops=train_ops(pls['xyzs'],pls['feats'],pls['ctx_pts'],pls['ctx_idxs'],
                      pls['lbls'],pls['is_training'],batch_num_per_epoch)

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


if __name__=="__main__":
    FLAGS=parser()
    if FLAGS.eval: eval()
    else: train()