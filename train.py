import argparse
import time
import numpy as np
import os

import tensorflow as tf
from model import graph_conv_net_v2,classifier
from train_util import *
from io_util import get_block_train_test_split,read_fn,get_class_names
from draw_util import get_class_colors,output_points
from provider import Provider,default_unpack_feats_labels

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=4, help='')
parser.add_argument('--batch_size', type=int, default=1, help='')

parser.add_argument('--lr_init', type=float, default=1e-3, help='')
parser.add_argument('--lr_clip', type=float, default=1e-5, help='')
parser.add_argument('--decay_rate', type=float, default=0.5, help='')
parser.add_argument('--decay_epoch', type=int, default=50, help='')
parser.add_argument('--num_classes', type=int, default=13, help='')

parser.add_argument('--restore',type=bool, default=False, help='')
parser.add_argument('--restore_epoch', type=int, default=0, help='')
parser.add_argument('--restore_model', type=str, default='', help='')

parser.add_argument('--log_step', type=int, default=50, help='')
parser.add_argument('--train_dir', type=str, default='train/s3dis_graph', help='')
parser.add_argument('--save_dir', type=str, default='model/s3dis_graph', help='')
parser.add_argument('--log_file', type=str, default='s3dis_graph.log', help='')


parser.add_argument('--eval',type=bool, default=False, help='')
parser.add_argument('--eval_model',type=str, default='model/label/unsupervise80.ckpt',help='')
parser.add_argument('--eval_output',type=bool, default=True,help='')

parser.add_argument('--train_epoch_num', type=int, default=500, help='')

FLAGS = parser.parse_args()


def tower_loss(xyz,feats,labels,cidxs,nidxs,nidxs_lens,nidxs_bgs,m,
               is_training,reuse=False,pmiu=None):

    with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
        global_feats,global_pfeats,local_pfeats=\
            graph_conv_net_v2(xyz, feats, cidxs, nidxs, nidxs_lens, nidxs_bgs, m, pmiu, reuse, 1024)

        global_feats=tf.expand_dims(global_feats,axis=0)    # [1,1024]
        k=tf.shape(global_pfeats)[0]
        global_feats=tf.tile(global_feats,[k,1])            #[pn,1024]
        all_feats=tf.concat([global_feats, global_pfeats, local_pfeats],axis=1)

        all_feats=tf.expand_dims(all_feats,0)                 # [1,pn,2048+252]
        local_pfeats=tf.expand_dims(local_pfeats,0)           # [1,pn,12]
        logits=classifier(all_feats, local_pfeats, is_training, FLAGS.num_classes, reuse, use_bn=False)  # [1,pn,num_classes]

    flatten_logits=tf.reshape(logits,[-1,FLAGS.num_classes])  # [pn,num_classes]
    labels_flatten=tf.reshape(labels,[-1,1])                  # [pn,1]
    labels_flatten=tf.squeeze(labels_flatten,axis=1)          # [pn]
    loss=tf.losses.sparse_softmax_cross_entropy(labels_flatten,flatten_logits)
    tf.summary.scalar(loss.op.name,loss)

    return loss,logits


def train_ops(xyz, rgbs, covars, labels, cidxs, nidxs, nidxs_lens, nidxs_bgs, m,
              is_training, epoch_batch_num, pmiu=None):

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
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:{}'.format(i)):
                with tf.name_scope('tower_{}'.format(i)):
                    # print points[i],labels[i]
                    feats=tf.concat([rgbs[i],covars[i]],axis=1)
                    loss,logits=tower_loss(xyz[i], feats, labels[i], cidxs[i],
                                           nidxs[i], nidxs_lens[i], nidxs_bgs[i],
                                           m,is_training, reuse, pmiu)

                    grad=opt.compute_gradients(loss)
                    tower_grads.append(grad)
                    tower_losses.append(loss)
                    tower_logits.append(tf.squeeze(logits,axis=0))
                    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    for g in grad:
                        print g
                    reuse=True

        avg_grad=average_gradients(tower_grads)

        with tf.control_dependencies(update_op):
            apply_grad_op=tf.group(opt.apply_gradients(avg_grad,global_step=global_step))

        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summary_op = tf.summary.merge(summaries)

        total_loss_op=tf.add_n(tower_losses)/FLAGS.num_gpus

        logits_op=tf.concat(tower_logits,axis=0)
        preds_op=tf.argmax(logits_op,axis=1)

        flatten_labels=[]
        for i in xrange(FLAGS.num_gpus):
            flatten_labels.append(labels[i])

        flatten_labels=tf.concat(flatten_labels,axis=0)
        correct_num_op=tf.reduce_sum(tf.cast(tf.equal(preds_op,flatten_labels),tf.float32))

        ops['total_loss']=total_loss_op
        ops['apply_grad']=apply_grad_op
        ops['logits']=logits_op
        ops['preds']=preds_op
        ops['correct_num']=correct_num_op
        ops['summary']=summary_op
        ops['global_step']=global_step

    return ops


def train_one_epoch(ops,pls,sess,summary_writer,trainset,epoch_num,feed_dict):
    epoch_begin=time.time()
    total_correct,total_block,total_points=0,0,0
    begin_time=time.time()
    total_losses=[]
    for i,feed_in in enumerate(trainset):
        xyzs, rgbs, covars, lbls, nidxs, nidxs_lens, nidxs_bgs, cidxs, block_bgs, block_lens=\
            default_unpack_feats_labels(feed_in,FLAGS.num_gpus)

        with open('test.log', 'a') as f:
            f.write('xyzs lens {}\t'.format(len(xyzs)))
            f.write('rgbs lens {}\t'.format(len(xyzs)))
            f.write('xyzs lens {}\t'.format(len(xyzs)))
        for k in xrange(FLAGS.num_gpus):
            feed_dict[pls['xyzs'][k]]=xyzs[k]
            feed_dict[pls['rgbs'][k]]=rgbs[k]
            feed_dict[pls['covars'][k]]=covars[k]
            feed_dict[pls['lbls'][k]]=lbls[k]
            feed_dict[pls['nidxs'][k]]=nidxs[k]
            feed_dict[pls['nidxs_lens'][k]]=nidxs_lens[k]
            feed_dict[pls['nidxs_bgs'][k]]=nidxs_bgs[k]
            feed_dict[pls['cidxs'][k]]=cidxs[k]

            total_points+=lbls[k].shape[0]

        feed_dict[pls['is_training']]=True
        total_block+=FLAGS.num_gpus

        _,loss_val,correct_num=sess.run([ops['apply_grad'],ops['total_loss'],ops['correct_num']],feed_dict)
        total_losses.append(loss_val)
        total_correct+=correct_num

        if i % FLAGS.log_step==0:
            summary,global_step=sess.run(
                [ops['summary'],ops['global_step']],feed_dict)

            log_str('epoch {} step {} loss {:.5} acc {:.5} | {:.5} examples/s'.format(
                epoch_num,i,np.mean(np.asarray(total_losses)),
                float(total_correct)/total_points,
                float(total_block)/(time.time()-begin_time)
            ),FLAGS.log_file)

            summary_writer.add_summary(summary,global_step)
            total_correct,total_block,total_points=0,0,0
            begin_time=time.time()
            total_losses=[]

    log_str('epoch {} cost {} s'.format(epoch_num, time.time()-epoch_begin), FLAGS.log_file)


def test_one_epoch(ops,pls,sess,saver,testset,epoch_num,feed_dict):
    begin_time=time.time()
    test_loss=[]
    all_preds,all_labels=[],[]
    colors=get_class_colors()
    for i,feed_in in enumerate(testset):
        xyzs, rgbs, covars, lbls, nidxs, nidxs_lens, nidxs_bgs, cidxs, block_bgs, block_lens=\
            default_unpack_feats_labels(feed_in,FLAGS.num_gpus)

        for k in xrange(FLAGS.num_gpus):
            feed_dict[pls['xyzs'][k]]=xyzs[k]
            feed_dict[pls['rgbs'][k]]=rgbs[k]
            feed_dict[pls['covars'][k]]=covars[k]
            feed_dict[pls['lbls'][k]]=lbls[k]
            feed_dict[pls['nidxs'][k]]=nidxs[k]
            feed_dict[pls['nidxs_lens'][k]]=nidxs_lens[k]
            feed_dict[pls['nidxs_bgs'][k]]=nidxs_bgs[k]
            feed_dict[pls['cidxs'][k]]=cidxs[k]
            all_labels.append(lbls[k])

        feed_dict[pls['is_training']] = False

        loss,preds=sess.run([ops['total_loss'],ops['preds']],feed_dict)
        test_loss.append(loss)
        all_preds.append(preds)

        # output labels and true
        if FLAGS.eval and FLAGS.eval_output:
            cur=0
            for k in xrange(FLAGS.num_gpus):
                restore_xyzs=xyzs[k]
                restore_xyzs[:,:2]=restore_xyzs[:,:2]*1.5+1.5
                restore_xyzs[:,2]+=1.0
                restore_xyzs[:,2]*=block_lens[k][2]/2
                restore_xyzs+=block_bgs[k]
                output_points('test_result/{}_{}_true.txt'.format(i,k),restore_xyzs,colors[lbls[k],:])
                output_points('test_result/{}_{}_pred.txt'.format(i,k),restore_xyzs,colors[preds[cur:cur+len(xyzs[k])],:])
                cur+=len(xyzs[k])

    all_preds=np.concatenate(all_preds,axis=0)
    all_labels=np.concatenate(all_labels,axis=0)

    test_loss=np.mean(np.asarray(test_loss))

    iou, miou, oiou, acc, macc, oacc = compute_iou(all_labels,all_preds)

    log_str('mean iou {:.5} overall iou {:5} loss {:5} \n mean acc {:5} overall acc {:5} cost {:3} s'.format(
        miou, oiou, test_loss, macc, oacc, time.time()-begin_time
    ),FLAGS.log_file)

    if not FLAGS.eval:
        checkpoint_path = os.path.join(FLAGS.save_dir, 'model{}.ckpt'.format(epoch_num))
        saver.save(sess,checkpoint_path)
    else:
        names=get_class_names()
        for i in xrange(len(names)):
            print '{} iou {} acc {}'.format(names[i],iou[i],acc[i])


def neighbor_anchors_v2():
    interval=2*np.pi/8
    pmiu=[]
    for va in np.arange(-np.pi/2+interval,np.pi/2,interval):
        for ha in np.arange(0,2*np.pi,interval):
            v=np.asarray([np.cos(va)*np.cos(ha),
                         np.cos(va)*np.sin(ha),
                         np.sin(va)],dtype=np.float32)
            pmiu.append(v)

    pmiu.append(np.asarray([0,0,1.0]))
    pmiu.append(np.asarray([0,0,-1.0]))
    return np.asarray(pmiu)


def neighbor_anchors():
    interval=2*np.pi/8
    pmiu=[]
    for va in np.arange(-np.pi/2,np.pi/2+interval,interval):
        for ha in np.arange(0,2*np.pi,interval):
            v=np.asarray([np.cos(va)*np.cos(ha),
                         np.cos(va)*np.sin(ha),
                         np.sin(va)],dtype=np.float32)
            pmiu.append(v)

    return np.asarray(pmiu).transpose()


def train():
    train_list,test_list=get_block_train_test_split()
    train_list=['data/S3DIS/room_block_10_10/'+fn for fn in train_list]
    test_list=['data/S3DIS/room_block_10_10/'+fn for fn in test_list]

    train_provider = Provider(train_list,'train',FLAGS.batch_size*FLAGS.num_gpus,read_fn)
    test_provider = Provider(test_list,'test',FLAGS.batch_size*FLAGS.num_gpus,read_fn)
    # test_provider = Provider(train_list[:2],'test',FLAGS.batch_size*FLAGS.num_gpus,read_fn)

    try:
        pls={}
        pls['xyzs'],pls['lbls'],pls['rgbs'],pls['covars'],pls['nidxs'],\
        pls['nidxs_lens'],pls['nidxs_bgs'],pls['cidxs']=[],[],[],[],[],[],[],[]
        for i in xrange(FLAGS.num_gpus):
            pls['xyzs'].append(tf.placeholder(tf.float32,[None,3],'xyz{}'.format(i)))
            pls['rgbs'].append(tf.placeholder(tf.float32,[None,3],'rgb{}'.format(i)))
            pls['covars'].append(tf.placeholder(tf.float32,[None,9],'covar{}'.format(i)))
            pls['lbls'].append(tf.placeholder(tf.int64,[None],'lbl{}'.format(i)))
            pls['nidxs'].append(tf.placeholder(tf.int32,[None],'nidxs{}'.format(i)))
            pls['nidxs_lens'].append(tf.placeholder(tf.int32,[None],'nidxs_lens{}'.format(i)))
            pls['nidxs_bgs'].append(tf.placeholder(tf.int32,[None],'nidxs_bgs{}'.format(i)))
            pls['cidxs'].append(tf.placeholder(tf.int32,[None],'cidxs{}'.format(i)))

        pmiu=neighbor_anchors()
        pls['is_training']=tf.placeholder(tf.bool,name='is_training')
        pls['pmiu']=tf.placeholder(tf.float32,name='pmiu')

        batch_num_per_epoch=2500/FLAGS.num_gpus
        ops=train_ops(pls['xyzs'],pls['rgbs'],pls['covars'],pls['lbls'],
                      pls['cidxs'],pls['nidxs'],pls['nidxs_lens'],pls['nidxs_bgs'],
                      pmiu.shape[1],pls['is_training'],batch_num_per_epoch,pls['pmiu'])

        feed_dict={}
        feed_dict[pls['pmiu']]=pmiu
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)
        saver = tf.train.Saver(max_to_keep=500)
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


def eval():
    train_list,test_list=get_block_train_test_split()
    test_list=['data/S3DIS/room_block_10_10/'+fn for fn in test_list]

    test_provider = Provider(test_list,'test',FLAGS.batch_size*FLAGS.num_gpus,read_fn)

    try:
        pls={}
        pls['xyzs'],pls['lbls'],pls['rgbs'],pls['covars'],pls['nidxs'],\
        pls['nidxs_lens'],pls['nidxs_bgs'],pls['cidxs']=[],[],[],[],[],[],[],[]
        for i in xrange(FLAGS.num_gpus):
            pls['xyzs'].append(tf.placeholder(tf.float32,[None,3],'xyz{}'.format(i)))
            pls['rgbs'].append(tf.placeholder(tf.float32,[None,3],'rgb{}'.format(i)))
            pls['covars'].append(tf.placeholder(tf.float32,[None,9],'covar{}'.format(i)))
            pls['lbls'].append(tf.placeholder(tf.int64,[None],'lbl{}'.format(i)))
            pls['nidxs'].append(tf.placeholder(tf.int32,[None],'nidxs{}'.format(i)))
            pls['nidxs_lens'].append(tf.placeholder(tf.int32,[None],'nidxs_lens{}'.format(i)))
            pls['nidxs_bgs'].append(tf.placeholder(tf.int32,[None],'nidxs_bgs{}'.format(i)))
            pls['cidxs'].append(tf.placeholder(tf.int32,[None],'cidxs{}'.format(i)))

        pmiu=neighbor_anchors()
        pls['is_training']=tf.placeholder(tf.bool,name='is_training')
        pls['pmiu']=tf.placeholder(tf.float32,name='pmiu')

        batch_num_per_epoch=2500/FLAGS.num_gpus
        ops=train_ops(pls['xyzs'],pls['rgbs'],pls['covars'],pls['lbls'],
                      pls['cidxs'],pls['nidxs'],pls['nidxs_lens'],pls['nidxs_bgs'],
                      pmiu.shape[1],pls['is_training'],batch_num_per_epoch,pls['pmiu'])

        feed_dict={}
        feed_dict[pls['pmiu']]=pmiu
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        saver = tf.train.Saver(max_to_keep=500)
        saver.restore(sess,FLAGS.eval_model)
        test_one_epoch(ops,pls,sess,saver,test_provider,0,feed_dict)

    finally:
        test_provider.close()


if __name__=="__main__":
    if FLAGS.eval: eval()
    else: train()