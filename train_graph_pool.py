import argparse
import time
import numpy as np
import os

import tensorflow as tf
from model import graph_conv_pool_v1,classifier_v2
from train_util import *
from io_util import get_block_train_test_split_ds,read_fn_hierarchy,get_class_names
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

parser.add_argument('--log_step', type=int, default=120, help='')
parser.add_argument('--train_dir', type=str, default='train/s3dis_graph', help='')
parser.add_argument('--save_dir', type=str, default='model/s3dis_graph', help='')
parser.add_argument('--log_file', type=str, default='s3dis_graph.log', help='')


parser.add_argument('--eval',type=bool, default=False, help='')
parser.add_argument('--eval_model',type=str, default='model/label/unsupervise80.ckpt',help='')

parser.add_argument('--train_epoch_num', type=int, default=500, help='')

FLAGS = parser.parse_args()


def tower_loss(cxyzs, dxyzs, rgbs, covars, vlens, vlens_bgs, vcidxs,
               cidxs, nidxs, nidxs_lens, nidxs_bgs,
               labels, m, is_training,reuse=False,pmiu=None):

    with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
        feats=graph_conv_pool_v1(cxyzs, dxyzs, rgbs, covars, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_lens, nidxs_bgs,m,pmiu,reuse)
        feats=tf.expand_dims(feats,axis=0)
        logits=classifier_v2(feats, is_training, FLAGS.num_classes, reuse, use_bn=False)  # [1,pn,num_classes]

    flatten_logits=tf.reshape(logits,[-1,FLAGS.num_classes])  # [pn,num_classes]
    labels_flatten=tf.reshape(labels,[-1,1])                  # [pn,1]
    labels_flatten=tf.squeeze(labels_flatten,axis=1)          # [pn]
    loss=tf.losses.sparse_softmax_cross_entropy(labels_flatten,flatten_logits)
    tf.summary.scalar(loss.op.name,loss)

    return loss,logits


def train_ops(cxyzs, dxyzs, rgbs, covars, vlens, vlens_bgs, vcidxs,
              cidxs, nidxs, nidxs_lens, nidxs_bgs, labels, m,
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
                    loss,logits=tower_loss(cxyzs[i], dxyzs[i], rgbs[i], covars[i], vlens[i], vlens_bgs[i], vcidxs[i],
                                           cidxs[i], nidxs[i], nidxs_lens[i], nidxs_bgs[i], labels[i],
                                           m, is_training, reuse, pmiu)

                    grad=opt.compute_gradients(loss)
                    tower_grads.append(grad)
                    tower_losses.append(loss)
                    tower_logits.append(tf.squeeze(logits,axis=0))
                    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
                    reuse=True

        avg_grad=average_gradients(tower_grads)

        with tf.control_dependencies(update_op):
            apply_grad_op=tf.group(opt.apply_gradients(avg_grad,global_step=global_step))

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


def fill_feed_dict(feed_in,feed_dict,pls):
    cxyzs, dxyzs, rgbs, covars, lbls, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens = \
        default_unpack_feats_labels(feed_in, FLAGS.num_gpus)

    batch_pt_num=0
    batch_labels=[]
    for k in xrange(FLAGS.num_gpus):
        for t in xrange(3):
            feed_dict[pls['cxyzs'][k][t]] = cxyzs[k][t]
            feed_dict[pls['nidxs'][k][t]] = nidxs[k][t]
            feed_dict[pls['nidxs_lens'][k][t]] = nidxs_lens[k][t]
            feed_dict[pls['nidxs_bgs'][k][t]] = nidxs_bgs[k][t]
            feed_dict[pls['cidxs'][k][t]] = cidxs[k][t]

        feed_dict[pls['rgbs'][k]] = rgbs[k]
        feed_dict[pls['covars'][k]] = covars[k]
        feed_dict[pls['lbls'][k]] = lbls[k]

        for t in xrange(2):
            feed_dict[pls['dxyzs'][k][t]] = dxyzs[k][t]
            feed_dict[pls['vlens'][k][t]] = vlens[k][t]
            feed_dict[pls['vlens_bgs'][k][t]] = vlens_bgs[k][t]
            feed_dict[pls['vcidxs'][k][t]] = vcidxs[k][t]

        batch_pt_num += lbls[k].shape[0]
        batch_labels.append(lbls[k])

    return batch_pt_num,batch_labels


def train_one_epoch(ops,pls,sess,summary_writer,trainset,epoch_num,feed_dict):
    epoch_begin=time.time()
    total_correct,total_block,total_points=0,0,0
    begin_time=time.time()
    total_losses=[]
    for i,feed_in in enumerate(trainset):
        batch_pt_num,_=fill_feed_dict(feed_in,feed_dict,pls)

        feed_dict[pls['is_training']]=True
        total_block+=FLAGS.num_gpus
        total_points+=batch_pt_num

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


def test_one_epoch(ops,pls,sess,saver,testset,epoch_num,feed_dict,summary_writer=None):
    begin_time=time.time()
    test_loss=[]
    all_preds,all_labels=[],[]
    for i,feed_in in enumerate(testset):
        _,batch_labels=fill_feed_dict(feed_in,feed_dict,pls)

        feed_dict[pls['is_training']] = False
        all_labels+=batch_labels

        if FLAGS.eval and FLAGS.num_monitor:
            loss,preds,summary=sess.run([ops['total_loss'],ops['preds'],ops['summary']],feed_dict)
            summary_writer.add_summary(summary)
        else:
            loss,preds=sess.run([ops['total_loss'],ops['preds']],feed_dict)
        test_loss.append(loss)
        all_preds.append(preds)

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
    return np.asarray(pmiu).transpose()


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
    train_list,test_list=get_block_train_test_split_ds()
    train_list=['data/S3DIS/room_block_10_10_ds0.03/'+fn for fn in train_list]
    test_list=['data/S3DIS/room_block_10_10_ds0.03/'+fn for fn in test_list]

    train_provider = Provider(train_list,'train',FLAGS.batch_size*FLAGS.num_gpus,read_fn_hierarchy)
    test_provider = Provider(test_list,'test',FLAGS.batch_size*FLAGS.num_gpus,read_fn_hierarchy)
    # test_provider = Provider(train_list[:2],'test',FLAGS.batch_size*FLAGS.num_gpus,read_fn_hierarchy)

    try:
        pls={}
        pls['cxyzs'],pls['lbls'],pls['rgbs'],pls['covars'],pls['nidxs'],\
        pls['nidxs_lens'],pls['nidxs_bgs'],pls['cidxs']=[],[],[],[],[],[],[],[]
        pls['vlens'],pls['vlens_bgs'],pls['vcidxs'],pls['dxyzs']=[],[],[],[]
        for i in xrange(FLAGS.num_gpus):
            cxyzs=[tf.placeholder(tf.float32,[None,3],'cxyz{}_{}'.format(j,i)) for j in xrange(3)]
            pls['cxyzs'].append(cxyzs)
            pls['rgbs'].append(tf.placeholder(tf.float32,[None,3],'rgb{}'.format(i)))
            pls['covars'].append(tf.placeholder(tf.float32,[None,9],'covar{}'.format(i)))
            pls['lbls'].append(tf.placeholder(tf.int64,[None],'lbl{}'.format(i)))

            nidxs=[tf.placeholder(tf.int32,[None],'nidxs{}_{}'.format(j,i)) for j in xrange(3)]
            nidxs_lens=[tf.placeholder(tf.int32,[None],'nidxs_lens{}_{}'.format(j,i)) for j in xrange(3)]
            nidxs_bgs=[tf.placeholder(tf.int32,[None],'nidxs{}_{}'.format(j,i)) for j in xrange(3)]
            cidxs=[tf.placeholder(tf.int32,[None],'cidxs{}_{}'.format(j,i)) for j in xrange(3)]
            pls['nidxs'].append(nidxs)
            pls['nidxs_lens'].append(nidxs_lens)
            pls['nidxs_bgs'].append(nidxs_bgs)
            pls['cidxs'].append(cidxs)

            vlens=[tf.placeholder(tf.int32,[None],'vlens{}_{}'.format(j,i)) for j in xrange(2)]
            vlens_bgs=[tf.placeholder(tf.int32,[None],'vlens_bgs{}_{}'.format(j,i)) for j in xrange(2)]
            vcidxs=[tf.placeholder(tf.int32,[None],'vcidxs{}_{}'.format(j,i)) for j in xrange(2)]
            dxyzs=[tf.placeholder(tf.float32,[None,3],'dxyzs{}_{}'.format(j,i)) for j in xrange(2)]

            pls['vlens'].append(vlens)
            pls['vlens_bgs'].append(vlens_bgs)
            pls['vcidxs'].append(vcidxs)
            pls['dxyzs'].append(dxyzs)

        pmiu=neighbor_anchors_v2()
        pls['is_training']=tf.placeholder(tf.bool,name='is_training')
        pls['pmiu']=tf.placeholder(tf.float32,name='pmiu')

        batch_num_per_epoch=2000/FLAGS.num_gpus
        ops=train_ops(pls['cxyzs'],pls['dxyzs'],pls['rgbs'],pls['covars'],
                      pls['vlens'],pls['vlens_bgs'],pls['vcidxs'],
                      pls['cidxs'],pls['nidxs'],pls['nidxs_lens'],pls['nidxs_bgs'],
                      pls['lbls'],pmiu.shape[1],pls['is_training'],
                      batch_num_per_epoch,pls['pmiu'])

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
    train_list,test_list=get_block_train_test_split_ds()
    test_list=['data/S3DIS/room_block_10_10_ds0.03/'+fn for fn in test_list]

    test_provider = Provider(test_list,'test',FLAGS.batch_size*FLAGS.num_gpus,read_fn_hierarchy)

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

        pmiu=neighbor_anchors_v2()
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
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir,graph=sess.graph)
        test_one_epoch(ops,pls,sess,saver,test_provider,0,feed_dict,summary_writer)

    finally:
        test_provider.close()


if __name__=="__main__":
    if FLAGS.eval: eval()
    else: train()