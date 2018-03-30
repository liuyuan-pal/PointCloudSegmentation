import argparse
import time
import numpy as np
import os

import tensorflow as tf
from model import graph_conv_pool_new_v2,classifier_v3
from train_util import *
from io_util import read_pkl,get_scannet_class_names
from provider import Provider,default_unpack_feats_labels
from draw_util import output_points,get_scannet_class_colors
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=4, help='')
parser.add_argument('--batch_size', type=int, default=1, help='')

parser.add_argument('--lr_init', type=float, default=1e-3, help='')
parser.add_argument('--lr_clip', type=float, default=1e-5, help='')
parser.add_argument('--decay_rate', type=float, default=0.5, help='')
parser.add_argument('--decay_epoch', type=int, default=50, help='')
parser.add_argument('--num_classes', type=int, default=21, help='')

parser.add_argument('--restore',type=bool, default=False, help='')
parser.add_argument('--restore_epoch', type=int, default=0, help='')
parser.add_argument('--restore_model', type=str, default='', help='')

parser.add_argument('--log_step', type=int, default=240, help='')
parser.add_argument('--train_dir', type=str, default='train/gpn_scannet', help='')
parser.add_argument('--save_dir', type=str, default='model/gpn_scannet', help='')
parser.add_argument('--log_file', type=str, default='gpn_scannet.log', help='')


parser.add_argument('--eval',type=bool, default=False, help='')
parser.add_argument('--eval_model',type=str, default='model/label/unsupervise80.ckpt',help='')
parser.add_argument('--eval_output',type=bool, default=False,help='')

parser.add_argument('--train_epoch_num', type=int, default=500, help='')

FLAGS = parser.parse_args()


lbl_weights=np.asarray([
    0.0,
    2.2230784893,
    2.69648623466,
    4.54655218124,
    4.92085981369,
    5.09989976883,
    4.91159963608,
    5.02148008347,
    4.90901327133,
    5.40208673477,
    5.40154600143,
    5.4178404808,
    5.14018535614,
    5.33298397064,
    4.96147441864,
    5.25951480865,
    5.43916702271,
    5.38037347794,
    5.39362192154,
    4.90917301178,
    4.93606853485,
])

def tower_loss(cxyzs, dxyzs, covars, vlens, vlens_bgs, vcidxs,
               cidxs, nidxs, nidxs_lens, nidxs_bgs,
               labels, weights, m, is_training, reuse=False,pmiu=None):

    with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
        feats,lf=graph_conv_pool_new_v2(cxyzs, dxyzs, covars, vlens, vlens_bgs, vcidxs,
                                        cidxs, nidxs, nidxs_lens, nidxs_bgs,m,pmiu,reuse)
        feats=tf.expand_dims(feats,axis=0)
        lf=tf.expand_dims(lf,axis=0)
        logits=classifier_v3(feats, lf, is_training, FLAGS.num_classes, reuse, use_bn=False)  # [1,pn,num_classes]

    flatten_logits=tf.reshape(logits,[-1,FLAGS.num_classes])  # [pn,num_classes]
    labels_flatten=tf.reshape(labels,[-1,1])                  # [pn,1]
    labels_flatten=tf.squeeze(labels_flatten,axis=1)          # [pn]
    loss=tf.losses.sparse_softmax_cross_entropy(labels_flatten,flatten_logits,weights=weights)
    tf.summary.scalar(loss.op.name,loss)

    return loss,logits


def train_ops(cxyzs, dxyzs, covars, vlens, vlens_bgs, vcidxs,
              cidxs, nidxs, nidxs_lens, nidxs_bgs, labels, weights, m,
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
                    loss,logits=tower_loss(cxyzs[i], dxyzs[i], covars[i], vlens[i], vlens_bgs[i], vcidxs[i],
                                           cidxs[i], nidxs[i], nidxs_lens[i], nidxs_bgs[i], labels[i], weights[i],
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

        ops['total_loss']=total_loss_op
        ops['apply_grad']=apply_grad_op
        ops['logits']=logits_op
        ops['summary']=summary_op
        ops['global_step']=global_step

    return ops


def fill_feed_dict(feed_in,feed_dict,pls,num_gpus):
    cxyzs, dxyzs, covars, lbls, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens, block_mins = \
        default_unpack_feats_labels(feed_in, num_gpus)

    batch_pt_num=0
    batch_labels=[]
    for k in xrange(num_gpus):
        for t in xrange(3):
            feed_dict[pls['cxyzs'][k][t]] = cxyzs[k][t]
            feed_dict[pls['nidxs'][k][t]] = nidxs[k][t]
            feed_dict[pls['nidxs_lens'][k][t]] = nidxs_lens[k][t]
            feed_dict[pls['nidxs_bgs'][k][t]] = nidxs_bgs[k][t]
            feed_dict[pls['cidxs'][k][t]] = cidxs[k][t]

        feed_dict[pls['covars'][k]] = covars[k]
        feed_dict[pls['lbls'][k]] = lbls[k]
        feed_dict[pls['weights'][k]] = lbl_weights[lbls[k]]

        for t in xrange(2):
            feed_dict[pls['dxyzs'][k][t]] = dxyzs[k][t]
            feed_dict[pls['vlens'][k][t]] = vlens[k][t]
            feed_dict[pls['vlens_bgs'][k][t]] = vlens_bgs[k][t]
            feed_dict[pls['vcidxs'][k][t]] = vcidxs[k][t]


        batch_pt_num += lbls[k].shape[0]
        batch_labels.append(lbls[k])

    return batch_pt_num,batch_labels,block_mins


def train_one_epoch(ops,pls,sess,summary_writer,trainset,epoch_num,feed_dict):
    epoch_begin=time.time()
    total_correct,total_block,total_points=0,0,0
    begin_time=time.time()
    total_losses=[]
    for i,feed_in in enumerate(trainset):
        batch_pt_num,batch_labels,_=fill_feed_dict(feed_in,feed_dict,pls,FLAGS.num_gpus)
        batch_labels=np.concatenate(batch_labels,axis=0)

        feed_dict[pls['is_training']]=True
        total_block+=FLAGS.num_gpus
        total_points+=batch_pt_num-np.sum(batch_labels==0)

        _,loss_val,logits=sess.run([ops['apply_grad'],ops['total_loss'],ops['logits']],feed_dict)
        total_losses.append(loss_val)
        preds=np.argmax(logits[:,1:],axis=1)+1
        total_correct+=np.sum((batch_labels!=0)&(batch_labels==preds))

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

        # the generated training set is too large
        # so every 10000 examples we test once and save the model
        if i>5000:
            break

    log_str('epoch {} cost {} s'.format(epoch_num, time.time()-epoch_begin), FLAGS.log_file)


def test_one_epoch(ops,pls,sess,saver,testset,epoch_num,feed_dict,summary_writer=None):
    begin_time=time.time()
    test_loss=[]
    all_preds,all_labels=[],[]
    colors=get_scannet_class_colors()
    for i,feed_in in enumerate(testset):
        _,batch_labels,block_mins=fill_feed_dict(feed_in,feed_dict,pls,FLAGS.num_gpus)

        feed_dict[pls['is_training']] = False
        all_labels+=batch_labels

        loss,logits=sess.run([ops['total_loss'],ops['logits']],feed_dict)
        preds=np.argmax(logits[:,1:],axis=1)+1
        test_loss.append(loss)
        all_preds.append(preds)

        # output labels and true
        if FLAGS.eval and FLAGS.eval_output:
            cur=0
            for k in xrange(FLAGS.num_gpus):
                xyzs=feed_dict[pls['cxyzs'][k][0]]
                lbls=feed_dict[pls['lbls'][k]]
                xyzs+=block_mins[k]
                output_points('test_result/{}_{}_true.txt'.format(i,k),xyzs,colors[lbls,:])
                output_points('test_result/{}_{}_pred.txt'.format(i,k),xyzs,colors[preds[cur:cur+len(xyzs)],:])
                cur+=len(xyzs)

    all_preds=np.concatenate(all_preds,axis=0)
    all_labels=np.concatenate(all_labels,axis=0)
    mask=all_labels!=0
    all_preds=all_preds[mask]
    all_labels=all_labels[mask]
    iou, miou, oiou, acc, macc, oacc = compute_iou(all_labels-1,all_preds-1,20)

    test_loss=np.mean(np.asarray(test_loss))
    log_str('mean iou {:.5} overall iou {:5} loss {:5} \n mean acc {:5} overall acc {:5} cost {:3} s'.format(
        miou, oiou, test_loss, macc, oacc, time.time()-begin_time
    ),FLAGS.log_file)

    if not FLAGS.eval:
        checkpoint_path = os.path.join(FLAGS.save_dir, 'model{}.ckpt'.format(epoch_num))
        saver.save(sess,checkpoint_path)
    else:
        names=get_scannet_class_names()[1:]
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


def build_placeholder(num_gpus):
    pls = {}
    pls['cxyzs'], pls['lbls'], pls['rgbs'], pls['covars'], pls['nidxs'], \
    pls['nidxs_lens'], pls['nidxs_bgs'], pls['cidxs'] = [], [], [], [], [], [], [], []
    pls['vlens'], pls['vlens_bgs'], pls['vcidxs'], pls['dxyzs'] = [], [], [], []
    pls['weights'] = []
    for i in xrange(num_gpus):
        cxyzs = [tf.placeholder(tf.float32, [None, 3], 'cxyz{}_{}'.format(j, i)) for j in xrange(3)]
        pls['cxyzs'].append(cxyzs)
        pls['covars'].append(tf.placeholder(tf.float32, [None, 9], 'covar{}'.format(i)))
        pls['lbls'].append(tf.placeholder(tf.int64, [None], 'lbl{}'.format(i)))
        pls['weights'].append(tf.placeholder(tf.float32, [None], 'weights{}'.format(i)))

        nidxs = [tf.placeholder(tf.int32, [None], 'nidxs{}_{}'.format(j, i)) for j in xrange(3)]
        nidxs_lens = [tf.placeholder(tf.int32, [None], 'nidxs_lens{}_{}'.format(j, i)) for j in xrange(3)]
        nidxs_bgs = [tf.placeholder(tf.int32, [None], 'nidxs{}_{}'.format(j, i)) for j in xrange(3)]
        cidxs = [tf.placeholder(tf.int32, [None], 'cidxs{}_{}'.format(j, i)) for j in xrange(3)]
        pls['nidxs'].append(nidxs)
        pls['nidxs_lens'].append(nidxs_lens)
        pls['nidxs_bgs'].append(nidxs_bgs)
        pls['cidxs'].append(cidxs)

        vlens = [tf.placeholder(tf.int32, [None], 'vlens{}_{}'.format(j, i)) for j in xrange(2)]
        vlens_bgs = [tf.placeholder(tf.int32, [None], 'vlens_bgs{}_{}'.format(j, i)) for j in xrange(2)]
        vcidxs = [tf.placeholder(tf.int32, [None], 'vcidxs{}_{}'.format(j, i)) for j in xrange(2)]
        dxyzs = [tf.placeholder(tf.float32, [None, 3], 'dxyzs{}_{}'.format(j, i)) for j in xrange(2)]

        pls['vlens'].append(vlens)
        pls['vlens_bgs'].append(vlens_bgs)
        pls['vcidxs'].append(vcidxs)
        pls['dxyzs'].append(dxyzs)

    pls['is_training'] = tf.placeholder(tf.bool, name='is_training')
    pls['pmiu'] = tf.placeholder(tf.float32, name='pmiu')
    return pls


def train():
    with open('cached/scannet_train_filenames.txt','r') as f:
        train_list=[line.strip('\n') for line in f.readlines()]
    train_list=['data/ScanNet/sampled_train/{}'.format(fn) for fn in train_list]
    test_list=['data/ScanNet/sampled_test/test_{}.pkl'.format(i) for i in xrange(312)]
    read_fn=lambda model,filename: read_pkl(filename)

    train_provider = Provider(train_list,'train',FLAGS.batch_size*FLAGS.num_gpus,read_fn)
    test_provider = Provider(test_list,'test',FLAGS.batch_size*FLAGS.num_gpus,read_fn)
    try:
        pls=build_placeholder(FLAGS.num_gpus)
        pmiu=neighbor_anchors_v2()

        batch_num_per_epoch=11000/FLAGS.num_gpus
        ops=train_ops(pls['cxyzs'],pls['dxyzs'],pls['covars'],
                      pls['vlens'],pls['vlens_bgs'],pls['vcidxs'],
                      pls['cidxs'],pls['nidxs'],pls['nidxs_lens'],pls['nidxs_bgs'],
                      pls['lbls'],pls['weights'],pmiu.shape[1],pls['is_training'],
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
    test_list=['data/ScanNet/sampled_test/test_{}.pkl'.format(i) for i in xrange(312)]
    read_fn=lambda model,filename: read_pkl(filename)

    test_provider = Provider(test_list,'test',FLAGS.batch_size*FLAGS.num_gpus,read_fn)

    try:
        pls=build_placeholder(FLAGS.num_gpus)
        pmiu=neighbor_anchors_v2()

        batch_num_per_epoch=11000/FLAGS.num_gpus
        ops=train_ops(pls['cxyzs'],pls['dxyzs'],pls['covars'],
                      pls['vlens'],pls['vlens_bgs'],pls['vcidxs'],
                      pls['cidxs'],pls['nidxs'],pls['nidxs_lens'],pls['nidxs_bgs'],
                      pls['lbls'],pls['weights'],pmiu.shape[1],pls['is_training'],
                      batch_num_per_epoch,pls['pmiu'])

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