import tensorflow as tf
from model_pgnet import *
from model_pointnet import *
from io_util import read_pkl,get_block_train_test_split,read_room_pkl,get_class_names
from aug_util import compute_nidxs_bgs
from draw_util import output_points,get_class_colors
from train_util import val2iou,acc_val
import numpy as np
import argparse
import libPointUtil
from train_graph_pool_fixed import build_placeholder,fill_feed_dict,test_fn

parser = argparse.ArgumentParser()

parser.add_argument('--restore_model', type=str, default='', help='')
parser.add_argument('--num_classes', type=int, default=13, help='')

FLAGS = parser.parse_args()


def build_network(xyzs, feats, labels, is_training, reuse=False):
    xyzs, dxyzs, feats, labels, vlens, vbegs, vcens = \
        points_pooling_two_layers(xyzs,feats,labels,voxel_size1=0.15,voxel_size2=0.45,block_size=3.0)
    global_feats, local_feats = pointnet_13_dilated_embed_fixed(xyzs,dxyzs,feats,vlens,vbegs,vcens,reuse)

    global_feats=tf.expand_dims(global_feats,axis=0)
    local_feats=tf.expand_dims(local_feats,axis=0)
    logits=classifier_v3(global_feats, local_feats, is_training, FLAGS.num_classes, False, use_bn=False)  # [1,pn,num_classes]

    flatten_logits=tf.reshape(logits,[-1,FLAGS.num_classes])  # [pn,num_classes]
    probs=tf.nn.softmax(flatten_logits)
    preds=tf.argmax(flatten_logits,axis=1)
    ops={}
    ops['feats']=feats
    ops['probs']=probs
    ops['logits']=flatten_logits
    ops['preds']=preds
    ops['labels']=labels
    ops['xyzs']=xyzs[0]
    return ops

def build_session():
    pls=build_placeholder(1)
    ops=build_network(pls['xyzs'][0],pls['feats'][0],pls['lbls'][0],pls['is_training'])

    feed_dict=dict()
    feed_dict[pls['is_training']]=False
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)
    saver = tf.train.Saver(max_to_keep=500)
    print 'restoring {}'.format(FLAGS.restore_model)
    saver.restore(sess,FLAGS.restore_model)
    return sess, pls, ops, feed_dict


def eval_room_probs(fn,sess,pls,ops,feed_dict):
    all_feed_in=test_fn('test','data/S3DIS/sampled_test_nolimits/'+fn)
    all_xyzs,all_lbls,all_probs=[],[],[]
    for i in xrange(len(all_feed_in[0])):
        feed_in=[]
        for fi in all_feed_in:
            feed_in.append([fi[i]])

        _,_,block_mins=fill_feed_dict(feed_in,feed_dict,pls,1)
        probs,sxyzs,lbls=sess.run([ops['probs'],ops['xyzs'],ops['labels']],feed_dict)

        all_xyzs.append(sxyzs+block_mins[0])
        all_lbls.append(lbls)
        all_probs.append(probs)

    return np.concatenate(all_xyzs,axis=0),np.concatenate(all_lbls,axis=0),np.concatenate(all_probs,axis=0)


def interpolate(sxyzs,sprobs,qxyzs,ratio=1.0/(2*0.075*0.075)):
    print sxyzs.shape
    print qxyzs.shape
    nidxs=libPointUtil.findNeighborInAnotherCPU(sxyzs,qxyzs,6)
    nidxs_lens=np.asarray([len(idxs) for idxs in nidxs],dtype=np.int32)
    nidxs_bgs=compute_nidxs_bgs(nidxs_lens)
    nidxs=np.concatenate(nidxs,axis=0)
    qprobs = libPointUtil.interpolateProbsGPU(sxyzs, qxyzs, sprobs, nidxs, nidxs_lens, nidxs_bgs, ratio)

    return qprobs

if __name__=="__main__":
    import random
    train_list,test_list=get_block_train_test_split()
    sess, pls, ops, feed_dict=build_session()
    all_preds,all_labels=[],[]
    fp = np.zeros(13, dtype=np.uint64)
    tp = np.zeros(13, dtype=np.uint64)
    fn = np.zeros(13, dtype=np.uint64)
    random.shuffle(test_list)
    for fi,fs in enumerate(test_list):
        sxyzs,slbls,sprobs=eval_room_probs(fs,sess,pls,ops,feed_dict)
        filename='data/S3DIS/room_block_10_10/'+fs
        points,labels=read_pkl(filename)
        qxyzs=np.ascontiguousarray(points[:,:3],np.float32)
        qn=qxyzs.shape[0]
        rn=1000000
        qrn=qn/rn
        if qn%rn!=0: qrn+=1
        # print 'qrn {} sxyzs num {}'.format(qrn,sxyzs.shape[0])
        qprobs=[]
        for t in xrange(qrn):
            beg_idxs=t*rn
            end_idxs=min((t+1)*rn,qn)
            qrprobs=interpolate(sxyzs,sprobs,qxyzs[beg_idxs:end_idxs])
            qprobs.append(qrprobs)

        qprobs=np.concatenate(qprobs,axis=0)
        qpreds=np.argmax(qprobs,axis=1)

        colors=get_class_colors()
        spreds=np.argmax(sprobs,axis=1)

        fp, tp, fn=acc_val(labels.flatten(),qpreds.flatten(),fp,tp,fn)

        # if fi<=5:
        #     idxs=libPointUtil.gridDownsampleGPU(sxyzs,0.01,False)
        #     sxyzs=sxyzs[idxs]
        #     spreds=spreds[idxs]
        #     slbls=slbls[idxs]
        #     output_points('test_result/{}spreds.txt'.format(fi),sxyzs,colors[spreds,:])
        #     output_points('test_result/{}slabel.txt'.format(fi),sxyzs,colors[slbls,:])
        #
        #     idxs=libPointUtil.gridDownsampleGPU(qxyzs,0.01,False)
        #     qxyzs=qxyzs[idxs]
        #     qpreds=qpreds[idxs]
        #     labels=labels[idxs]
        #     points=points[idxs]
        #     output_points('test_result/{}qpreds.txt'.format(fi),qxyzs,colors[qpreds,:])
        #     output_points('test_result/{}qlabel.txt'.format(fi),qxyzs,colors[labels.flatten(),:])
        #     output_points('test_result/{}qcolor.txt'.format(fi),points)

        # iou, miou, oiou, acc, macc, oacc=val2iou(fp,tp,fn)
        # print 'mean iou {:.5} overall iou {:5} \nmean acc {:5} overall acc {:5}'.format(miou, oiou, macc, oacc)
        # break

    iou, miou, oiou, acc, macc, oacc=val2iou(fp,tp,fn)
    print 'mean iou {:.5} overall iou {:5} \nmean acc {:5} overall acc {:5}'.format(miou, oiou, macc, oacc)

    names = get_class_names()
    for i in xrange(len(names)):
        print '{} iou {} acc {}'.format(names[i], iou[i], acc[i])
