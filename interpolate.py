import tensorflow as tf
from model import graph_conv_pool_edge_shallow,classifier_v3
from train_graph_pool import neighbor_anchors_v2
from io_util import read_pkl,get_block_train_test_split,read_room_pkl,get_class_names
from aug_util import compute_nidxs_bgs
from draw_util import output_points,get_class_colors
from train_util import val2iou,acc_val
import numpy as np
import argparse
import libPointUtil

parser = argparse.ArgumentParser()

parser.add_argument('--restore_model', type=str, default='', help='')
parser.add_argument('--num_classes', type=int, default=13, help='')

FLAGS = parser.parse_args()

def build_placeholder():
    pls = {}

    cxyzs = [tf.placeholder(tf.float32, [None, 3], 'cxyz{}'.format(j)) for j in xrange(3)]
    pls['cxyzs']=(cxyzs)
    pls['rgbs']=(tf.placeholder(tf.float32, [None, 3], 'rgb'))
    pls['covars']=(tf.placeholder(tf.float32, [None, 9], 'covar'))
    pls['lbls']=(tf.placeholder(tf.int64, [None], 'lbl'))

    nidxs = [tf.placeholder(tf.int32, [None], 'nidxs{}'.format(j)) for j in xrange(3)]
    nidxs_lens = [tf.placeholder(tf.int32, [None], 'nidxs_lens{}'.format(j)) for j in xrange(3)]
    nidxs_bgs = [tf.placeholder(tf.int32, [None], 'nidxs{}'.format(j)) for j in xrange(3)]
    cidxs = [tf.placeholder(tf.int32, [None], 'cidxs{}'.format(j)) for j in xrange(3)]
    pls['nidxs']=(nidxs)
    pls['nidxs_lens']=(nidxs_lens)
    pls['nidxs_bgs']=(nidxs_bgs)
    pls['cidxs']=(cidxs)

    vlens = [tf.placeholder(tf.int32, [None], 'vlens{}'.format(j)) for j in xrange(2)]
    vlens_bgs = [tf.placeholder(tf.int32, [None], 'vlens_bgs{}'.format(j)) for j in xrange(2)]
    vcidxs = [tf.placeholder(tf.int32, [None], 'vcidxs{}'.format(j)) for j in xrange(2)]
    dxyzs = [tf.placeholder(tf.float32, [None, 3], 'dxyzs{}'.format(j)) for j in xrange(2)]

    pls['vlens']=(vlens)
    pls['vlens_bgs']=(vlens_bgs)
    pls['vcidxs']=(vcidxs)
    pls['dxyzs']=(dxyzs)

    pls['is_training'] = tf.placeholder(tf.bool, name='is_training')
    pls['pmiu'] = tf.placeholder(tf.float32, name='pmiu')
    return pls

def fill_feed_dict(feed_in,feed_dict,pls,idx):
    cxyzs, dxyzs, rgbs, covars, lbls, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens, block_mins = feed_in
    for t in xrange(3):
        feed_dict[pls['cxyzs'][t]] = cxyzs[idx][t]
        feed_dict[pls['nidxs'][t]] = nidxs[idx][t]
        feed_dict[pls['nidxs_lens'][t]] = nidxs_lens[idx][t]
        feed_dict[pls['nidxs_bgs'][t]] = nidxs_bgs[idx][t]
        feed_dict[pls['cidxs'][t]] = cidxs[idx][t]

    feed_dict[pls['rgbs']] = rgbs[idx]
    feed_dict[pls['covars']] = covars[idx]
    feed_dict[pls['lbls']] = lbls[idx]

    for t in xrange(2):
        feed_dict[pls['dxyzs'][t]] = dxyzs[idx][t]
        feed_dict[pls['vlens'][t]] = vlens[idx][t]
        feed_dict[pls['vlens_bgs'][t]] = vlens_bgs[idx][t]
        feed_dict[pls['vcidxs'][t]] = vcidxs[idx][t]

    return cxyzs[idx][0],lbls[idx],block_mins[idx]

def build_network(cxyzs, dxyzs, rgbs, covars, vlens, vlens_bgs, vcidxs,
                   cidxs, nidxs, nidxs_lens, nidxs_bgs,
                   m, pmiu=None,is_training=None):

    rgb_covars = tf.concat([rgbs, covars], axis=1)
    feats,lf=graph_conv_pool_edge_shallow(cxyzs, dxyzs, rgb_covars, vlens, vlens_bgs, vcidxs,
                                   cidxs, nidxs, nidxs_lens, nidxs_bgs, m, pmiu, False)
    feats=tf.expand_dims(feats,axis=0)
    lf=tf.expand_dims(lf,axis=0)
    logits=classifier_v3(feats, lf, is_training, FLAGS.num_classes, False, use_bn=False)  # [1,pn,num_classes]

    flatten_logits=tf.reshape(logits,[-1,FLAGS.num_classes])  # [pn,num_classes]
    probs=tf.nn.softmax(flatten_logits)
    preds=tf.argmax(flatten_logits,axis=1)
    ops={}
    ops['feats']=feats
    ops['probs']=probs
    ops['logits']=flatten_logits
    ops['preds']=preds
    return ops

def build_session():
    pls=build_placeholder()
    pmiu=neighbor_anchors_v2()
    ops=build_network(pls['cxyzs'],pls['dxyzs'],pls['rgbs'],pls['covars'],
                      pls['vlens'],pls['vlens_bgs'],pls['vcidxs'],
                      pls['cidxs'],pls['nidxs'],pls['nidxs_lens'],pls['nidxs_bgs'],
                      pmiu.shape[1],pls['pmiu'],pls['is_training'])

    feed_dict=dict()
    feed_dict[pls['pmiu']]=pmiu
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

def _fetch_all_feed_in_idx(all_feed_in,idx):
    feed_in=[]
    for afi in all_feed_in:
        feed_in.append([afi[idx]])
    return feed_in

colors=get_class_colors()
def eval_room_probs(fn,sess,pls,ops,feed_dict):
    all_feed_in=read_pkl('data/S3DIS/sampled_test/'+fn)
    all_xyzs,all_lbls,all_probs=[],[],[]
    for i in xrange(len(all_feed_in[0])):
        sxyzs,lbls,block_mins=fill_feed_dict(all_feed_in,feed_dict,pls,i)
        probs=sess.run(ops['probs'],feed_dict)

        all_xyzs.append(sxyzs+block_mins)
        all_lbls.append(lbls)
        all_probs.append(probs)

    return np.concatenate(all_xyzs,axis=0),np.concatenate(all_lbls,axis=0),np.concatenate(all_probs,axis=0)


def interpolate(sxyzs,sprobs,qxyzs,ratio=1.0/(2*0.075*0.075)):
    nidxs=libPointUtil.findNeighborInAnotherCPU(sxyzs,qxyzs,6)
    nidxs_lens=np.asarray([len(idxs) for idxs in nidxs],dtype=np.int32)
    nidxs_bgs=compute_nidxs_bgs(nidxs_lens)
    nidxs=np.concatenate(nidxs,axis=0)
    qprobs = libPointUtil.interpolateProbsGPU(sxyzs, qxyzs, sprobs, nidxs, nidxs_lens, nidxs_bgs, ratio)

    return qprobs

if __name__=="__main__":
    train_list,test_list=get_block_train_test_split()
    sess, pls, ops, feed_dict=build_session()
    all_preds,all_labels=[],[]
    fp = np.zeros(13, dtype=np.uint64)
    tp = np.zeros(13, dtype=np.uint64)
    fn = np.zeros(13, dtype=np.uint64)
    for fs in test_list:
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

        # print np.min(sxyzs,axis=0)
        # print np.min(qxyzs,axis=0)

        colors=get_class_colors()
        spreds=np.argmax(sprobs,axis=1)
        # print labels.shape
        # print qpreds.shape
        fp, tp, fn=acc_val(labels.flatten(),qpreds.flatten(),fp,tp,fn)
        # output_points('test_result/spreds.txt',sxyzs,colors[spreds,:])
        # output_points('test_result/slabel.txt',sxyzs,colors[slbls.flatten(),:])
        # output_points('test_result/qpreds.txt',qxyzs,colors[qpreds,:])
        # output_points('test_result/qlabel.txt',qxyzs,colors[labels.flatten(),:])
        # break

    iou, miou, oiou, acc, macc, oacc=val2iou(fp,tp,fn)
    print 'mean iou {:.5} overall iou {:5} \nmean acc {:5} overall acc {:5}'.format(miou, oiou, macc, oacc)

    names = get_class_names()
    for i in xrange(len(names)):
        print '{} iou {} acc {}'.format(names[i], iou[i], acc[i])
