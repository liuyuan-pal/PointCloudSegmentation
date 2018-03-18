import tensorflow as tf
from model import graph_conv_pool_v3,classifier_v3
from train_graph_pool import neighbor_anchors_v2
from io_util import read_room_pkl,get_semantic3d_testset,read_pkl
from aug_util import compute_nidxs_bgs
from draw_util import output_points,get_semantic3d_class_colors
import numpy as np
import argparse
import libPointUtil
import time
import pyflann

parser = argparse.ArgumentParser()

parser.add_argument('--restore_model', type=str, default='', help='')
parser.add_argument('--num_classes', type=int, default=13, help='')

FLAGS = parser.parse_args()

def build_placeholder():
    pls = {}

    cxyzs = [tf.placeholder(tf.float32, [None, 3], 'cxyz{}'.format(j)) for j in xrange(3)]
    pls['cxyzs']=(cxyzs)
    pls['rgbs']=(tf.placeholder(tf.float32, [None, 4], 'rgb'))
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
    feats,lf=graph_conv_pool_v3(cxyzs, dxyzs, rgb_covars, vlens, vlens_bgs, vcidxs,
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

def eval_room_probs(fn,sess,pls,ops,feed_dict):
    all_xyzs, all_lbls, all_probs = [], [], []
    all_feed_in=read_pkl('data/Semantic3D.Net/block/test/'+fn+'.pkl')
    for i in xrange(len(all_feed_in[0])):
        sxyzs,lbls,block_mins=fill_feed_dict(all_feed_in,feed_dict,pls,i)
        probs=sess.run(ops['probs'],feed_dict)
        all_xyzs.append(sxyzs+block_mins)
        all_lbls.append(lbls)
        all_probs.append(probs)

    return np.concatenate(all_xyzs,axis=0),np.concatenate(all_lbls,axis=0),np.concatenate(all_probs,axis=0)


def interpolate(sxyzs,sprobs,qxyzs,ratio=1.0/(2*0.125*0.125)):
    bg=time.time()
    nidxs=libPointUtil.findNeighborInAnotherCPU(sxyzs,qxyzs,0.25)
    print 'search done {} s'.format(time.time()-bg)
    nidxs_lens=np.asarray([len(idxs) for idxs in nidxs],dtype=np.int32)
    nidxs_bgs=compute_nidxs_bgs(nidxs_lens)
    nidxs=np.concatenate(nidxs,axis=0)
    qprobs = libPointUtil.interpolateProbsGPU(sxyzs, qxyzs, sprobs, nidxs, nidxs_lens, nidxs_bgs, ratio)

    return qprobs

if __name__=="__main__":
    fns,pns=get_semantic3d_testset()

    sess, pls, ops, feed_dict=build_session()
    all_preds,all_labels=[],[]
    for fn,pn in zip(fns,pns):
        begin=time.time()
        sxyzs,slbls,sprobs=eval_room_probs(fn,sess,pls,ops,feed_dict)


        points,labels=read_room_pkl('data/Semantic3D.Net/pkl/test/'+fn+'.pkl')
        qxyzs=np.ascontiguousarray(points[:,:3],np.float32)

        qn=qxyzs.shape[0]
        rn=1000000
        qrn=qn/rn
        if qn%rn!=0: qrn+=1
        print 'qrn {} sxyzs num {}'.format(qrn,sxyzs.shape[0])

        qprobs=[]
        for t in xrange(qrn):
            beg_idxs=t*rn
            end_idxs=min((t+1)*rn,qn)
            qrprobs=interpolate(sxyzs,sprobs,qxyzs[beg_idxs:end_idxs])
            print 'interpolate {} done'.format(t)
            qprobs.append(qrprobs)

        qprobs=np.concatenate(qprobs,axis=0)
        qpreds=np.argmax(qprobs[:,1:],axis=1)+1

        colors=get_semantic3d_class_colors()
        spreds=np.argmax(sprobs[:,1:],axis=1)+1

        print 'total cost {} s'.format(time.time() - begin)

        with open('data/Semantic3D.Net/{}.labels'.format(fn),'w') as f:
            for p in qpreds:
                f.write('{}\n'.format(p))


        idxs=libPointUtil.gridDownsampleGPU(sxyzs,0.1,False)
        sxyzs=sxyzs[idxs]
        spreds=spreds[idxs]
        output_points('test_result/{}_sparse.txt'.format(fn),sxyzs,colors[spreds,:])
        idxs=libPointUtil.gridDownsampleGPU(qxyzs,0.1,False)
        qxyzs=qxyzs[idxs]
        qpreds=qpreds[idxs]
        output_points('test_result/{}_dense.txt'.format(fn),qxyzs,colors[qpreds,:])





