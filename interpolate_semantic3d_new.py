import tensorflow as tf
from model_pooling import points_pooling_two_layers,graph_conv_pool_edge_simp_2layers,classifier_v3
from train_gpn_semantic3d_new import build_placeholder,fill_feed_dict
from io_util import read_room_pkl,get_semantic3d_testset,read_pkl,save_pkl
from aug_util import compute_nidxs_bgs,rotate
from draw_util import output_points,get_semantic3d_class_colors
import numpy as np
import argparse
import libPointUtil
import time
import pyflann

parser = argparse.ArgumentParser()

parser.add_argument('--restore_model', type=str, default='', help='')
parser.add_argument('--rotation_index', type=int, default=0, help='')
parser.add_argument('--num_classes', type=int, default=9, help='')

FLAGS = parser.parse_args()


def build_network(xyzs, feats, labels, is_training):
    xyzs, dxyzs, feats, labels, vlens, vbegs, vcens = \
        points_pooling_two_layers(xyzs, feats, labels, voxel_size1=0.25, voxel_size2=1.0, block_size=10.0)
    global_feats, local_feats, ops = graph_conv_pool_edge_simp_2layers(xyzs, dxyzs, feats, vlens, vbegs, vcens,
                                                                       [0.25, 1.0], 10.0, [0.25, 0.5, 2.0], False)
    global_feats=tf.expand_dims(global_feats,axis=0)
    local_feats=tf.expand_dims(local_feats,axis=0)
    logits=classifier_v3(global_feats, local_feats, is_training, FLAGS.num_classes, False, use_bn=False)  # [1,pn,num_classes]

    flatten_logits=tf.reshape(logits,[-1,FLAGS.num_classes])  # [pn,num_classes]
    probs=tf.nn.softmax(flatten_logits)
    preds=tf.argmax(flatten_logits,axis=1)
    ops={}
    ops['feats']=feats
    ops['probs']=probs
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


def eval_room_probs(fn,sess,pls,ops,feed_dict,ri):
    all_xyzs, all_lbls, all_probs = [], [], []
    all_feed_in=read_pkl('data/Semantic3D.Net/block/test_{}/'.format(ri)+fn+'.pkl')
    for i in xrange(len(all_feed_in[0])):
        cur_feed_in=[[fi[i]] for fi in all_feed_in]
        block_min=all_feed_in[4][i]
        fill_feed_dict(cur_feed_in,feed_dict,pls,1)
        probs,sxyzs,lbls=sess.run([ops['probs'],ops['xyzs'],ops['labels']],feed_dict)
        all_xyzs.append(sxyzs+block_min)
        all_lbls.append(lbls)
        all_probs.append(probs)

    return np.concatenate(all_xyzs,axis=0),np.concatenate(all_lbls,axis=0),np.concatenate(all_probs,axis=0)


def interpolate(sxyzs,sprobs,qxyzs,ratio=1.0/(2*0.125*0.125)):
    nidxs=libPointUtil.findNeighborInAnotherCPU(sxyzs,qxyzs,0.125)
    nidxs_lens=np.asarray([len(idxs) for idxs in nidxs],dtype=np.int32)
    nidxs_bgs=compute_nidxs_bgs(nidxs_lens)
    nidxs=np.concatenate(nidxs,axis=0)
    qprobs = libPointUtil.interpolateProbsGPU(sxyzs, qxyzs, sprobs, nidxs, nidxs_lens, nidxs_bgs, ratio)

    return qprobs


def interpolate_scene(qxyzs,sxyzs,sprobs):
    qn=qxyzs.shape[0]
    rn=1000000
    qrn=qn/rn
    if qn%rn!=0: qrn+=1
    print 'qrn {} sxyzs num {}'.format(qrn,sxyzs.shape[0])

    sxyzs=np.ascontiguousarray(sxyzs)
    qxyzs=np.ascontiguousarray(qxyzs)
    sprobs=np.ascontiguousarray(sprobs)

    qprobs=[]
    for t in xrange(qrn):
        beg_idxs=t*rn
        end_idxs=min((t+1)*rn,qn)
        bg=time.time()
        qrprobs=interpolate(sxyzs,sprobs,qxyzs[beg_idxs:end_idxs])
        print 'interpolate {} done cost {} s'.format(t,time.time()-bg)
        qprobs.append(qrprobs)

    qprobs=np.concatenate(qprobs,axis=0)
    qpreds=np.argmax(qprobs[:,1:],axis=1)+1

    return qpreds


def interpolate_probs():
    fns,pns=get_semantic3d_testset()
    for fn,pn in zip(fns,pns):
        points, labels = read_room_pkl('data/Semantic3D.Net/pkl/test/' + fn + '.pkl')
        qxyzs = np.ascontiguousarray(points[:, :3], np.float32)
        sxyzs,sprobs=[],[]
        for t in xrange(6):
        # for t in [3]:
            sxyz,sprob=read_pkl('data/Semantic3D.Net/result/{}_{}.pkl'.format(fn,t))
            sxyzs.append(sxyz)
            sprobs.append(sprob)

        sxyzs=np.concatenate(sxyzs,axis=0)
        sprobs=np.concatenate(sprobs,axis=0)
        # sxyzs=sxyzs[0]
        # sprobs=sprobs[0]

        spreds = np.argmax(sprobs[:, 1:], axis=1) + 1
        qpreds=interpolate_scene(qxyzs, sxyzs, sprobs)

        with open('data/Semantic3D.Net/{}.labels'.format(fn),'w') as f:
            for pred in qpreds:
                f.write('{}\n'.format(pred))

        # output
        colors = get_semantic3d_class_colors()
        idxs = libPointUtil.gridDownsampleGPU(sxyzs, 0.1, False)
        sxyzs = sxyzs[idxs]
        spreds = spreds[idxs]
        output_points('test_result/{}_sparse.txt'.format(fn), sxyzs, colors[spreds, :])
        idxs = libPointUtil.gridDownsampleGPU(qxyzs, 0.1, False)
        qxyzs = qxyzs[idxs]
        qpreds = qpreds[idxs]
        output_points('test_result/{}_dense.txt'.format(fn), qxyzs, colors[qpreds, :])


def predict_block():
    fns,pns=get_semantic3d_testset()
    sess, pls, ops, feed_dict=build_session()
    # ri=FLAGS.rotation_index
    for ri in xrange(6):
        for fn,pn in zip(fns,pns):
            bg=time.time()
            sxyzs,slbls,sprobs=eval_room_probs(fn,sess,pls,ops,feed_dict,ri)
            # rotate back
            sxyzs=rotate(sxyzs,-np.pi/12.*ri)
            sxyzs=np.ascontiguousarray(sxyzs,np.float32)
            print '{} done cost {} s'.format(fn,time.time()-bg)
            # save
            save_pkl('data/Semantic3D.Net/result/{}_{}.pkl'.format(fn,ri),[sxyzs,sprobs])


if __name__=="__main__":
    predict_block()