import sys
sys.path.append('..')
from graph_conv_layer import *
import tensorflow as tf
import numpy as np
from data_util import *
import libPointUtil
import time
from tensorflow.python.client import timeline
final_dims=128

def neighbor_anchors():
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


def conv_xyz_feats_model(xyz, feats, nidxs, nidxs_lens, nidxs_bgs,pmiu,dpfeats):
    pfeats0,lw=graph_conv_xyz(xyz,nidxs,nidxs_lens,nidxs_bgs,'conv0',3,26,16,pmiu=pmiu)
    pfeats0=tf.concat([pfeats0,feats],axis=1)
    pfeats1=graph_conv_feats(pfeats0,nidxs,nidxs_lens,nidxs_bgs,'conv1',19,26,32,lw=lw)
    pfeats2=graph_conv_feats(pfeats1,nidxs,nidxs_lens,nidxs_bgs,'conv2',32,26,64,lw=lw)
    pfeats=graph_conv_feats(pfeats2,nidxs,nidxs_lens,nidxs_bgs,'conv3',64,26,final_dims,lw=lw)
    print 'trainable vars:'
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES): print var
    grads=tf.gradients(pfeats,tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),dpfeats)
    return pfeats,grads


def eval_conv_xyz_feats(xyz, feats, nidxs, nidxs_lens, nidxs_bgs, pmiu, dpfeats):
    xyz_pl=tf.placeholder(tf.float32,[None,3],'xyz')
    feats_pl=tf.placeholder(tf.float32,[None,3],'feats')
    nidxs_pl=tf.placeholder(tf.int32,[None],'nidxs')
    nidxs_lens_pl=tf.placeholder(tf.int32,[None],'nidxs_lens')
    nidxs_bgs_pl=tf.placeholder(tf.int32,[None],'nidxs_bgs')
    pmiu_pl=tf.placeholder(tf.float32,[3,26],'pmiu')
    dpfeats_pl=tf.placeholder(tf.float32,[None,final_dims],'pmiu')

    pfeats,grads=conv_xyz_feats_model(xyz_pl, feats_pl, nidxs_pl, nidxs_lens_pl, nidxs_bgs_pl, pmiu_pl, dpfeats_pl)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.Session(config=config) as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        sess.run(tf.global_variables_initializer())
        eval_pls=[pfeats]
        eval_pls+=grads
        for pl in eval_pls: print pl
        begin=time.time()
        for i in xrange(10):
            vals=sess.run(
                eval_pls, feed_dict={
                    xyz_pl: xyz,
                    feats_pl: feats,
                    nidxs_pl: nidxs,
                    nidxs_lens_pl: nidxs_lens,
                    nidxs_bgs_pl: nidxs_bgs,
                    pmiu_pl: pmiu,
                    dpfeats_pl: dpfeats,
                },options=run_options, run_metadata=run_metadata)

        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline3.json', 'w') as f:
            f.write(ctf)

        print 'cost {} s'.format((time.time()-begin)/10)

    return vals


if __name__ == "__main__":
    pts,_=read_room_h5('data/S3DIS/room/16_Area_1_office_15.h5')

    ds_idxs,_=libPointUtil.gridDownsampleGPU(pts,0.05,True)
    pts=pts[ds_idxs,:]
    block_idxs=libPointUtil.sampleRotatedBlockGPU(pts,1.5,3.0,0.0)
    block_idxs=[idxs for idxs in block_idxs if len(idxs)>2048]
    print 'mean block pt_num: {}'.format(np.mean(np.asarray([len(idxs) for idxs in block_idxs])))
    # bid=np.random.randint(0,len(block_idxs))
    pts=pts[block_idxs[0],:]

    spts=np.ascontiguousarray(pts[:,:3])
    nidxs=libPointUtil.findNeighborRadiusGPU(spts,0.1)
    nidxs_lens=np.asarray([len(idxs) for idxs in nidxs],dtype=np.int32)
    nidxs_bgs=compute_nidxs_bgs(nidxs_lens)
    nidxs=np.concatenate(nidxs,axis=0)
    print 'pn*n: {}'.format(nidxs.shape)
    print 'pn: {}'.format(nidxs_bgs.shape)
    print 'avg n: {}'.format(float(nidxs.shape[0])/nidxs_bgs.shape[0])

    pmiu=neighbor_anchors()
    pmiu=pmiu.transpose()
    print 'pmiu {}'.format(pmiu.shape)

    dpfeats=np.random.uniform(-1.0,1.0,[pts.shape[0],final_dims])
    dpfeats=np.asarray(dpfeats,dtype=np.float32)
    vals=eval_conv_xyz_feats(pts[:,:3],pts[:,3:],nidxs,nidxs_lens,nidxs_bgs,pmiu,dpfeats)


