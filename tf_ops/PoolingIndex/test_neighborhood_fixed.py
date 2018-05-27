import sys
sys.path.append("..")
sys.path.append("../..")
import libPointUtil
import tensorflow as tf
import numpy as np
import time

SearchNeighborhoodOp=tf.load_op_library('./SearchNeighborhoodFixedOp.so')

pn=4096
min_search_radius=0.1
max_search_radius=0.15
pts=np.random.uniform(-1.0,1.0,[pn,3])
pts-=np.min(pts,axis=0,keepdims=True)

pts_pl=tf.placeholder(tf.float32,[None,3],'pts')
idxs,lens,begs,cens=SearchNeighborhoodOp.search_neighborhood_fixed_brute_force_range(
    pts_pl,squared_min_nn_size=min_search_radius*min_search_radius,
    squared_max_nn_size=max_search_radius*max_search_radius,fixed_size=10)
ridxs,rlens,rbegs,rcens=SearchNeighborhoodOp.search_neighborhood_fixed_brute_force(
    pts_pl,squared_nn_size=min_search_radius*min_search_radius,fixed_size=10)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
with tf.Session(config=config) as sess:
    nidxs,nlens,nbegs,ncens,rnidxs,rnlens,rnbegs,rncens=sess.run([idxs,lens,begs,cens,
                                      ridxs, rlens, rbegs, rcens],feed_dict={pts_pl:pts})

for ni,nlen in enumerate(nlens):
    cur_nidxs=nidxs[nbegs[ni]:nbegs[ni] + nlen]
    cur_nidxs=cur_nidxs[cur_nidxs!=ni]
    if len(cur_nidxs)==0: continue
    dist=np.sqrt(np.sum((np.expand_dims(pts[ni],axis=0)-pts[cur_nidxs])**2,axis=1))
    mask=dist>max_search_radius
    mask=np.logical_or(mask,dist<min_search_radius)
    assert np.sum(mask)==0

print '/////////////'

for ni,nlen in enumerate(rnlens):
    cur_nidxs=rnidxs[rnbegs[ni]:rnbegs[ni] + nlen]
    dist=np.sqrt(np.sum((np.expand_dims(pts[ni],axis=0)-pts[cur_nidxs])**2,axis=1))
    mask=dist>min_search_radius
    assert np.sum(mask)==0

