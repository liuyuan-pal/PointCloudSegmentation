import sys
sys.path.append("..")
sys.path.append("../..")
import libPointUtil
import tensorflow as tf
import numpy as np
import time

SearchNeighborhoodOp=tf.load_op_library('./SearchNeighborhoodOp.so')
ComputeVoxelIdxOp=tf.load_op_library('./ComputeVoxelIdxOp.so')


pn=1024
min_search_radius=0.1
max_search_radius=0.15
pts=np.random.uniform(-1.0,1.0,[pn,3])
pts-=np.min(pts,axis=0,keepdims=True)

pts_pl=tf.placeholder(tf.float32,[None,3],'pts')
idxs,lens,begs,cens=SearchNeighborhoodOp.search_neighborhood_brute_force_range(
    pts_pl,squared_min_nn_size=min_search_radius*min_search_radius,
    squared_max_nn_size=max_search_radius*max_search_radius)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
with tf.Session(config=config) as sess:
    nidxs,nlens,nbegs,ncens=sess.run([idxs,lens,begs,cens],feed_dict={pts_pl:pts})

print ncens[:10]
print nlens[:10]
print nbegs[:10]
for ni,nlen in enumerate(nlens):
    dist=np.sqrt(np.sum((np.expand_dims(pts[ni],axis=0)-
                         pts[nbegs[ni]:nbegs[ni]+nlen])**2,axis=1))
    mask=dist>max_search_radius
    mask=np.bitwise_and(mask,dist<min_search_radius)
    assert np.sum(mask)==0

