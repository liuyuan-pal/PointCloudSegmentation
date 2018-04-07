import sys
sys.path.append("..")
sys.path.append("../..")
import libPointUtil
import tensorflow as tf
import numpy as np
import time

SearchNeighborhoodOp=tf.load_op_library('./SearchNeighborhoodOp.so')
ComputeVoxelIdxOp=tf.load_op_library('./ComputeVoxelIdxOp.so')


pn=10240
search_radius=0.1
pts=np.random.uniform(-1.0,1.0,[pn,3])
pts-=np.min(pts,axis=0,keepdims=True)
pts[:,:2]-=1.1
pts[:,2]=0.0
print np.min(pts,axis=0)

pts_pl=tf.placeholder(tf.float32,[None,3],'pts')
idxs,lens,begs,cens=SearchNeighborhoodOp.search_neighborhood_brute_force(pts_pl,squared_nn_size=search_radius*search_radius)

voxel_idxs=ComputeVoxelIdxOp.compute_voxel_index(pts_pl,voxel_len=search_radius,block_size=2.0)
idxs2,lens2,begs2=SearchNeighborhoodOp.search_neighborhood_with_bins(pts_pl,voxel_idxs,bin_len=search_radius,squared_nn_size=search_radius*search_radius)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
with tf.Session(config=config) as sess:
    bg=time.time()
    for i in xrange(10):
        nidxs,nlens,nbegs,ncens=sess.run([idxs,lens,begs,cens],feed_dict={pts_pl:pts})
    print 'cost {} s'.format(time.time()-bg)

    bg=time.time()
    for i in xrange(10):
        nidxs2,nlens2,nbegs2=sess.run([idxs2,lens2,begs2],feed_dict={pts_pl:pts})
    print 'cost {} s'.format(time.time()-bg)



print nlens[:10],nbegs[:10]
print nlens[-1]+nbegs[-1]
print nlens2[:10],nbegs2[:10]
print nlens2[-1]+nbegs2[-1]

pts=np.asarray(pts,np.float32)
bg=time.time()
for i in xrange(10):
    nidxs_py=libPointUtil.findNeighborRadiusGPU(pts, search_radius*search_radius, 15)
print 'cost {} s'.format(time.time()-bg)

print 'radius new:'
for i in xrange(10):
    print np.mean(np.sqrt(np.sum((pts[nidxs[nbegs[i]:nbegs[i]+nlens[i]]]-pts[i,:])**2,axis=1)))


print 'radius origin:'
for i in xrange(10):
    print np.mean(np.sqrt(np.sum((pts[nidxs_py[i]]-pts[i,:])**2,axis=1)))

print len(np.concatenate(nidxs_py,axis=0))

for i in xrange(pn):
    for j in xrange(nlens[i]):
        assert ncens[nbegs[i]+j]==i

