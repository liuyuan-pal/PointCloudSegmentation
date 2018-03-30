import numpy as np
import tensorflow as tf
from test_util.draw_util import output_points
ops=tf.load_op_library('./ComputeVoxelIdxOp.so')
pn=30000

pts=np.random.uniform(-1.0,1.0,[pn,3])
pts[:,2]-=np.min(pts[:,2])

pts_pl=tf.placeholder(tf.float32,[None,3],'pts')
voxel_idxs=ops.compute_voxel_index(pts_pl,voxel_len=0.3,block_size=2.0)
with tf.Session() as sess:
    voxel_idxs_val=sess.run(voxel_idxs,feed_dict={pts_pl:pts})

print np.min(voxel_idxs_val,axis=0)
print np.max(voxel_idxs_val,axis=0)

voxel_idxs_set=[]
for i in xrange(pn):
    val=tuple(voxel_idxs_val[i])
    if val not in voxel_idxs_set:
        voxel_idxs_set.append(val)

print len(voxel_idxs_set)

colors=np.random.randint(0,256,[len(voxel_idxs_set),3])

single_idxs=[voxel_idxs_set.index(tuple(val)) for val in voxel_idxs_val]
single_idxs=np.asarray(single_idxs)

output_points('compute_voxel_index_result.txt',pts,colors[single_idxs])



