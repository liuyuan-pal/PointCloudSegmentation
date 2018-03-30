import numpy as np
import tensorflow as tf
from test_util.draw_util import output_points

ComputeVoxelIdxOp=tf.load_op_library('./ComputeVoxelIdxOp.so')
ComputePermutationInfoOp=tf.load_op_library('./ComputePermutationInfoOp.so')
PermutateFeatureOp=tf.load_op_library('./PermutateFeatureOp.so')
neighbor_ops=tf.load_op_library('../build/libTFNeighborOps.so')
ComputeDiffXYZOp=tf.load_op_library('./ComputeDiffXYZOp.so')

pn=30000

pts=np.random.uniform(-1.0,1.0,[pn,3])
pts[:,2]-=np.min(pts[:,2])

pts_pl=tf.placeholder(tf.float32,[None,3],'pts')
voxel_idxs=ComputeVoxelIdxOp.compute_voxel_index(pts_pl,voxel_len=0.3,block_size=2.0)
origin2permutation_idxs,permutation2origin_idxs,voxel_idxs_lens,voxel_idxs_begs,voxel_idxs_cens\
    =ComputePermutationInfoOp.compute_permutation_info(voxel_idxs)
permutated_pts=PermutateFeatureOp.permutate_feature(pts_pl,origin2permutation_idxs)
repermutated_pts=PermutateFeatureOp.permutate_feature(permutated_pts,permutation2origin_idxs)
center_pts=neighbor_ops.neighbor_sum_feat_gather(permutated_pts,voxel_idxs_cens,voxel_idxs_lens,voxel_idxs_begs) # [vn,3]
center_pts=center_pts/tf.expand_dims(tf.cast(voxel_idxs_lens,tf.float32),axis=1)

diff_pts=ComputeDiffXYZOp.compute_diff_xyz(pts_pl,center_pts,voxel_idxs_cens)

with tf.Session() as sess:
    vlens,vbegs,ppts,rppts,cpts,dpts=sess.run([
        voxel_idxs_lens,voxel_idxs_begs,
        permutated_pts,repermutated_pts,center_pts,diff_pts],
        feed_dict={pts_pl:pts})

# test vlens permutate
colors=np.random.randint(0,256,[len(vlens),3])
pcolors=[]
for c,l in zip(colors,vlens):
    pcolors+=[c for _ in xrange(l)]

pcolors=np.asarray(pcolors,np.int32)
output_points('test_result/permutated_colors.txt',ppts,pcolors)

# test permutate back
print 'reper max {} mean {} sum {}'.format(np.max(rppts-pts),np.mean(rppts-pts),np.sum(rppts-pts))

# test begs
cur_len=0
for i in xrange(len(vlens)):
    assert cur_len==vbegs[i]
    cur_len+=vlens[i]

# test cxyzs
output_points('test_result/center_colors.txt',cpts,colors)

# test dxyzs
for i in xrange(len(vlens)):
    bg=vbegs[i]
    ed=bg+vlens[i]
    dpts[bg:ed]+=cpts[i]

print 'diff max {} mean {} sum {}'.format(np.max(dpts-pts),np.mean(dpts-pts),np.sum(dpts-pts))






