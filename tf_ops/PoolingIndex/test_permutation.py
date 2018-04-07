import numpy as np
import tensorflow as tf
from test_util.draw_util import output_points

ComputeVoxelIdxOp=tf.load_op_library('./ComputeVoxelIdxOp.so')
ComputePermutationInfoOp=tf.load_op_library('./ComputePermutationInfoOp.so')
ComputeRepermutationInfoOp=tf.load_op_library('./ComputeRepermutationInfoOp.so')
PermutateFeatureOp=tf.load_op_library('./PermutateFeatureOp.so')
neighbor_ops=tf.load_op_library('../build/libTFNeighborOps.so')
ComputeDiffXYZOp=tf.load_op_library('./ComputeDiffXYZOp.so')


def eval_tf_permutation(pts,sess):
    pts_pl=tf.placeholder(tf.float32,[None,3],'pts') # pn1
    vidxs1=ComputeVoxelIdxOp.compute_voxel_index(pts_pl, voxel_len=0.3, block_size=2.0)
    o2p_idxs1, p2o_idxs1, vlens1, vbegs1, vcens1=ComputePermutationInfoOp.compute_permutation_info(vidxs1)
    pts1=PermutateFeatureOp.permutate_feature(pts_pl, o2p_idxs1, p2o_idxs1) # pn1

    pts2=neighbor_ops.neighbor_sum_feat_gather(pts1, vcens1, vlens1, vbegs1) # pn2
    pts2= pts2 / tf.expand_dims(tf.cast(vlens1, tf.float32), axis=1)
    dpts1=ComputeDiffXYZOp.compute_diff_xyz(pts1, pts2, vcens1)

    vidxs2=ComputeVoxelIdxOp.compute_voxel_index(pts2, voxel_len=0.75, block_size=2.0)
    o2p_idxs2,p2o_idxs2,vlens2,vbegs2,vcens2=ComputePermutationInfoOp.compute_permutation_info(vidxs2)
    pts2=PermutateFeatureOp.permutate_feature(pts2,o2p_idxs2,p2o_idxs2) # pn2

    pts3=neighbor_ops.neighbor_sum_feat_gather(pts2, vcens2, vlens2, vbegs2) # pn2
    pts3= pts3 / tf.expand_dims(tf.cast(vlens2, tf.float32), axis=1)
    dpts2=ComputeDiffXYZOp.compute_diff_xyz(pts2, pts3, vcens2)

    reper_o2p_idxs1,reper_p2o_idxs1,vlens1,vbegs1,vcens1=\
        ComputeRepermutationInfoOp.compute_repermutation_info(o2p_idxs2,vlens1,vbegs1,vcens1)
    pts1=PermutateFeatureOp.permutate_feature(pts1,reper_o2p_idxs1,reper_p2o_idxs1)
    dpts1=PermutateFeatureOp.permutate_feature(dpts1,reper_o2p_idxs1,reper_p2o_idxs1)

    xyzs1,xyzs2,xyzs3,dxyzs1,dxyzs2,\
        vl1,vb1,vc1,vl2,vb2,vc2=sess.run(
        [pts1,pts2,pts3,dpts1,dpts2,
         vlens1,vbegs1,vcens1,
         vlens2,vbegs2,vcens2],
        feed_dict={pts_pl:pts})

    return xyzs1,xyzs2,xyzs3,dxyzs1,dxyzs2


def test_single():
    pn=30000

    pts=np.random.uniform(-1.0,1.0,[pn,3])
    pts[:,2]-=np.min(pts[:,2])
    pts[:,2]=0

    # assert cens lens begs

    def check_vidxs(lens,begs,cens):
        pn2=lens.shape[0]
        pn1=cens.shape[0]

        cur_len=0
        for i in xrange(pn2):
            assert cur_len==begs[i]
            cur_len+=lens[i]

        for i in xrange(pn2):
            for j in xrange(lens[i]):
                assert cens[begs[i]+j]==i

    check_vidxs(vl1,vb1,vc1)
    check_vidxs(vl2,vb2,vc2)

    def output_hierarchy(pts1,pts2,cens,name):
        colors=np.random.randint(0,256,[len(pts2),3])
        output_points('test_result/{}_dense.txt'.format(name),pts1,colors[cens,:])
        output_points('test_result/{}_sparse.txt'.format(name),pts2,colors)

    output_hierarchy(xyzs1,xyzs2,vc1,'12')
    output_hierarchy(xyzs2,xyzs3,vc2,'23')

    def check_dxyzs(pts1,pts2,dpts1,vcens):
        pn1=pts1.shape[0]
        tmp_dpts1=np.copy(dpts1)
        for i in xrange(pn1):
            tmp_dpts1[i]+=pts2[vcens[i]]

        print np.mean(np.abs(tmp_dpts1-pts1),axis=0),np.max(np.abs(tmp_dpts1-pts1),axis=0)

    check_dxyzs(xyzs1,xyzs2,dxyzs1,vc1)
    check_dxyzs(xyzs2,xyzs3,dxyzs2,vc2)





