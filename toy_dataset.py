import numpy as np
from model_pgnet import *
from aug_util import compute_cidxs,compute_nidxs_bgs
import libPointUtil
import random
from io_util import get_block_train_test_split,read_pkl
import time

def build_model(radius):
    pls={}
    pls['xyzs']=tf.placeholder(tf.float32,[None,3])
    pls['covars']=tf.placeholder(tf.float32,[None,9])

    nidxs,nlens,nbegs,ncens=search_neighborhood(pls['xyzs'],radius)
    sxyzs = neighbor_ops.neighbor_scatter(pls['xyzs'], nidxs, nlens, nbegs, use_diff=True)  # [en,ifn]
    # pred_covars=ecd_xyz(sxyzs,[16,16],[16,16],9,nlens,nbegs,ncens,'xyz',False)
    point_feats=ecd_xyz_v2(sxyzs,[8,8],16,[8,8],[8,8],32,nlens,nbegs,ncens,'xyz',False)
    # point_feats=pointnet_conv(sxyzs,pls['xyzs'],[16,16,16,16],32,'xyz',nidxs,nlens,nbegs,ncens,False)
    pred_covars = tf.contrib.layers.fully_connected(point_feats, num_outputs=9,
                                                    scope='covars',activation_fn=None, reuse=False)
    loss=tf.reduce_mean(tf.reduce_sum(tf.squared_difference(pls['covars'],pred_covars),axis=1),axis=0)

    decay_steps = 500 * 3
    global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0), trainable=False)
    lr = tf.train.exponential_decay(1e-3, global_step, decay_steps, 0.5, staircase=True)
    lr = tf.maximum(1e-5,lr)
    tf.summary.scalar('learning rate', lr)
    optimizer=tf.train.AdamOptimizer(lr)
    train_op=optimizer.minimize(loss)

    ops={}
    ops['train']=train_op
    ops['pred_covars']=pred_covars
    ops['loss']=loss

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    return sess,pls,ops

def compute_covar(xyzs,radius):
    covar_nidxs=libPointUtil.findNeighborRadiusCPU(xyzs,radius)

    covar_nidxs_lens=np.ascontiguousarray([len(idxs) for idxs in covar_nidxs],np.int32)
    covar_nidxs_bgs=compute_nidxs_bgs(covar_nidxs_lens)
    covar_nidxs=np.ascontiguousarray(np.concatenate(covar_nidxs,axis=0),dtype=np.int32)

    covars=libPointUtil.computeCovarsGPU(xyzs,covar_nidxs,covar_nidxs_lens,covar_nidxs_bgs)

    return covars

radius=0.15
def train_one_epoch(epoch_num,data_set,sess,pls,ops):
    total_loss=[]
    random.shuffle(data_set)
    for i,xyz in enumerate(data_set):
        covars=compute_covar(xyz,radius)
        _,loss=sess.run([ops['train'],ops['loss']],{pls['xyzs']:xyz,pls['covars']:covars})
        total_loss.append(loss)
        if i %100==0:
            print 'epoch {} step {} loss {}'.format(epoch_num,i,np.mean(total_loss))

    print 'train end epoch {} loss {}'.format(epoch_num,np.mean(total_loss))

def test_one_epoch(epoch_num,data_set,sess,pls,ops):
    total_loss=[]
    for xyz in data_set:
        covars=compute_covar(xyz,radius)
        loss=sess.run(ops['loss'],{pls['xyzs']:xyz,pls['covars']:covars})
        total_loss.append(loss)

    print '///test epoch {} loss {}'.format(epoch_num,np.mean(total_loss))

def test_cluster_covar(xyz,covars,name):
    from sklearn.cluster import KMeans
    from draw_util import output_points
    kmeans=KMeans(5)
    colors=np.random.randint(0,256,[5,3])
    preds=kmeans.fit_predict(covars)
    output_points('test_result/{}.txt'.format(name),xyz,colors[preds])


def test_nn():
    xyzs=np.random.uniform(-1,1,[1024,3])
    xyzs=np.asarray(xyzs,np.float32)
    covar_nidxs = libPointUtil.findNeighborRadiusCPU(xyzs, radius)
    for k in xrange(1024):
        for pi,pt in enumerate(xyzs):
            if np.sum((pt-xyzs[k])**2,axis=0)<radius**2:
               assert pi in covar_nidxs[k]

        for ni in covar_nidxs[k]:
            assert np.sum((xyzs[ni]-xyzs[k])**2,axis=0)<radius**2


if __name__=="__main__":
    train_list,test_list=get_block_train_test_split()
    train_list = [fs for fs in train_list if fs.split('_')[3] == 'office']
    test_list = [fs for fs in test_list if fs.split('_')[3] == 'office']
    train_list = ['data/S3DIS/sampled_train_nolimits/' + fn for fn in train_list[:10]]
    test_list = ['data/S3DIS/sampled_test_nolimits/' + fn for fn in test_list[:5]]
    random.shuffle(train_list)

    # trainset=[]
    # for fs in train_list:
    #     cxyzs = read_pkl(fs)[0]
    #     trainset+=[cxyz[0] for cxyz in cxyzs]
    #
    # testset=[]
    # for fs in test_list:
    #     cxyzs = read_pkl(fs)[0]
    #     testset+=[cxyz[0] for cxyz in cxyzs]

    trainset=[]
    for fs in train_list:
        cxyzs = read_pkl(fs)[0]
        trainset+=cxyzs

    testset=[]
    for fs in test_list:
        cxyzs = read_pkl(fs)[0]
        testset+=cxyzs

    print len(trainset),len(testset)

    # for i,xyz in enumerate(trainset[:5]):
    #     covar=compute_covar(xyz,radius)
    #     test_cluster_covar(xyz,covar,i)

    sess, pls, ops = build_model(radius)
    sess.run(tf.global_variables_initializer())
    saver=tf.train.Saver()
    for epoch_num in xrange(100):
        bg=time.time()
        train_one_epoch(epoch_num,trainset,sess,pls,ops)
        print 'train cost {} s'.format(time.time()-bg)
        test_one_epoch(epoch_num,testset,sess,pls,ops)
        if epoch_num%5==0:
            saver.save(sess,'model/toy_pointnet/ecd{}.ckpt'.format(epoch_num))



