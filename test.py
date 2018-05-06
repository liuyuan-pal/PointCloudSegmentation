from model_pgnet import *
from io_util import read_pkl,get_block_train_test_split

def build_model():
    pls={}
    pls['xyzs'] = tf.placeholder(tf.float32, [None, 3], 'xyzs')
    pls['feats'] = tf.placeholder(tf.float32, [None, 3], 'feats')
    pls['logits_grad'] = tf.placeholder(tf.float32, [None, 13], 'logits_grad')
    pls['labels'] = tf.placeholder(tf.int32, [None], 'labels')
    pls['is_training'] = tf.placeholder(tf.bool, name='is_training')


    xyzs, dxyzs, feats, labels, vlens, vbegs, vcens = \
        points_pooling_two_layers(pls['xyzs'],pls['feats'],pls['labels'],voxel_size1=0.15,voxel_size2=0.3,block_size=3.0)
    global_feats,local_feats, ops_op=pgnet_model_v3_bug(
        xyzs, dxyzs, feats, vlens, vbegs, vcens, [0.15,0.3], 3.0, [0.15,0.3,0.5], False)

    global_feats = tf.expand_dims(global_feats, axis=0)
    local_feats = tf.expand_dims(local_feats, axis=0)
    logits = classifier_v3(global_feats, local_feats, pls['is_training'], 13, False, use_bn=False)  # [1,pn,num_classes]

    flatten_logits = tf.reshape(logits, [-1, 13])  # [pn,num_classes]
    grads=tf.gradients(flatten_logits,ops_op,pls['logits_grad'])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    ops={}
    ops['global_feats']=global_feats
    ops['all']=ops_op
    ops['grads']=grads

    return sess,pls,ops



if __name__=="__main__":
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import random

    names=['fc0','lf0','fc1','lf1','fc2','lf2']

    train_list,test_list=get_block_train_test_split()
    random.shuffle(test_list)
    test_list=['data/S3DIS/sampled_test_new/'+fn for fn in test_list]
    sess,pls,ops=build_model()
    saver=tf.train.Saver()
    saver.restore(sess,'model/pgnet_v3_no_covar/model56.ckpt')

    # abs value
    # all_feats=[[] for i in xrange(6)]
    # for fn in test_list[:3]:
    #     xyzs,rgbs,covars,labels,block_mins=read_pkl(fn)
    #     for i in xrange(len(xyzs[:35])):
    #         feats_list=sess.run(ops['all'],feed_dict={
    #             pls['xyzs']:xyzs[i],pls['feats']:rgbs[i],pls['labels']:labels[i]
    #         })
    #         for k,feats in enumerate(feats_list):
    #             all_feats[k].append(feats)
    #
    # for k in xrange(6):
    #     print names[k]
    #     all_feats[k]=np.concatenate(all_feats[k],axis=0)
    #     all_feats[k]=np.abs(all_feats[k])
    #
    #     print np.sum(np.max(all_feats[k],axis=0)<1e-6)
    #
    #     print all_feats[k].shape
    #
    #     plt.figure(0)
    #     plt.plot(np.arange(all_feats[k].shape[1]),np.mean(all_feats[k],axis=0),'-')
    #     plt.savefig('{}_mean.png'.format(names[k]))
    #     plt.close()
    #
    #     plt.figure(0)
    #     plt.plot(np.arange(all_feats[k].shape[1]),np.max(all_feats[k],axis=0),'-')
    #     plt.savefig('{}_max.png'.format(names[k]))
    #     plt.close()

    # grads
    all_feats_grads=[[] for i in xrange(6)]
    class_idx=6
    for fn in test_list[:3]:
        xyzs,rgbs,covars,labels,block_mins=read_pkl(fn)
        for i in xrange(len(xyzs[:35])):
            logits_grad=np.zeros([xyzs[i].shape[0],13])
            logits_grad[:,class_idx]=1.0
            grads_list=sess.run(ops['grads'],feed_dict={
                pls['xyzs']:xyzs[i],pls['feats']:rgbs[i],pls['labels']:labels[i],
                pls['is_training']:False,pls['logits_grad']:logits_grad
            })
            for k,grads in enumerate(grads_list):
                all_feats_grads[k].append(grads)

    for k in xrange(6):
        print names[k]
        all_feats_grads[k]=np.concatenate(all_feats_grads[k],axis=0)
        all_feats_grads[k]=np.abs(all_feats_grads[k])

        print np.sum(np.max(all_feats_grads[k],axis=0)<1e-6)

        print all_feats_grads[k].shape

        plt.figure(0)
        plt.plot(np.arange(all_feats_grads[k].shape[1]),np.mean(all_feats_grads[k],axis=0),'-')
        plt.savefig('{}_mean_grad.png'.format(names[k]))
        plt.close()
        all_feats_grads[k]=np.mean(all_feats_grads[k],axis=0)

    all_feats_grads=np.concatenate(all_feats_grads,axis=0)
    plt.figure(0)
    plt.plot(np.arange(all_feats_grads.shape[0]),all_feats_grads,'-')
    plt.savefig('all_grad.png')
    plt.close()

        # plt.figure(0)
        # plt.plot(np.arange(all_feats_grads[k].shape[1]),np.max(all_feats_grads[k],axis=0),'-')
        # plt.savefig('{}_max_grad.png'.format(names[k]))
        # plt.close()




