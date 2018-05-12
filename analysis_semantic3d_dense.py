from model_pointnet_semantic3d import *
from model_pooling import *

def build_model():
    pls={}
    pls['xyzs'] = tf.placeholder(tf.float32, [None, 3], 'xyzs')
    pls['feats'] = tf.placeholder(tf.float32, [None, 3], 'feats')
    pls['nidxs'] = tf.placeholder(tf.int32, [None], 'nidxs')
    pls['nlens'] = tf.placeholder(tf.int32, [None], 'nlens')
    pls['nbegs'] = tf.placeholder(tf.int32, [None], 'nbegs')
    pls['nlens'] = tf.placeholder(tf.int32, [None], 'nlens')
    pls['logits_grad'] = tf.placeholder(tf.float32, [None, 13], 'logits_grad')
    pls['labels'] = tf.placeholder(tf.int32, [None], 'labels')
    pls['is_training'] = tf.placeholder(tf.bool, name='is_training')

    xyzs, feats, labels = dense_feats(pls['xyzs'], pls['feats'], pls['labels'], idxs, nidxs, nlens, nbegs, ncens, reuse)
    xyzs, dxyzs, feats, labels, vlens, vbegs, vcens = \
        points_pooling_two_layers(xyzs, feats, labels, 0.45, 1.5, 10.0)
    global_feats, local_feats = pointnet_13_dilate_embed_semantic3d(xyzs, dxyzs, feats, vlens, vbegs, vcens, reuse)

    global_feats = tf.expand_dims(global_feats, axis=0)
    local_feats = tf.expand_dims(local_feats, axis=0)
    logits = classifier_v3(global_feats, local_feats, pls['is_training'], 13, False, use_bn=False)  # [1,pn,num_classes]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    ops['global_feats']=global_feats
    ops['logits']=logits

    return sess,pls