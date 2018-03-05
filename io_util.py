import numpy as np
import libPointUtil
import cPickle
import random
from aug_util import sample_block,normalize_block
import os
import time


def read_room_pkl(filename):
    with open(filename,'rb') as f:
        points,labels=cPickle.load(f)
    return points,labels


BLOCK_SIZE=3.0
BLOCK_STRIDE=1.5
SAMPLE_STRIDE=0.1
RESAMPLE_RATIO_LOW=0.8
RESAMPLE_RATIO_HIGH=1.0
NEIGHBOR_RADIUS=0.2
MIN_POINT_NUM=2048


def get_block_train_test_split(test_area=5):
    '''
    :param test_area: default use area 5 as testset
    :return:
    '''
    path = os.path.split(os.path.realpath(__file__))[0]
    f = open(path + '/cached/room_block_stems.txt', 'r')
    file_stems = [line.strip('\n') for line in f.readlines()]
    f.close()

    train, test = [], []
    for fs in file_stems:
        if fs.split('_')[2] == str(test_area):
            test.append(fs)
        else:
            train.append(fs)

    return train, test


def get_class_names():
    names=[]
    path=os.path.split(os.path.realpath(__file__))[0]
    with open(path+'/cached/class_names.txt','r') as f:
        for line in f.readlines():
            names.append(line.strip('\n'))

    return names


def read_fn(model,filename):
    points,labels=read_room_pkl(filename) # [n,6],[n,1]
    if model=='train':
        xyzs, rgbs, covars, lbls=sample_block(points,labels,SAMPLE_STRIDE,BLOCK_SIZE,BLOCK_STRIDE,min_pn=MIN_POINT_NUM,
                                              use_rescale=True,use_flip=True,use_rotate=False)

        xyzs, rgbs, covars, lbls, nidxs, nidxs_lens, nidxs_bgs, cidxs, block_bgs, block_lens = \
            normalize_block(xyzs,rgbs,covars,lbls,neighbor_radius=NEIGHBOR_RADIUS,resample=True,jitter_color=True,
                            resample_low=RESAMPLE_RATIO_LOW,resample_high=RESAMPLE_RATIO_HIGH)
    else:
        xyzs, rgbs, covars, lbls=sample_block(points,labels,SAMPLE_STRIDE,BLOCK_SIZE,BLOCK_SIZE,min_pn=MIN_POINT_NUM/2)
        xyzs, rgbs, covars, lbls, nidxs, nidxs_lens, nidxs_bgs, cidxs, block_bgs, block_lens = \
            normalize_block(xyzs,rgbs,covars,lbls,neighbor_radius=NEIGHBOR_RADIUS)

    return xyzs, rgbs, covars, lbls, nidxs, nidxs_lens, nidxs_bgs, cidxs, block_bgs, block_lens


def test_data_iter():
    from provider import Provider,default_unpack_feats_labels
    from draw_util import output_points,get_class_colors
    import time
    import random

    train_list,test_list=get_block_train_test_split()
    random.shuffle(train_list)
    train_list=['data/S3DIS/room_block_10_10/'+fn for fn in train_list]
    test_list=['data/S3DIS/room_block_10_10/'+fn for fn in test_list]

    train_provider = Provider(train_list,'train',4,read_fn)
    test_provider = Provider(test_list,'test',4,read_fn)
    print len(train_list)
    try:
        begin=time.time()
        i=0
        for data in test_provider:
            i+=1
            pass
        print 'batch_num {}'.format(i*4)
        print 'test set cost {} s'.format(time.time()-begin)
        begin=time.time()
        i=0
        for data in train_provider:
            i+=1
            pass
        print 'batch_num {}'.format(i*4)
        print 'train set cost {} s'.format(time.time()-begin)


    finally:
        print 'done'
        train_provider.close()
        test_provider.close()


def test_sample():
    from draw_util import output_points,get_class_colors
    from sklearn.cluster import KMeans
    colors=get_class_colors()

    train_list,test_list=get_block_train_test_split()
    random.shuffle(train_list)
    train_list=['data/S3DIS/room_block_10_10/'+fn for fn in train_list]
    filename=train_list[0]
    # filename='data/S3DIS/room_block_10_10/58_Area_2_auditorium_2.pkl'
    points,labels=read_room_pkl(filename) # [n,6],[n,1]
    begin=time.time()
    xyzs, rgbs, covars, lbls=sample_block(points,labels,SAMPLE_STRIDE,BLOCK_SIZE,BLOCK_SIZE,min_pn=MIN_POINT_NUM/2)
                                          #use_rescale=True,use_flip=True,use_rotate=True)
    print 'sample_block cost {} s'.format(time.time()-begin)

    kc=np.random.randint(0,255,[5,3])
    for j in xrange(len(xyzs)):
        # print xyzs[j].shape,lbls[j].shape
        output_points('test_result/label{}.txt'.format(j),xyzs[j],colors[lbls[j].flatten(),:])
        output_points('test_result/lrgbs{}.txt'.format(j),xyzs[j],rgbs[j])

    kmeans=KMeans(5)
    preds=kmeans.fit_predict(np.concatenate(covars,axis=0))
    output_points('test_result/kmeans.txt',np.concatenate(xyzs,axis=0),kc[preds.flatten(),:])

    pt_num=[len(xyz) for xyz in xyzs]
    print 'avg pt num: {}'.format(np.mean(pt_num))


def test_normalize():
    from draw_util import output_points,get_class_colors
    from sklearn.cluster import KMeans
    colors=get_class_colors()

    train_list,test_list=get_block_train_test_split()
    random.shuffle(train_list)
    train_list=['data/S3DIS/room_block_10_10/'+fn for fn in train_list]
    # filename=train_list[0]
    filename='data/S3DIS/room_block_10_10/49_Area_1_office_8.pkl'
    points,labels=read_room_pkl(filename) # [n,6],[n,1]
    begin=time.time()
    xyzs, rgbs, covars, lbls=sample_block(points,labels,SAMPLE_STRIDE,BLOCK_SIZE,BLOCK_STRIDE,min_pn=2048,
                                          use_rescale=True,use_flip=True,use_rotate=True)
    print 'sample_block cost {} s'.format(time.time()-begin)

    # for j in xrange(len(xyzs)):
    #     print np.min(xyzs[j],axis=0),np.max(xyzs[j],axis=0)
    #     print np.min(rgbs[j],axis=0),np.max(rgbs[j],axis=0)
    #     print xyzs[j].shape,lbls[j].shape
        # output_points('test_result/label_init{}.txt'.format(j),xyzs[j],colors[lbls[j].flatten(),:])
        # output_points('test_result/lrgbs_init{}.txt'.format(j),xyzs[j],rgbs[j])

    xyzs, rgbs, covars, lbls, nidxs, nidxs_lens, nidxs_bgs, cidxs=\
        normalize_block(xyzs,rgbs,covars,lbls,0.2,True,0.8,1.0,True,2.5)

    for j in xrange(len(xyzs)):
        print xyzs[j].shape,rgbs[j].shape,covars[j].shape,lbls[j].shape,nidxs[j].shape,nidxs_lens[j].shape,nidxs_bgs[j].shape,cidxs[j].shape
        print np.min(xyzs[j],axis=0),np.max(xyzs[j],axis=0)
        print np.min(rgbs[j],axis=0),np.max(rgbs[j],axis=0)
        print 'avg nn size: {}'.format(len(nidxs[j])/float(len(xyzs[j])))
        # print xyzs[j].shape,lbls[j].shape
        output_points('test_result/label{}.txt'.format(j),xyzs[j],colors[lbls[j].flatten(),:])
        output_points('test_result/lrgbs{}.txt'.format(j),xyzs[j],np.asarray(rgbs[j]*128+127,np.int32))

    for j in xrange(len(xyzs[0])):
        output_points('test_result/nn{}.txt'.format(j),xyzs[0][nidxs[0][nidxs_bgs[0][j]:nidxs_bgs[0][j]+nidxs_lens[0][j]],:])


def test_time():
    from draw_util import output_points,get_class_colors
    from sklearn.cluster import KMeans
    colors=get_class_colors()

    train_list,test_list=get_block_train_test_split()
    random.shuffle(train_list)
    train_list=['data/S3DIS/room_block_10_10/'+fn for fn in train_list]
    # filename=train_list[0]
    filename='data/S3DIS/room_block_10_10/49_Area_1_office_8.pkl'
    points,labels=read_room_pkl(filename) # [n,6],[n,1]
    for i in xrange(10):
        t=time.time()
        xyzs, rgbs, covars, lbls=sample_block(points,labels,0.1,3.0,1.5,min_pn=2048,
                                      use_rescale=True,use_flip=True,use_rotate=True)
        print 'sample block cost {} s '.format(time.time()-t)

    t=time.time()
    xyzs, rgbs, covars, lbls, nidxs, nidxs_lens, nidxs_bgs, cidxs=\
        normalize_block(xyzs,rgbs,covars,lbls,0.2,True,0.8,1.0,True,2.5)
    print 'normalize cost {} s '.format(time.time()-t)



if __name__ =="__main__":
    test_data_iter()