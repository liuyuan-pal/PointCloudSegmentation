import numpy as np
import libPointUtil
import cPickle
import random
from aug_util import sample_block,normalize_block,normalize_block_hierarchy,sample_block_v2
import os
import time


def read_room_pkl(filename):
    with open(filename,'rb') as f:
        points,labels=cPickle.load(f)
    return points,labels


def save_room_pkl(filename,points,labels):
    with open(filename,'wb') as f:
        cPickle.dump([points,labels],f,protocol=2)


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

def get_block_train_test_split_ds(test_area=5):
    '''
    :param test_area: default use area 5 as testset
    :return:
    '''
    path = os.path.split(os.path.realpath(__file__))[0]
    f = open(path + '/cached/room_block_ds0.03_stems.txt', 'r')
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


def get_class_loss_weights():
    return np.asarray([1.0,1.0,1.0,100.0,1.5,1.0,1.0,1.0,1.0,10.0,1.0,2.0,1.0],np.float32)


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


def read_fn_hierarchy(model,filename):
    nr1,nr2,nr3=0.1,0.4,1.0
    vc1,vc2=0.2,0.5
    sstride=0.075
    bsize=3.0
    bstride=1.5
    min_pn=1024
    points,labels=read_room_pkl(filename) # [n,6],[n,1]
    if model=='train':
        xyzs, rgbs, covars, lbls=sample_block_v2(points,labels,sstride,bsize,bstride,min_pn=min_pn,
                                              use_rescale=True,use_flip=True,use_rotate=False)
        cxyzs, dxyzs, rgbs, covars, lbls, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens = \
            normalize_block_hierarchy(xyzs,rgbs,covars,lbls,nr1=nr1,nr2=nr2,nr3=nr3,vc1=vc1,vc2=vc2,
                resample=True,jitter_color=True,resample_low=RESAMPLE_RATIO_LOW,resample_high=RESAMPLE_RATIO_HIGH)
    else:
        xyzs, rgbs, covars, lbls=sample_block_v2(points,labels,sstride,bsize,bsize,min_pn=min_pn/2)
        cxyzs, dxyzs, rgbs, covars, lbls, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens = \
            normalize_block_hierarchy(xyzs,rgbs,covars,lbls,nr1=nr1,nr2=nr2,nr3=nr3,vc1=vc1,vc2=vc2)

    return cxyzs,dxyzs,rgbs,covars,lbls,vlens,vlens_bgs,vcidxs,cidxs,nidxs,nidxs_bgs,nidxs_lens


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


def output_hierarchy(cxyz1,cxyz2,cxyz3,rgbs,lbls,vlens1,vlens2,dxyz1,dxyz2,vc1,vc2):
    from draw_util import get_class_colors,output_points
    output_points('test_result/cxyz1_rgb.txt',cxyz1,rgbs)
    colors=get_class_colors()
    output_points('test_result/cxyz1_lbl.txt',cxyz1,colors[lbls.flatten(),:])

    # test cxyz
    vidxs=[]
    for i,l in enumerate(vlens1):
        vidxs+=[i for _ in xrange(l)]
    colors=np.random.randint(0,256,[vlens1.shape[0],3])
    vidxs=np.asarray(vidxs,np.int32)

    output_points('test_result/cxyz1.txt',cxyz1,colors[vidxs,:])
    output_points('test_result/cxyz2.txt',cxyz2,colors)

    vidxs=[]
    for i,l in enumerate(vlens2):
        vidxs+=[i for _ in xrange(l)]
    colors=np.random.randint(0,256,[vlens2.shape[0],3])
    vidxs=np.asarray(vidxs,np.int32)

    output_points('test_result/cxyz2a.txt',cxyz2,colors[vidxs,:])
    output_points('test_result/cxyz3a.txt',cxyz3,colors)

    # test dxyz
    c=0
    for k,l in enumerate(vlens1):
        for t in xrange(l):
            dxyz1[c+t]*=vc1
            dxyz1[c+t]+=cxyz2[k]
        c+=l
    output_points('test_result/dxyz1.txt',dxyz1)

    c=0
    for k,l in enumerate(vlens2):
        for t in xrange(l):
            dxyz2[c+t]*=vc2
            dxyz2[c+t]+=cxyz3[k]
        c+=l
    output_points('test_result/dxyz2.txt',dxyz2)


def test_data_iter_hierarchy():
    from provider import Provider,default_unpack_feats_labels
    from draw_util import output_points,get_class_colors
    import time
    import random

    train_list,test_list=get_block_train_test_split_ds()
    # random.shuffle(train_list)
    train_list=['data/S3DIS/room_block_10_10_ds0.03/'+fn for fn in train_list]
    test_list=['data/S3DIS/room_block_10_10_ds0.03/'+fn for fn in test_list]
    train_list=train_list[:251]
    test_list=test_list[:len(test_list)/5]

    train_provider = Provider(train_list,'train',4,read_fn_hierarchy)
    test_provider = Provider(test_list,'test',4,read_fn_hierarchy)
    print len(train_list)
    try:
        # begin=time.time()
        # i=0
        # for data in test_provider:
        #     i+=1
        #     pass
        # print 'batch_num {}'.format(i*4)
        # print 'test set cost {} s'.format(time.time()-begin)
        begin=time.time()
        i=0
        for data in train_provider:
            i+=1
            cxyzs, dxyzs, rgbs, covars, lbls, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens = \
                default_unpack_feats_labels(data, 4)
            for k in xrange(4):
                for t in xrange(3):
                    print 'batch {} data {} lvl {} cxyz min {} max {} ptnum {}'.format(i,k,t,np.min(cxyzs[k][t],axis=0),
                                                                                       np.max(cxyzs[k][t],axis=0),
                                                                                       cxyzs[k][t].shape[0])
                    assert cidxs[k][t].shape[0]==nidxs[k][t].shape[0]
                    assert nidxs_bgs[k][t].shape[0]==cxyzs[k][t].shape[0]
                    assert nidxs_lens[k][t].shape[0]==cxyzs[k][t].shape[0]
                    assert np.sum(nidxs_lens[k][t])==nidxs[k][t].shape[0]
                    assert nidxs_bgs[k][t][-1]+nidxs_lens[k][t][-1]==nidxs[k][t].shape[0]
                    assert np.max(cidxs[k][t])==cxyzs[k][t].shape[0]-1
                    print 'lvl {} avg nsize {}'.format(t,cidxs[k][t].shape[0]/float(cxyzs[k][t].shape[0]))

                print 'rgb min {} max {}'.format(np.min(rgbs[k],axis=0),np.max(rgbs[k],axis=0))
                # print 'covars min {} max {}'.format(np.min(covars[k],axis=0),np.max(covars[k],axis=0))
                # print np.min(covars[k],axis=0)
                # print np.max(covars[k],axis=0)

                for t in xrange(2):
                    print 'batch {} data {} lvl {} dxyz min {} max {} ptnum {}'.format(i,k,t,np.min(dxyzs[k][t],axis=0),
                                                                                       np.max(dxyzs[k][t],axis=0),
                                                                                       dxyzs[k][t].shape[0])
                    assert vlens[k][t].shape[0]==cxyzs[k][t+1].shape[0]
                    assert vlens_bgs[k][t].shape[0]==cxyzs[k][t+1].shape[0]
                    assert np.sum(vlens[k][t])==cxyzs[k][t].shape[0]
                    assert vlens_bgs[k][t][-1]+vlens[k][t][-1]==cxyzs[k][t].shape[0]
                    assert np.max(vcidxs[k][t])==cxyzs[k][t+1].shape[0]-1
                print '////////////////////'

            output_hierarchy(cxyzs[0][0],cxyzs[0][1],cxyzs[0][2],rgbs[0]*127+128,lbls[0],vlens[0][0],vlens[0][1],dxyzs[0][0],dxyzs[0][1],0.2,0.5)

            if i>1:
                break

        print 'batch_num {}'.format(i*4)
        print 'train set cost {} s'.format(time.time()-begin)


    finally:
        print 'done'
        train_provider.close()
        test_provider.close()

def test_hierarchy_speed():
    from draw_util import output_points,get_class_colors

    train_list,test_list=get_block_train_test_split_ds()
    random.shuffle(train_list)
    train_list=['data/S3DIS/room_block_10_10_ds0.03/'+fn for fn in train_list]
    filename='data/S3DIS/room_block_10_10_ds0.03/53_Area_2_auditorium_1.pkl'

    for i in xrange(10):
        read_fn_hierarchy('test',filename)


if __name__ =="__main__":
    test_data_iter_hierarchy()