import numpy as np
import libPointUtil
import cPickle
import random
from aug_util import sample_block,normalize_block,normalize_block_hierarchy,sample_block_v2,normalize_model_hierarchy
import os
import time
import h5py

def read_room_h5(room_h5_file):
    f=h5py.File(room_h5_file,'r')
    data,label = f['data'][:],f['label'][:]
    f.close()

    return data, label

def read_model_h5(fn):
    f=h5py.File(fn,'r')
    data,label = f['point'][:],f['label'][:]
    f.close()

    return data, label

def save_pkl(filename,obj):
    with open(filename,'wb') as f:
        cPickle.dump(obj,f,protocol=2)

def read_pkl(filename):
    with open(filename,'rb') as f:
        obj=cPickle.load(f)
    return obj

def read_room_pkl(filename):
    with open(filename,'rb') as f:
        points,labels=cPickle.load(f)
    return points,labels


def save_room_pkl(filename,points,labels):
    with open(filename,'wb') as f:
        cPickle.dump([points,labels],f,protocol=2)


def get_train_test_split(test_area=5):
    '''
    :param test_area: default use area 5 as testset
    :return:
    '''
    path = os.path.split(os.path.realpath(__file__))[0]
    f = open(path + '/cached/room_stems.txt', 'r')
    file_stems = [line.strip('\n') for line in f.readlines()]
    f.close()

    train, test = [], []
    for fs in file_stems:
        if fs.split('_')[2] == str(test_area):
            test.append(fs)
        else:
            train.append(fs)

    return train, test


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

def get_scannet_class_names():
    g_label_names = ['unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 'sink',
                     'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refridgerator',
                     'picture', 'cabinet', 'otherfurniture']
    return g_label_names

def read_fn(model,filename):
    BLOCK_SIZE=3.0
    BLOCK_STRIDE=1.5
    SAMPLE_STRIDE=0.1
    RESAMPLE_RATIO_LOW=0.8
    RESAMPLE_RATIO_HIGH=1.0
    NEIGHBOR_RADIUS=0.2
    MIN_POINT_NUM=2048
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


def read_fn_hierarchy(model, filename,
                      presample=True,use_rotate=False,
                      nr1=0.1,nr2=0.4,nr3=1.0,
                      vc1=0.2,vc2=0.5,
                      sstride=0.075,
                      bsize=3.0,
                      bstride=1.5,
                      min_pn=1024,
                      resample_ratio_low=0.8,
                      resample_ratio_high=1.0,
                      covar_ds_stride=0.03,
                      covar_nn_size=0.1,
                      max_pt_num=10240):

    if filename.endswith('.pkl'):
        points,labels=read_room_pkl(filename) # [n,6],[n,1]
    else:
        points,labels=read_room_h5(filename) # [n,6],[n,1]

    if model=='train':
        if presample:
            xyzs, rgbs, covars, lbls=sample_block_v2(points,labels,sstride,bsize,bstride,min_pn=min_pn,
                                                     use_rescale=True,use_flip=True,use_rotate=use_rotate,
                                                     covar_nn_size=covar_nn_size)
        else:
            xyzs, rgbs, covars, lbls=sample_block(points,labels,sstride,bsize,bstride,min_pn=min_pn,
                                                  use_rescale=True,use_flip=True,use_rotate=use_rotate,
                                                  covar_ds_stride=covar_ds_stride,covar_nn_size=covar_nn_size)

        cxyzs, dxyzs, rgbs, covars, lbls, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens, block_mins = \
            normalize_block_hierarchy(xyzs,rgbs,covars,lbls,bsize=bsize,nr1=nr1,nr2=nr2,nr3=nr3,vc1=vc1,vc2=vc2,
                                      resample=True,jitter_color=True,
                                      resample_low=resample_ratio_low,resample_high=resample_ratio_high,
                                      max_pt_num=max_pt_num)

    else:
        if presample:
            xyzs, rgbs, covars, lbls=sample_block_v2(points,labels,sstride,bsize,bsize,min_pn=min_pn/2,
                                                     covar_nn_size=covar_nn_size)
        else:
            xyzs, rgbs, covars, lbls=sample_block(points,labels,sstride,bsize,bsize,min_pn=min_pn/2,
                                                  covar_ds_stride=covar_ds_stride,covar_nn_size=covar_nn_size)

        cxyzs, dxyzs, rgbs, covars, lbls, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens, block_mins = \
            normalize_block_hierarchy(xyzs,rgbs,covars,lbls,bsize=bsize,nr1=nr1,nr2=nr2,nr3=nr3,vc1=vc1,vc2=vc2,
                                      max_pt_num=max_pt_num)

    return cxyzs,dxyzs,rgbs,covars,lbls,vlens,vlens_bgs,vcidxs,cidxs,nidxs,nidxs_bgs,nidxs_lens,block_mins


def read_model_hierarchy(model, filename):
    points,labels=read_model_h5(filename)

    if model=='train':
        cxyzs, dxyzs, covars, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens= \
            normalize_model_hierarchy(points,True)
    else:
        cxyzs, dxyzs, covars, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens = \
            normalize_model_hierarchy(points,False)

    return labels,cxyzs,dxyzs,covars,vlens,vlens_bgs,vcidxs,cidxs,nidxs,nidxs_bgs,nidxs_lens

def get_semantic3d_class_names():
    return ['unknown','man-made terrain','natural terrain','high vegetation',
            'low vegetation','buildings','hard scape','scanning artefacts','cars']

def get_semantic3d_block_train_test_split(
        test_stems=('sg28_station4_intensity_rgb','untermaederbrunnen_station3_xyz_intensity_rgb')
):
    train_list,test_list=[],[]
    path = os.path.split(os.path.realpath(__file__))[0]
    with open(path+'/cached/semantic3d_merged_train.txt','r') as f:
        for line in f.readlines():
            line=line.strip('\n')
            stem=line.split(' ')[0]
            num=int(line.split(' ')[1])
            # print stem
            if stem in test_stems:
                test_list+=[stem+'_{}.pkl'.format(i) for i in xrange(num)]
            else:
                train_list+=[stem+'_{}.pkl'.format(i) for i in xrange(num)]

    return train_list,test_list


def get_semantic3d_all_block():
    with open('cached/semantic3d_train_pkl.txt','r') as f:
        fs=[line.strip('\n') for line in f.readlines()]
    return fs



def get_semantic3d_testset():
    with open('cached/semantic3d_test_stems.txt','r') as f:
        lines=f.readlines()
        fns=[fn.strip('\n').split(' ')[0] for fn in lines]
        pns=[int(fn.strip('\n').split(' ')[1]) for fn in lines]

    return fns,pns


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
    print np.min(points,axis=0)
    begin=time.time()
    xyzs, rgbs, covars, lbls=sample_block(points,labels,0.075,1.5,1.5,min_pn=2048/2)
                                          #use_rescale=True,use_flip=True,use_rotate=True)
    print 'sample_block cost {} s'.format(time.time()-begin)

    print np.min(np.concatenate(xyzs,axis=0),axis=0)
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


from draw_util import get_class_colors,output_points
s3dis_colors=get_class_colors()
def output_hierarchy(cxyz1,cxyz2,cxyz3,rgbs,lbls,vlens1,vlens2,dxyz1,dxyz2,vc1,vc2,idx=0,colors=s3dis_colors):
    output_points('test_result/cxyz1_rgb_{}.txt'.format(idx),cxyz1,rgbs)
    output_points('test_result/cxyz1_lbl_{}.txt'.format(idx),cxyz1,colors[lbls.flatten(),:])

    # test cxyz
    vidxs=[]
    for i,l in enumerate(vlens1):
        vidxs+=[i for _ in xrange(l)]
    colors=np.random.randint(0,256,[vlens1.shape[0],3])
    vidxs=np.asarray(vidxs,np.int32)

    output_points('test_result/cxyz1_{}.txt'.format(idx),cxyz1,colors[vidxs,:])
    output_points('test_result/cxyz2_{}.txt'.format(idx),cxyz2,colors)

    vidxs=[]
    for i,l in enumerate(vlens2):
        vidxs+=[i for _ in xrange(l)]
    colors=np.random.randint(0,256,[vlens2.shape[0],3])
    vidxs=np.asarray(vidxs,np.int32)

    output_points('test_result/cxyz2a_{}.txt'.format(idx),cxyz2,colors[vidxs,:])
    output_points('test_result/cxyz3a_{}.txt'.format(idx),cxyz3,colors)

    # test dxyz
    c=0
    for k,l in enumerate(vlens1):
        for t in xrange(l):
            dxyz1[c+t]*=vc1
            dxyz1[c+t]+=cxyz2[k]
        c+=l
    output_points('test_result/dxyz1_{}.txt'.format(idx),dxyz1)

    c=0
    for k,l in enumerate(vlens2):
        for t in xrange(l):
            dxyz2[c+t]*=vc2
            dxyz2[c+t]+=cxyz3[k]
        c+=l
    output_points('test_result/dxyz2_{}.txt'.format(idx),dxyz2)


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
            cxyzs, dxyzs, rgbs, covars, lbls, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens, block_mins = \
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

            output_hierarchy(cxyzs[0][0],cxyzs[0][1],cxyzs[0][2],rgbs[0]*127+128,lbls[0],vlens[0][0],vlens[0][1],dxyzs[0][0],
                             dxyzs[0][1],0.2,0.5)

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


def test_semantic_hierarchy():
    from provider import Provider,default_unpack_feats_labels
    from functools import partial
    train_list,test_list=get_semantic3d_block_train_test_split()
    train_list=['data/Semantic3D.Net/block/train/'+fn for fn in train_list]
    test_list=['data/Semantic3D.Net/block/train/'+fn for fn in test_list]

    random.shuffle(train_list)

    tmp_read_fn = partial(read_fn_hierarchy,
                          use_rotate=False,presample=False,
                          nr1=0.1, nr2=0.4, nr3=1.0,
                          vc1=0.2, vc2=0.5,
                          sstride=0.075,
                          bsize=5.0,
                          bstride=2.5,
                          min_pn=1024,
                          resample_ratio_low=0.8,
                          resample_ratio_high=1.0)

    train_provider = Provider(train_list,'train',4,tmp_read_fn)
    test_provider = Provider(test_list,'test',4,tmp_read_fn)
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
            # i+=1
            # cxyzs, dxyzs, rgbs, covars, lbls, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens, block_mins = \
            #     default_unpack_feats_labels(data, 4)
            # for k in xrange(4):
            #     for t in xrange(3):
            #         print 'batch {} data {} lvl {} cxyz min {} max {} ptnum {}'.format(i,k,t,np.min(cxyzs[k][t],axis=0),
            #                                                                            np.max(cxyzs[k][t],axis=0),
            #                                                                            cxyzs[k][t].shape[0])
            #         assert cidxs[k][t].shape[0]==nidxs[k][t].shape[0]
            #         assert nidxs_bgs[k][t].shape[0]==cxyzs[k][t].shape[0]
            #         assert nidxs_lens[k][t].shape[0]==cxyzs[k][t].shape[0]
            #         assert np.sum(nidxs_lens[k][t])==nidxs[k][t].shape[0]
            #         assert nidxs_bgs[k][t][-1]+nidxs_lens[k][t][-1]==nidxs[k][t].shape[0]
            #         assert np.max(cidxs[k][t])==cxyzs[k][t].shape[0]-1
            #         print 'lvl {} avg nsize {}'.format(t,cidxs[k][t].shape[0]/float(cxyzs[k][t].shape[0]))
            #
            #     print 'rgb min {} max {}'.format(np.min(rgbs[k],axis=0),np.max(rgbs[k],axis=0))
            #     # print 'covars min {} max {}'.format(np.min(covars[k],axis=0),np.max(covars[k],axis=0))
            #     # print np.min(covars[k],axis=0)
            #     # print np.max(covars[k],axis=0)
            #
            #     for t in xrange(2):
            #         print 'batch {} data {} lvl {} dxyz min {} max {} ptnum {}'.format(i,k,t,np.min(dxyzs[k][t],axis=0),
            #                                                                            np.max(dxyzs[k][t],axis=0),
            #                                                                            dxyzs[k][t].shape[0])
            #         assert vlens[k][t].shape[0]==cxyzs[k][t+1].shape[0]
            #         assert vlens_bgs[k][t].shape[0]==cxyzs[k][t+1].shape[0]
            #         assert np.sum(vlens[k][t])==cxyzs[k][t].shape[0]
            #         assert vlens_bgs[k][t][-1]+vlens[k][t][-1]==cxyzs[k][t].shape[0]
            #         assert np.max(vcidxs[k][t])==cxyzs[k][t+1].shape[0]-1
            #     print '////////////////////'
            #
            # output_hierarchy(cxyzs[0][0],cxyzs[0][1],cxyzs[0][2],rgbs[0]*127+128,lbls[0],vlens[0][0],vlens[0][1],dxyzs[0][0],dxyzs[0][1],0.2,0.5)
            #
            # if i>1:
            #     break
            pass

        print 'batch_num {}'.format(i*4)
        print 'train set cost {} s'.format(time.time()-begin)


    finally:
        print 'done'
        train_provider.close()
        test_provider.close()


def test_semantic_hierarchy_speed():
    from draw_util import output_points,get_class_colors

    train_list,test_list=get_block_train_test_split_ds()
    random.shuffle(train_list)
    # train_list=['data/S3DIS/room_block_10_10_ds0.03/'+fn for fn in train_list]
    filename='data/Semantic3D.Net/block/train/sg28_station4_intensity_rgb_9_3.pkl'

    for i in xrange(10):
        read_fn_hierarchy('test',filename)

def test_semantic_read_pkl():
    from provider import Provider,default_unpack_feats_labels
    train_list,test_list=get_semantic3d_block_train_test_split()
    train_list=['data/Semantic3D.Net/block/sampled/train_merge/{}.pkl'.format(i) for i in xrange(231)]
    test_list=['data/Semantic3D.Net/block/sampled/test/'+fn for fn in test_list]
    simple_read_fn=lambda model,filename: read_pkl(filename)

    train_provider = Provider(train_list,'train',4,simple_read_fn)
    # test_provider = Provider(test_list,'test',4,simple_read_fn)

    print len(train_list)
    try:
        # begin = time.time()
        # i = 0
        # for data in test_provider:
        #     i += 1
        #     pass
        # print 'batch_num {}'.format(i * 4)
        # print 'test set cost {} s'.format(time.time() - begin)

        begin = time.time()
        i = 0
        for data in train_provider:
            i+=1
            if i%2500==0:
                print 'cost {} s'.format(time.time()-begin)

        print 'batch_num {}'.format(i * 4)
        print 'train set cost {} s'.format(time.time() - begin)


    finally:
        print 'done'
        train_provider.close()
        test_provider.close()


def test_model_hierarchy():
    from provider import Provider,default_unpack_feats_labels
    train_list=['data/ModelNet40/ply_data_train{}.h5'.format(i) for i in xrange(5)]
    test_list=['data/ModelNet40/ply_data_test{}.h5'.format(i) for i in xrange(2)]

    # train_provider = Provider(train_list,'train',4,read_model_hierarchy,max_cache=1)
    test_provider = Provider(test_list[1:],'test',4,read_model_hierarchy,max_cache=1)

    print len(train_list)
    try:
        begin = time.time()
        i = 0
        for data in test_provider:
            print data[0][0]
            i += 1
            print 'cost {}s'.format(time.time()-begin)
            labels, cxyzs, dxyzs, covars, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens = \
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

            output_hierarchy(cxyzs[0][0],cxyzs[0][1],cxyzs[0][2],
                             np.ones([cxyzs[0][0].shape[0],3]),
                             np.ones([cxyzs[0][0].shape[0]],dtype=np.int32),
                             vlens[0][0],vlens[0][1],dxyzs[0][0],dxyzs[0][1],0.2,0.5)

            if i>1:
                break
        print 'batch_num {}'.format(i * 4)
        print 'test set cost {} s'.format(time.time() - begin)
        # begin = time.time()
        # i = 0
        # for data in train_provider:
        #     i+=1
        #     print data[0]
        #     if i%2500==0:
        #         print 'cost {} s'.format(time.time()-begin)
        #
        # print 'batch_num {}'.format(i * 4)
        # print 'train set cost {} s'.format(time.time() - begin)


    finally:
        print 'done'
        # train_provider.close()
        test_provider.close()


def test_model_read():
    from provider import Provider,default_unpack_feats_labels
    train_list=['data/ModelNet40/ply_data_train{}.pkl'.format(i) for i in xrange(5)]
    test_list=['data/ModelNet40/ply_data_test{}.pkl'.format(i) for i in xrange(2)]
    fn=lambda model,filename:read_pkl(filename)
    train_provider = Provider(train_list,'train',4,fn,max_cache=1)
    test_provider = Provider(test_list,'test',4,fn,max_cache=1)

    try:
        begin = time.time()
        i = 0
        for data in train_provider:
            print len(data[0])
            i += 1
            print 'cost {}s'.format(time.time()-begin)
            labels, cxyzs, dxyzs, covars, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens = \
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

            # output_hierarchy(cxyzs[0][0],cxyzs[0][1],cxyzs[0][2],
            #                  np.ones([cxyzs[0][0].shape[0],3]),
            #                  np.ones([cxyzs[0][0].shape[0]],dtype=np.int32),
            #                  vlens[0][0],vlens[0][1],dxyzs[0][0],dxyzs[0][1],0.2,0.5)

        print 'batch_num {}'.format(i * 4)
        print 'test set cost {} s'.format(time.time() - begin)
    finally:
        train_provider.close()
        test_provider.close()

def test_scannet():
    from provider import Provider,default_unpack_feats_labels
    with open('cached/scannet_train_filenames.txt','r') as f:
        train_list=[line.strip('\n') for line in f.readlines()]
    train_list=['data/ScanNet/sampled_train/{}'.format(fn) for fn in train_list]
    test_list=['data/ScanNet/sampled_test/test_{}.pkl'.format(i) for i in xrange(312)]
    read_fn=lambda model,filename: read_pkl(filename)

    train_provider = Provider(train_list,'train',4,read_fn)
    test_provider = Provider(test_list,'test',4,read_fn)

    try:
        begin = time.time()
        i = 0
        class_count=np.zeros(21)
        for data in train_provider:
            i += 1
            cxyzs, dxyzs, covars, lbls, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens, block_mins = \
                default_unpack_feats_labels(data, 4)
            for t in xrange(4):
                cur_count,_=np.histogram(lbls[t],np.arange(22))
                class_count+=cur_count
            if i%500==0:
                print i
            # for k in xrange(4):
            #     for t in xrange(3):
            #         print 'batch {} data {} lvl {} cxyz min {} max {} ptnum {}'.format(i,k,t,np.min(cxyzs[k][t],axis=0),
            #                                                                            np.max(cxyzs[k][t],axis=0),
            #                                                                            cxyzs[k][t].shape[0])
            #         assert cidxs[k][t].shape[0]==nidxs[k][t].shape[0]
            #         assert nidxs_bgs[k][t].shape[0]==cxyzs[k][t].shape[0]
            #         assert nidxs_lens[k][t].shape[0]==cxyzs[k][t].shape[0]
            #         assert np.sum(nidxs_lens[k][t])==nidxs[k][t].shape[0]
            #         assert nidxs_bgs[k][t][-1]+nidxs_lens[k][t][-1]==nidxs[k][t].shape[0]
            #         assert np.max(cidxs[k][t])==cxyzs[k][t].shape[0]-1
            #         print 'lvl {} avg nsize {}'.format(t,cidxs[k][t].shape[0]/float(cxyzs[k][t].shape[0]))
            #
            #     # print 'covars min {} max {}'.format(np.min(covars[k],axis=0),np.max(covars[k],axis=0))
            #     # print np.min(covars[k],axis=0)
            #     # print np.max(covars[k],axis=0)
            #
            #     for t in xrange(2):
            #         print 'batch {} data {} lvl {} dxyz min {} max {} ptnum {}'.format(i,k,t,np.min(dxyzs[k][t],axis=0),
            #                                                                            np.max(dxyzs[k][t],axis=0),
            #                                                                            dxyzs[k][t].shape[0])
            #         assert vlens[k][t].shape[0]==cxyzs[k][t+1].shape[0]
            #         assert vlens_bgs[k][t].shape[0]==cxyzs[k][t+1].shape[0]
            #         assert np.sum(vlens[k][t])==cxyzs[k][t].shape[0]
            #         assert vlens_bgs[k][t][-1]+vlens[k][t][-1]==cxyzs[k][t].shape[0]
            #         assert np.max(vcidxs[k][t])==cxyzs[k][t+1].shape[0]-1
            #     print '////////////////////'
            #
            # colors=np.random.randint(0,256,[21,3])
            # print lbls[0].shape,colors[lbls[0],:].shape,cxyzs[0][0].shape
            # output_hierarchy(cxyzs[0][0],cxyzs[0][1],cxyzs[0][2],
            #                  np.ones([cxyzs[0][0].shape[0],3]),lbls[0],
            #                  vlens[0][0],vlens[0][1],dxyzs[0][0],dxyzs[0][1],0.15,0.5,colors=colors)
            # break

        class_names=get_scannet_class_names()
        for count,name in zip(class_count,class_names):
            print '{}:\t\t{}'.format(name,count)

        print 'batch_num {}'.format(i * 4)
        print 'train set cost {} s'.format(time.time() - begin)
    finally:
        train_provider.close()
        test_provider.close()


def test_read_s3dis_dataset():
    from provider import Provider,default_unpack_feats_labels
    train_list,test_list=get_block_train_test_split()
    train_list=['data/S3DIS/sampled_train/'+fn for fn in train_list]
    test_list=['data/S3DIS/sampled_test/'+fn for fn in test_list]
    train_list+=test_list

    def fn(model,filename):
        data=read_pkl(filename)
        return data[0],data[2],data[3],data[4],data[12]

    train_provider = Provider(train_list,'train',4,fn)
    test_provider = Provider(test_list,'test',4,fn)

    try:
        begin = time.time()
        i = 0
        for data in train_provider:
            i += 1
            cxyzs, rgbs, covars, lbls, block_mins = default_unpack_feats_labels(data, 4)
            for k in xrange(4):
                min_xyz=np.min(cxyzs[k][0],axis=0)
                # print min_xyz
                eps=1e-5
                min_val=np.asarray([-1.5, -1.5, 0.0])-eps
                val=np.asarray(np.floor(min_xyz - min_val),np.int32)
                print val
                assert val[0]>=0
                assert val[1]>=0
                assert val[2]>=0

        print 'batch_num {}'.format(i * 4)
        print 'train set cost {} s'.format(time.time() - begin)

    finally:
        train_provider.close()
        test_provider.close()


def test_read_semantic_dataset():
    from provider import Provider,default_unpack_feats_labels
    train_list,test_list=get_semantic3d_block_train_test_split()
    # print train_list
    # exit(0)
    train_list=['data/Semantic3D.Net/block/sampled/merged/'+fn for fn in train_list]
    test_list=['data/Semantic3D.Net/block/sampled/merged/'+fn for fn in test_list]
    read_fn=lambda model,filename: read_pkl(filename)

    train_provider = Provider(train_list,'train',4,read_fn)
    test_provider = Provider(test_list,'test',4,read_fn)


    try:
        begin = time.time()
        i = 0
        for data in train_provider:
            i += 1
            cxyzs, rgbs, covars, lbls, = default_unpack_feats_labels(data, 4)
            for k in xrange(4):
                print len(cxyzs[k])

        print 'batch_num {}'.format(i * 4)
        print 'train set cost {} s'.format(time.time() - begin)

    finally:
        train_provider.close()
        test_provider.close()


if __name__ =="__main__":
    # data=read_pkl('data/ModelNet40/ply_data_test1.pkl')
    # print len(data)
    # test_read_semantic_dataset()
    cxyzs, rgbs, covars, lbls=read_pkl('cur_data.pkl')
    for i in xrange(4):
        print np.min(lbls[i]),np.max(lbls[i])
        print np.min(cxyzs[i],axis=0),np.max(cxyzs[i],axis=0)
        print np.min(rgbs[i],axis=0),np.max(rgbs[i],axis=0)