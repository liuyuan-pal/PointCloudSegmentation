from io_util import read_pkl,save_pkl,read_fn_hierarchy,get_block_train_test_split
import time
from aug_util import sample_block,normalize_block_hierarchy

bsize=3.0
bstride=1.5
nr1=0.1
nr2=0.4
nr3=1.0
sstride=0.05
vc1=0.2
vc2=0.5
min_pn=512
resample_ratio_low=0.8
resample_ratio_high=1.0
covar_ds_stride=0.03
covar_nn_size=0.1
max_pt_num=10240


def prepare_data(fn,use_rescale,use_flip,use_rotate,use_jitter,use_resample,
                 nr1,nr2,nr3,vc1,vc2,sstride,bsize,bstride,min_pn,
                 resample_ratio_low,resample_ratio_high,covar_ds_stride,covar_nn_size,max_pt_num):

    points, labels = read_pkl(fn)
    xyzs, rgbs, covars, lbls = sample_block(points, labels, sstride, bsize, bstride, min_pn=min_pn,
                                            use_rescale=use_rescale, use_flip=use_flip, use_rotate=use_rotate,
                                            covar_ds_stride=covar_ds_stride, covar_nn_size=covar_nn_size)


    cxyzs, dxyzs, rgbs, covars, lbls, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens, block_mins = \
        normalize_block_hierarchy(xyzs, rgbs, covars, lbls, bsize=bsize, nr1=nr1, nr2=nr2, nr3=nr3, vc1=vc1, vc2=vc2,
                                  resample=use_resample, jitter_color=use_jitter,resample_low=resample_ratio_low,
                                  resample_high=resample_ratio_high,max_pt_num=max_pt_num)

    return cxyzs, dxyzs, rgbs, covars, lbls, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens, block_mins


def prepare_s3dis_train_single_file(fn):
    in_fn='data/S3DIS/room_block_10_10/'+fn
    all_data=[[] for _ in xrange(13)]
    bg=time.time()
    for i in xrange(5):
        data=prepare_data(in_fn,True,True,False,True,True,nr1,nr2,nr3,vc1,vc2,sstride,bsize,
                          bstride,min_pn,resample_ratio_low,resample_ratio_high,covar_ds_stride,covar_nn_size,max_pt_num)
        for t in xrange(13):
            all_data[t]+=data[t]

    out_fn='data/S3DIS/sampled_train/'+fn
    save_pkl(out_fn,all_data)
    print 'done {} cost {} s'.format(fn,time.time()-bg)


def prepare_s3dis_test_single_file(fn):
    in_fn='data/S3DIS/room_block_10_10/'+fn
    data=prepare_data(in_fn,False,False,False,False,False,nr1,nr2,nr3,vc1,vc2,sstride,bsize,
                      bstride,min_pn,resample_ratio_low,resample_ratio_high,covar_ds_stride,covar_nn_size,max_pt_num)

    out_fn='data/S3DIS/sampled_test/'+fn
    save_pkl(out_fn,data)

def prepare_s3dis_train():
    train_list,test_list=get_block_train_test_split()

    from concurrent.futures import ProcessPoolExecutor
    executor=ProcessPoolExecutor(8)
    futures=[]
    for fn in train_list:
        futures.append(executor.submit(prepare_s3dis_train_single_file,fn))

    for future in futures:
        future.result()


def prepare_s3dis_test():
    train_list,test_list=get_block_train_test_split()

    for fn in test_list:
        bg=time.time()
        prepare_s3dis_test_single_file(fn)
        print 'done {} cost {} s'.format(fn,time.time()-bg)

def test_s3dis_preparing():
    import random
    train_list,test_list=get_block_train_test_split()
    random.shuffle(train_list)
    for fn in train_list[:1]:
        data=read_pkl('data/S3DIS/sampled_train/'+fn)
        for i in xrange(13):
            print len(data[i])

def compute_weight():
    from io_util import get_block_train_test_split
    import numpy as np
    train_list,test_list=get_block_train_test_split()

    test_list=['data/S3DIS/sampled_test/'+fs for fs in test_list]
    train_list=['data/S3DIS/sampled_train/'+fs for fs in train_list]
    test_list+=train_list
    labels=[]
    for fs in test_list:
        labels+=read_pkl(fs)[4]
    labels=np.concatenate(labels,axis=0)

    labelweights, _ = np.histogram(labels, range(14))
    print labelweights
    labelweights = labelweights.astype(np.float32)
    labelweights = labelweights / np.sum(labelweights)
    labelweights = 1 / np.log(1.2 + labelweights)

    print labelweights


def sample_subset():
    train_list,test_list=get_block_train_test_split()

    train_list=[fs for fs in train_list if fs.split('_')[3]=='office']
    test_list=[fs for fs in test_list if fs.split('_')[3]=='office']

    train_list=['data/S3DIS/sampled_train/'+fn for fn in train_list]
    test_list=['data/S3DIS/sampled_test/'+fn for fn in test_list]

    total_train=0
    for fs in train_list:
        total_train+=len(read_pkl(fs)[0])

    print total_train

    total_test=0
    for fs in test_list:
        total_test+=len(read_pkl(fs)[0])

    print total_test


def tmp_test():
    import numpy as np
    xyzs, rgbs, covars, lbls = read_pkl('cur_data.pkl')
    for i in xrange(4):
        print np.min(xyzs[i],axis=0)
        print np.max(xyzs[i],axis=0)


if __name__=="__main__":
    # prepare_s3dis_train()
    # prepare_s3dis_test()
    # test_s3dis_preparing()
    # compute_weight()
    tmp_test()