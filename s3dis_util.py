from io_util import read_pkl,save_pkl,read_fn_hierarchy,get_block_train_test_split
import time
from aug_util import swap_xy,flip,compute_nidxs_bgs,compute_cidxs,uniform_sample_block
import numpy as np
import libPointUtil
import random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


bsize=3.0
bstride=1.5
sstride=0.05
min_pn=512
covar_ds_stride=0.03
covar_nn_size=0.1
max_pt_num=10240


def fetch_subset(all,idxs,subsets=None):
    if subsets is None:
        subsets=[[] for _ in xrange(len(all))]

    for item,subset in zip(all,subsets):
        for idx in idxs:
            subset.append(item[idx])

    return subsets


def sample_block(points, labels, ds_stride, block_size, block_stride, min_pn,
                 use_rescale=False, swap=False,flip_x=False,flip_y=False,
                 covar_ds_stride=0.03, covar_nn_size=0.1):

    xyzs=np.ascontiguousarray(points[:,:3])
    rgbs=np.ascontiguousarray(points[:,3:])
    min_xyzs=np.min(xyzs,axis=0,keepdims=True)
    max_xyzs=np.max(xyzs,axis=0,keepdims=True)

    covar_ds_idxs=libPointUtil.gridDownsampleGPU(xyzs, covar_ds_stride, False)
    covar_ds_xyzs=np.ascontiguousarray(xyzs[covar_ds_idxs,:])

    # flip
    if swap:
        covar_ds_xyzs=swap_xy(covar_ds_xyzs)
        min_xyzs=swap_xy(min_xyzs)
        max_xyzs=swap_xy(max_xyzs)
    if flip_x:
        covar_ds_xyzs=flip(covar_ds_xyzs,axis=0)
        min_xyzs[:,0],max_xyzs[:,0]=-max_xyzs[:,0],-min_xyzs[:,0]

    if flip_y:
        covar_ds_xyzs=flip(covar_ds_xyzs,axis=1)
        min_xyzs[:,1],max_xyzs[:,1]=-max_xyzs[:,1],-min_xyzs[:,1]

    # rescale
    if use_rescale:
        rescale=np.random.uniform(0.9,1.1,[1,3])
        covar_ds_xyzs[:,:3]*=rescale
        min_xyzs*=rescale
        max_xyzs*=rescale

    ds_idxs=libPointUtil.gridDownsampleGPU(covar_ds_xyzs,ds_stride,False)

    covar_nidxs=libPointUtil.findNeighborRadiusCPU(covar_ds_xyzs,ds_idxs,covar_nn_size)

    covar_nidxs_lens=np.ascontiguousarray([len(idxs) for idxs in covar_nidxs],np.int32)
    covar_nidxs_bgs=compute_nidxs_bgs(covar_nidxs_lens)
    covar_nidxs=np.ascontiguousarray(np.concatenate(covar_nidxs,axis=0),dtype=np.int32)

    covars=libPointUtil.computeCovarsGPU(covar_ds_xyzs,covar_nidxs,covar_nidxs_lens,covar_nidxs_bgs)

    xyzs=covar_ds_xyzs[ds_idxs,:]
    rgbs=rgbs[covar_ds_idxs,:][ds_idxs,:]
    lbls=labels[covar_ds_idxs][ds_idxs]

    xyzs-=min_xyzs
    idxs = uniform_sample_block(xyzs,block_size,block_stride,normalized=True,min_pn=min_pn)
    xyzs+=min_xyzs

    xyzs, rgbs, covars, lbls=fetch_subset([xyzs,rgbs,covars,lbls],idxs)

    return xyzs, rgbs, covars, lbls



def _append(lists, items):
    for l,i in zip(lists,items):
        l.append(i)

def normalize_block(xyzs,rgbs,covars,lbls,bsize,max_pt_num,
                    resample=False,resample_low=0.8,resample_high=1.0,
                    jitter_color=False,jitter_val=2.5):
    bn=len(xyzs)
    block_mins=[]
    for bid in xrange(bn):
        # if resample:
        #     pt_num=len(xyzs[bid])
        #     random_down_ratio=np.random.uniform(resample_low,resample_high)
        #     idxs=np.random.choice(pt_num,int(pt_num*random_down_ratio))
        #     xyzs[bid]=xyzs[bid][idxs,:]
        #     rgbs[bid]=rgbs[bid][idxs,:]
        #     lbls[bid]=lbls[bid][idxs,:]
        #     covars[bid]=covars[bid][idxs,:]

        # if len(xyzs[bid])>max_pt_num:
        #     pt_num=len(xyzs[bid])
        #     ratio=max_pt_num/float(len(xyzs[bid]))
        #     idxs=np.random.choice(pt_num,int(pt_num*ratio))
        #     xyzs[bid]=xyzs[bid][idxs,:]
        #     rgbs[bid]=rgbs[bid][idxs,:]
        #     lbls[bid]=lbls[bid][idxs]
        #     covars[bid]=covars[bid][idxs,:]

        # offset center to zero
        # !!! dont rescale here since it will affect the neighborhood size !!!
        min_xyz=np.min(xyzs[bid],axis=0,keepdims=True)
        min_xyz[:,:2]+=bsize/2.0
        xyzs[bid]-=min_xyz
        block_mins.append(min_xyz)

        if jitter_color:
            rgbs[bid]+=np.random.uniform(-jitter_val,jitter_val,rgbs[bid].shape)

            rgbs[bid]-=128
            rgbs[bid]/=(128+jitter_val)
        else:
            rgbs[bid]-=128
            rgbs[bid]/=(128+jitter_val)

        mask=lbls[bid]>12
        if np.sum(mask)>0:
            lbls[bid][mask]=12
        lbls[bid]=lbls[bid].flatten()

    return xyzs,rgbs,covars,lbls,block_mins


def prepare_data(fn,use_rescale,use_swap,use_flip_x,use_flip_y,resample,jitter_color,cur_min_pn=min_pn):
    points, labels = read_pkl(fn)
    xyzs, rgbs, covars, lbls = sample_block(points, labels, sstride, bsize, bstride, min_pn=cur_min_pn,
                                            use_rescale=use_rescale, swap=use_swap, flip_x=use_flip_x, flip_y=use_flip_y,
                                            covar_ds_stride=covar_ds_stride, covar_nn_size=covar_nn_size)

    xyzs, rgbs, covars, lbls, block_mins = normalize_block(xyzs,rgbs,covars,lbls,bsize,max_pt_num,
                                                           resample=resample,resample_low=0.8,resample_high=1.0,
                                                           jitter_color=jitter_color,jitter_val=2.5)

    return xyzs, rgbs, covars, lbls, block_mins


def prepare_s3dis_train_single_file(fn):
    in_fn='data/S3DIS/room_block_10_10/'+fn
    all_data=[[] for _ in xrange(5)]
    bg=time.time()

    data = prepare_data(in_fn,True,True,False,False,True,True)
    for t in xrange(5):
        all_data[t]+=data[t]
    data = prepare_data(in_fn,True,True,True,False,True,True)
    for t in xrange(5):
        all_data[t]+=data[t]
    data = prepare_data(in_fn,True,True,False,True,True,True)
    for t in xrange(5):
        all_data[t]+=data[t]
    data = prepare_data(in_fn,True,True,True,True,True,True)
    for t in xrange(5):
        all_data[t]+=data[t]

    data = prepare_data(in_fn,True,False,False,False,True,True)
    for t in xrange(5):
        all_data[t]+=data[t]
    data = prepare_data(in_fn,True,False,True,False,True,True)
    for t in xrange(5):
        all_data[t]+=data[t]
    data = prepare_data(in_fn,True,False,False,True,True,True)
    for t in xrange(5):
        all_data[t]+=data[t]
    data = prepare_data(in_fn,True,False,True,True,True,True)
    for t in xrange(5):
        all_data[t]+=data[t]

    out_fn='data/S3DIS/sampled_train_nolimits/'+fn
    save_pkl(out_fn,all_data)
    print 'done {} cost {} s'.format(fn,time.time()-bg)


def prepare_s3dis_train_single_file_no_aug(fn):
    in_fn='data/S3DIS/room_block_10_10/'+fn
    bg=time.time()
    data = prepare_data(in_fn,False,False,False,False,False,False,256)

    out_fn='data/S3DIS/sampled_no_aug/'+fn
    save_pkl(out_fn,data)
    print 'done {} cost {} s'.format(fn,time.time()-bg)


def prepare_s3dis_test_single_file(fn):
    in_fn='data/S3DIS/room_block_10_10/'+fn
    data = prepare_data(in_fn,False,False,False,False,False,False,256)

    out_fn='data/S3DIS/sampled_test_nolimits/'+fn
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


def prepare_s3dis_train_no_aug():
    train_list,test_list=get_block_train_test_split()
    train_list+=test_list
    from concurrent.futures import ProcessPoolExecutor
    executor=ProcessPoolExecutor(8)
    futures=[]
    for fn in train_list:
        futures.append(executor.submit(prepare_s3dis_train_single_file_no_aug,fn))

    for future in futures:
        future.result()


def prepare_s3dis_test():
    train_list,test_list=get_block_train_test_split()

    for fn in test_list:
        bg=time.time()
        prepare_s3dis_test_single_file(fn)
        print 'done {} cost {} s'.format(fn,time.time()-bg)


def compute_weight():
    from io_util import get_block_train_test_split,get_class_names
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
    plt.figure(0,figsize=(10, 8), dpi=80)
    plt.bar(np.arange(len(labelweights)),labelweights,tick_label=get_class_names())
    plt.savefig('s3dis_dist.png')
    plt.close()

    print labelweights
    labelweights = labelweights.astype(np.float32)
    labelweights = labelweights / np.sum(labelweights)
    labelweights = 1 / np.log(1.2 + labelweights)

    print labelweights


def get_area(fn):
    return int(fn.split('_')[2])

def merge_train_by_area():
    from io_util import get_block_train_test_split
    train_list,test_list=get_block_train_test_split()
    random.shuffle(train_list)
    f=open('cached/s3dis_merged_train.txt','w')
    for ai in xrange(1,7):
        cur_data=[[] for _ in xrange(5)]
        cur_idx=0
        for fn in train_list:
            an=get_area(fn)
            if an!=ai: continue
            data=read_pkl('data/S3DIS/sampled_train_new/'+fn)
            for i in xrange(5):
                cur_data[i]+=data[i]

            if len(cur_data[0])>1000:
                save_pkl('data/S3DIS/merged_train_new/{}_{}.pkl'.format(ai,cur_idx),cur_data)
                f.write('data/S3DIS/merged_train_new/{}_{}.pkl\n'.format(ai,cur_idx))
                cur_idx+=1
                cur_data=[[] for _ in xrange(5)]

        if len(cur_data[0])>0:
            save_pkl('data/S3DIS/merged_train_new/{}_{}.pkl'.format(ai, cur_idx), cur_data)
            f.write('data/S3DIS/merged_train_new/{}_{}.pkl\n'.format(ai, cur_idx))
            cur_idx += 1

        print 'area {} done'.format(ai)

    f.close()

def test_block_train():
    train_list,test_list=get_block_train_test_split()

    from draw_util import get_class_colors,output_points
    # colors=get_class_colors()
    # for fn in train_list[:1]:
    #     xyzs, rgbs, covars, lbls, block_mins=read_pkl(fn)
    #
    #     for i in xrange(len(xyzs[:5])):
    #         rgbs[i]+=128
    #         rgbs[i]*=127
    #         output_points('test_result/{}clr.txt'.format(i),xyzs[i],rgbs[i])
    #         output_points('test_result/{}lbl.txt'.format(i),xyzs[i],colors[lbls[i]])
    # count=0
    # pt_nums=[]
    #
    # stem2num={}
    # for fn in train_list:
    #     xyzs, rgbs, covars, lbls, block_mins=read_pkl('data/S3DIS/sampled_train_nolimits/'+fn)
    #     stem='_'.join(fn.split('_')[1:])
    #     if stem in stem2num:
    #         stem2num[stem]+=len(xyzs)
    #     else:
    #         stem2num[stem]=len(xyzs)
    #
    #     print stem,stem2num[stem]
    #     count+=len(xyzs)
    #     pt_nums+=[len(pts) for pts in xyzs]
    #
    # print count
    # print np.max(pt_nums)
    # print np.histogram(pt_nums)


    xyzs, rgbs, covars, lbls, block_mins = read_pkl('data/S3DIS/sampled_train_nolimits/{}'.format('1_Area_1_conferenceRoom_2.pkl'))
    for i in xrange(len(xyzs)):
        output_points('test_result/{}.txt'.format(i),xyzs[i]+block_mins[i],rgbs[i]*127+128)


def compare():
    from draw_util import output_points
    from sklearn.cluster import KMeans
    train_list,test_list=get_block_train_test_split()
    random.shuffle(train_list)

    train_list_add=['data/S3DIS/sampled_train/'+fn for fn in train_list]
    for fi,fs in enumerate(train_list_add[:3]):
        cxyzs, dxyzs, rgbs, covars, lbls, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens, block_mins=read_pkl(fs)

        for i in xrange(len(cxyzs[:10])):
            print np.sum(np.sum(np.abs(covars[i]),axis=1)<1e-3)
            kmeans=KMeans(5)
            colors=np.random.randint(0,256,[5,3])
            preds=kmeans.fit_predict(covars[i])
            output_points('test_result/{}_{}.txt'.format(fi,i),cxyzs[i][0],colors[preds])


    print '//////////////////////////'

    # train_list_add=['data/S3DIS/sampled_train_new/'+fn for fn in train_list]
    # for fs in train_list_add[:3]:
    #     xyzs, rgbs, covars, lbls, block_mins=read_pkl(fs)
    #
    #     for i in xrange(len(cxyzs[:10])):
    #         print np.min(xyzs[i],axis=0)
    #         print np.max(xyzs[i],axis=0)
    #         print np.min(rgbs[i],axis=0)
    #         print np.max(rgbs[i],axis=0)

    # cxyzs, dxyzs, rgbs, covars, lbls, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens, block_mins = \
    #     read_pkl('data/S3DIS/sampled_test/'+test_list[4])
    # for i in xrange(len(cxyzs)):
    #     cxyzs[i][0]+=block_mins[i]
    #     rgbs[i]*=127
    #     rgbs[i]+=128
    #
    #     output_points('test_result/{}.txt'.format(i),cxyzs[i][0],rgbs[i])
    #
    # xyzs, rgbs, covars, lbls, block_mins = read_pkl('data/S3DIS/sampled_test_new/'+test_list[4])
    # for i in xrange(len(xyzs)):
    #     xyzs[i]+=block_mins[i]
    #     rgbs[i]*=127
    #     rgbs[i]+=128
    #
    #     output_points('test_result/{}new.txt'.format(i),xyzs[i],rgbs[i])


def test_covar():
    train_list,test_list=get_block_train_test_split()
    points,labels=read_pkl('data/S3DIS/room_block_10_10/'+train_list[0])
    xyzs, rgbs, covars, lbls = sample_block(points, labels, sstride, bsize, bstride, min_pn=512,
                                            use_rescale=False, swap=False, flip_x=False, flip_y=False,
                                            covar_ds_stride=0.075, covar_nn_size=0.15)

    from sklearn.cluster import KMeans
    from draw_util import output_points
    for i in xrange(len(xyzs[:5])):
        kmeans=KMeans(5)
        colors=np.random.randint(0,256,[5,3])
        preds=kmeans.fit_predict(covars[i])
        output_points('test_result/{}.txt'.format(i),xyzs[i],colors[preds])


def prepare_subset_single_file(fn, sstride, bsize, bstride, min_pn, use_scale,
                               use_swap, use_flip_x, use_flip_y, resample, jitter_color):
    points, labels = read_pkl(fn)
    xyzs, rgbs, covars, lbls = sample_block(points, labels, sstride, bsize, bstride, min_pn=min_pn,
                                            use_rescale=use_scale, swap=use_swap, flip_x=use_flip_x, flip_y=use_flip_y,
                                            covar_ds_stride=covar_ds_stride, covar_nn_size=covar_nn_size)

    xyzs, rgbs, covars, lbls, block_mins = normalize_block(xyzs,rgbs,covars,lbls,bsize,max_pt_num,
                                                           resample=resample,resample_low=0.8,resample_high=1.0,
                                                           jitter_color=jitter_color,jitter_val=2.5)

    return xyzs, rgbs, covars, lbls, block_mins

def prepare_subset():
    train_list,test_list=get_block_train_test_split()
    train_list+=test_list
    file_list=[fn for fn in train_list if fn.split('_')[-2]=='office']


    for fn in file_list:
        bg=time.time()
        path='data/S3DIS/room_block_10_10/'+fn
        flip_x=random.random()<0.5
        flip_y=random.random()<0.5
        swap=random.random()<0.5
        all_data=[[] for _ in xrange(5)]
        for i in xrange(1):
            data=prepare_subset_single_file(path,0.075,1.5,0.75,128,True,swap,flip_x,flip_y,True,True)

            for k in xrange(5):
                all_data[k]+=data[k]

        save_pkl('data/S3DIS/office_block/'+fn,all_data)
        print 'done {} cost {} s pn {}'.format(fn,time.time()-bg,np.mean([len(xyzs) for xyzs in all_data[0]]))

def visual_room():
    train_list,test_list=get_block_train_test_split()
    train_list+=test_list
    file_list=[fn for fn in train_list if fn.split('_')[-2]=='office']
    from draw_util import get_class_colors,output_points
    colors=get_class_colors()
    for fn in file_list:
        xyzs,rgbs,covars,labels,block_mins=read_pkl('data/S3DIS/office_block/'+fn)
        for k in xrange(len(xyzs)):
            xyzs[k]+=block_mins[k]
        xyzs=np.concatenate(xyzs,axis=0)
        labels=np.concatenate(labels,axis=0)

        output_points('test_result/{}.txt'.format(fn),xyzs,colors[labels])



if __name__=="__main__":
    test_block_train()
