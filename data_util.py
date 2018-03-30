import libPointUtil
from aug_util import *
from io_util import get_block_train_test_split,read_room_pkl,save_room_pkl,get_block_train_test_split_ds,get_semantic3d_testset
import numpy as np
from draw_util import output_points
from aug_util import get_list_without_back_sample
from concurrent.futures import ProcessPoolExecutor,wait


def downsample(points,labels,stride):
    idxs=libPointUtil.gridDownsampleGPU(points, stride, False)
    points=points[idxs,:]
    labels=labels[idxs,:]

    return points,labels


def downsample_and_save():
    train_list,test_list=get_block_train_test_split()
    idx=0
    train_list+=test_list
    with open('cached/room_block_ds0.03_stems.txt','w') as f:
        for _ in xrange(5):
            for fs in train_list:
                points,labels=read_room_pkl('data/S3DIS/room_block_10_10/'+fs)
                points, labels=downsample(points,labels,0.03)

                names=fs.split('_')
                names[0]=str(idx)
                nfs='_'.join(names)
                f.write(nfs+'\n')
                ofs='data/S3DIS/room_block_10_10_ds0.03/'+nfs
                save_room_pkl(ofs,points,labels)
                idx+=1


def get_class_num():
    train_list,test_list=get_block_train_test_split_ds()

    counts=np.zeros(13)
    for fs in test_list:
        points, labels = read_room_pkl('data/S3DIS/room_block_10_10_ds0.03/'+fs)
        for i in xrange(13):
            counts[i]+=np.sum(labels==i)

    for i in xrange(13):
        print counts[i]/counts[0]


def read_semantic3d_points_file(fn):
    f=open(fn,'r')
    count=0
    while True:
        line=f.readline()
        if not line:
            break
        count+=1

    f.seek(0)
    points=np.empty([count,7],dtype=np.float32)
    pi=0
    while True:
        line=f.readline()
        if not line:
            break
        line=line.strip('\n')
        sublines=line.split(' ')
        x=float(sublines[0])
        y=float(sublines[1])
        z=float(sublines[2])
        intensity=float(sublines[3])
        r=float(sublines[4])
        g=float(sublines[5])
        b=float(sublines[6])
        points[pi,:]=np.asarray([x,y,z,r,g,b,intensity],dtype=np.float32)
        pi+=1
    f.close()

    return np.asarray(points,np.float32)


def read_semantic3d_label_file(fn):
    f=open(fn,'r')
    labels=[]
    while True:
        line=f.readline()
        if not line:
            break
        line=line.strip('\n')
        labels.append(int(line))

    return np.asarray(labels,np.int32)


def prepare_semantic3d():
    with open('cached/semantic3d_stems.txt','r') as f:
        fns=f.readlines()
        fns=[fn.strip('\n') for fn in fns]

    for fn in fns:
        labels=read_semantic3d_label_file('/home/pal/data/Semantic3D.Net/raw/train/'+fn+'.labels')
        points=read_semantic3d_points_file('/home/pal/data/Semantic3D.Net/raw/train/'+fn+'.txt')
        save_room_pkl('/home/pal/data/Semantic3D.Net/pkl/train/'+fn+'.pkl',points,labels)
        print 'done'


def prepare_semantic3d_partition():
    with open('cached/semantic3d_stems.txt','r') as f:
        fns=f.readlines()
        fns=[fn.strip('\n').split(' ')[0] for fn in fns]

    for fn in fns[:7]:
        lf=open('data/Semantic3D.Net/raw/train/' + fn + '.labels')
        pf=open('data/Semantic3D.Net/raw/train/' + fn + '.txt')

        count=0
        while True:
            line=lf.readline()
            if not line:
                break
            count+=1
        lf.seek(0)

        points,labels=[],[]
        pi,part_id=0,0
        while True:
            line=pf.readline()
            if not line:
                break
            line=line.strip('\n')
            sublines=line.split(' ')
            x=float(sublines[0])
            y=float(sublines[1])
            z=float(sublines[2])
            intensity=float(sublines[3])
            r=float(sublines[4])
            g=float(sublines[5])
            b=float(sublines[6])
            points.append(np.asarray([x,y,z,r,g,b,intensity],dtype=np.float32))
            labels.append(int(lf.readline()))
            pi+=1
            if pi>=10000000:
                print 'output {} part {}'.format(fn,part_id)
                pi=0
                save_room_pkl('data/Semantic3D.Net/pkl/train2/'+fn+'_{}.pkl'.format(part_id),
                              np.asarray(points,np.float32),np.asarray(labels,np.int32))
                points,labels=[],[]
                part_id+=1

        lf.close()
        pf.close()

        if pi!=0:
            save_room_pkl('data/Semantic3D.Net/pkl/train2/'+fn+'_{}.pkl'.format(part_id),
                          np.asarray(points,np.float32),np.asarray(labels,np.int32))

        print '{} done'.format(fn)


def read_semantic3d_pkl_stems():
    with open('cached/semantic3d_stems.txt','r') as f:
        lines=f.readlines()
        fss,fns=[],[]
        for line in lines:
            line=line.strip('\n')
            fs,fn=line.split(' ')
            fss.append(fs)
            fns.append(int(fn))

    return fss,fns


def semantic3d_sample_block(beg, bi, ri, rm, fs, fn, min_p, bsize, ds_stride):
    # from io_util import get_semantic3d_class_colors
    # colors=get_semantic3d_class_colors()
    lbls, pts = [], []
    pn=0
    for i in xrange(fn):
        points, labels = read_room_pkl('data/Semantic3D.Net/pkl/train/' + fs + '_{}.pkl'.format(i))
        idxs = libPointUtil.gridDownsampleGPU(points, ds_stride, False)

        points = points[idxs]
        labels = labels[idxs]
        points[:, :3] = np.dot(points[:, :3], rm)
        points[:, :3] -= np.expand_dims(min_p, axis=0)

        x_cond = (points[:, 0] >= beg[0]) & (points[:, 0] < beg[0] + bsize)
        y_cond = (points[:, 1] >= beg[1]) & (points[:, 1] < beg[1] + bsize)
        cond = x_cond & y_cond
        pn+=np.sum(cond)
        pts.append(points[cond])
        lbls.append(labels[cond])

    if pn>1024:
        pts = np.concatenate(pts, axis=0)
        lbls = np.concatenate(lbls, axis=0)

        print 'block {} pn {}'.format(bi,pn)
        save_room_pkl('data/Semantic3D.Net/block/train/' + fs + '_{}_{}.pkl'.format(bi,ri), pts, lbls)
        # output_points('test_result/{}_{}.txt'.format(bi,ri), pts, colors[lbls,:])


def semantic3d_to_block(bsize=20.0,bstride=10.0,ds_stride=0.03):
    executor=ProcessPoolExecutor(6)

    fss,fns=read_semantic3d_pkl_stems()
    for fs,fn in zip(fss[7:],fns[7:]):
        for ri in xrange(6):
            rot_ang=np.pi/2.0*ri
            cosval = np.cos(rot_ang)
            sinval = np.sin(rot_ang)
            rot_m = np.array([[cosval,  sinval, 0],
                              [-sinval, cosval, 0],
                              [0, 0, 1]], dtype=np.float32)
            min_p,max_p=[],[]
            for i in xrange(fn):
                points,labels=read_room_pkl('data/Semantic3D.Net/pkl/train/'+fs+'_{}.pkl'.format(i))
                points[:, :3]=np.dot(points[:,:3],rot_m)
                min_p.append(np.min(points,axis=0)[:3])
                max_p.append(np.max(points,axis=0)[:3])

            min_p=np.min(np.asarray(min_p),axis=0)
            max_p=np.max(np.asarray(max_p),axis=0)
            print min_p,max_p

            x_list=get_list_without_back_sample(max_p[0]-min_p[0],bsize,bstride)
            y_list=get_list_without_back_sample(max_p[1]-min_p[1],bsize,bstride)

            beg_list=[]
            for x in x_list:
                for y in y_list:
                    beg_list.append((x,y))

            print 'fs {} block num {}'.format(fs,len(beg_list))

            futures=[]
            for bi,beg in enumerate(beg_list):
                futures.append(executor.submit(semantic3d_sample_block,beg,bi,ri,rot_m,fs,fn,min_p,bsize,ds_stride))
                # semantic3d_sample_block(beg,bi,fs,fn,min_p,bsize,ds_stride)

            for f in futures:
                f.result()

        print 'fs {} done'.format(fs)


def test_labels():
    import os
    fss,fns=read_semantic3d_pkl_stems()
    from io_util import get_semantic3d_class_colors
    colors=get_semantic3d_class_colors()
    for fn in os.listdir('data/Semantic3D.Net/block/train'):
        if fn.startswith(fss[6]) and fn.endswith('_0.pkl'): # or fn.endswith('_3.pkl')):
            points,labels=read_room_pkl('data/Semantic3D.Net/block/train/'+fn)
            idxs = libPointUtil.gridDownsampleGPU(points, 0.1, False)
            output_points('test_result/'+fn[:-4]+'.txt',points[idxs],colors[labels[idxs],:])


def get_intensity_distribution():
    fss,fns=read_semantic3d_pkl_stems()
    intensities=[]
    for fs,fn in zip(fss,fns):
        for i in xrange(fn):
            points,labels=read_room_pkl('data/Semantic3D.Net/pkl/train/'+fs+'_{}.pkl'.format(i))
            idxs=libPointUtil.gridDownsampleGPU(points,0.1,False)
            intensities.append(points[idxs,-1])

    intensities=np.concatenate(intensities,axis=0)
    print np.min(intensities),np.max(intensities)
    print np.mean(intensities),np.std(intensities)

    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.hist(intensities,100)
    plt.savefig('test_result/intensities.png')
    plt.close()

    # min -2039.0 max 2047.0
    # mean -1164.05 std 599.572


def write_train_file_list():
    with open('cached/semantic3d_train_pkl.txt','w') as f:
        import os
        for fs in os.listdir('data/Semantic3D.Net/block/train/'):
            f.write(fs+'\n')


def test_block():
    import os
    fss,fns=read_semantic3d_pkl_stems()
    from draw_util import get_semantic3d_class_colors
    colors=get_semantic3d_class_colors()
    for fs in fss:
        all_points,all_labels=[],[]
        for fn in os.listdir('data/Semantic3D.Net/block/train'):
            if fn.startswith(fs) and fn.endswith('_0.pkl'): # or fn.endswith('_3.pkl')):
                points,labels=read_room_pkl('data/Semantic3D.Net/block/train/'+fn)
                idxs = libPointUtil.gridDownsampleGPU(points, 0.1, False)
                all_points.append(points[idxs])
                all_labels.append(labels[idxs])

        all_points = np.concatenate(all_points, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        output_points('test_result/'+fs+'_labels.txt',all_points,colors[all_labels,:])
        output_points('test_result/'+fs+'_colors.txt',all_points)


nr1 = 0.125
nr2 = 0.5
nr3 = 2.0
vc1 = 0.25
vc2 = 1.0
sstride = 0.125
bsize = 10.0
bstride = 5.0
min_pn = 128
resample_ratio_low = 0.8
resample_ratio_high = 1.0
covar_ds_stride = 0.05
covar_nn_size = 0.1
max_pt_num = 10240
from io_util import read_fn_hierarchy,get_semantic3d_block_train_test_split,save_pkl

def semantic3d_process_block(filename,
                             use_rescale=False,
                             use_rotate=False,
                             use_flip=False,
                             use_resample=False,
                             jitter_color=False,
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

    points,labels=read_room_pkl(filename) # [n,6],[n,1]
    xyzs, rgbs, covars, lbls=sample_block(points,labels,sstride,bsize,bstride,min_pn=min_pn,
                                          use_rescale=use_rescale,use_flip=use_flip,use_rotate=use_rotate,
                                          covar_ds_stride=covar_ds_stride,covar_nn_size=covar_nn_size,gpu_gather=True)

    cxyzs, dxyzs, rgbs, covars, lbls, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens, block_mins = \
        normalize_block_hierarchy(xyzs,rgbs,covars,lbls,bsize=bsize,nr1=nr1,nr2=nr2,nr3=nr3,vc1=vc1,vc2=vc2,
                                  resample=use_resample,jitter_color=jitter_color,
                                  resample_low=resample_ratio_low,resample_high=resample_ratio_high,
                                  max_pt_num=max_pt_num)

    return cxyzs,dxyzs,rgbs,covars,lbls,vlens,vlens_bgs,vcidxs,cidxs,nidxs,nidxs_bgs,nidxs_lens,block_mins


def semantic3d_sample_single_file_training_block(tfs):
    fs='data/Semantic3D.Net/block/train/'+tfs
    out_data=[[] for _ in xrange(13)]
    for i in xrange(5):
        data=semantic3d_process_block(fs,True,False,True,True,True,
                                      nr1=nr1, nr2=nr2, nr3=nr3,
                                      vc1=vc1, vc2=vc2,
                                      sstride=sstride,
                                      bsize=bsize,
                                      bstride=bstride,
                                      min_pn=min_pn,
                                      resample_ratio_low=resample_ratio_low,
                                      resample_ratio_high=resample_ratio_high,
                                      covar_ds_stride=covar_ds_stride,
                                      covar_nn_size=covar_nn_size,
                                      max_pt_num=max_pt_num)

        for t in xrange(13):
            out_data[t]+=data[t]

        save_pkl('data/Semantic3D.Net/block/sampled/train/'+tfs,out_data)
    print '{} done'.format(tfs)



def semantic3d_sample_training_block():
    executor=ProcessPoolExecutor(8)
    train_list,test_list=get_semantic3d_block_train_test_split()
    futures=[executor.submit(semantic3d_sample_single_file_training_block,tfs) for tfs in test_list]
    for f in futures: f.result()

    # print 'testing set generating ...'
    # for tfs in test_list:
    #     fs='data/Semantic3D.Net/block/train/'+tfs
    #
    #     data=semantic3d_process_block(fs,False,False,False,False,False,
    #                                   nr1=nr1, nr2=nr2, nr3=nr3,
    #                                   vc1=vc1, vc2=vc2,
    #                                   sstride=sstride,
    #                                   bsize=bsize,
    #                                   bstride=bstride,
    #                                   min_pn=min_pn,
    #                                   resample_ratio_low=resample_ratio_low,
    #                                   resample_ratio_high=resample_ratio_high,
    #                                   covar_ds_stride=covar_ds_stride,
    #                                   covar_nn_size=covar_nn_size,
    #                                   max_pt_num=max_pt_num)
    #
    #
    #     save_pkl('data/Semantic3D.Net/block/sampled/test/'+tfs,data)
    #     print '{} done'.format(tfs)


from io_util import read_pkl
def merge_train_files():
    train_list,test_list=get_semantic3d_block_train_test_split()
    all_data=[[] for _ in xrange(13)]
    idx,size=0,0
    random.shuffle(train_list)
    for tfs in train_list:
        fs='data/Semantic3D.Net/block/sampled/train/'+tfs
        data=read_pkl(fs)
        for i in xrange(13):
            all_data[i]+=data[i]
        size+=len(data[0])

        if size>=1000:
            save_pkl('data/Semantic3D.Net/block/sampled/train_merge/{}.pkl'.format(idx),all_data)
            all_data=[[] for _ in xrange(13)]
            size=0
            idx+=1
            print 'output idx {} size {}'.format(idx,size)

    if size>0:
        save_pkl('data/Semantic3D.Net/block/sampled/train_merge/{}.pkl'.format(idx),all_data)

def test_single_sample():
    data=read_fn_hierarchy(
        'test','data/Semantic3D.Net/block/train/sg27_station2_intensity_rgb_23_3.pkl',
        False,False,
        nr1=0.125, nr2=0.5, nr3=2.0,
        vc1=0.25, vc2=1.0,
        sstride=0.125,
        bsize=10.0,
        bstride=5.0,
        min_pn=256,
        resample_ratio_low=0.8,
        resample_ratio_high=1.0,
        covar_ds_stride=0.05,
        covar_nn_size=0.1,
        max_pt_num=10240
    )

    max_nn_size,max_pt_num=0,0
    for i,cxyzs in enumerate(data[0]):
        print cxyzs[0].shape,data[9][i][0].shape
        max_nn_size=max(max_nn_size,data[9][i][0].shape[0])
        max_pt_num=max(max_pt_num,cxyzs[0].shape[0])
        output_points('test_result/{}.txt'.format(i),cxyzs[0],127*data[2][i]+128)

    print 'max {} {}'.format(max_pt_num,max_nn_size)


def prepare_semantic3d_test_partition():
    with open('cached/semantic3d_test_stems.txt','r') as f:
        fns=f.readlines()
        fns=[fn.strip('\n').split(' ')[0] for fn in fns]

    for fn in fns:
        pf=open('data/Semantic3D.Net/raw/test/' + fn + '.txt')

        count=0
        while True:
            line=pf.readline()
            if not line:
                break
            count+=1
        pf.seek(0)

        points,labels=[],[]
        while True:
            line=pf.readline()
            if not line:
                break
            line=line.strip('\n')
            sublines=line.split(' ')
            x=float(sublines[0])
            y=float(sublines[1])
            z=float(sublines[2])
            intensity=float(sublines[3])
            r=float(sublines[4])
            g=float(sublines[5])
            b=float(sublines[6])
            points.append(np.asarray([x,y,z,r,g,b,intensity],dtype=np.float32))
            labels.append(0)

        pf.close()

        save_room_pkl('data/Semantic3D.Net/pkl/test/'+fn+'.pkl',
                      np.asarray(points,np.float32),np.asarray(labels,np.int32))

        print '{} done'.format(fn)


def semantic3d_presample_block():
    with open('cached/semantic3d_test_stems.txt','r') as f:
        fns=f.readlines()
        fns=[fn.strip('\n').split(' ')[0] for fn in fns]

    for fn in fns:
        fs=('data/Semantic3D.Net/pkl/test/' + fn + '.pkl')
        points,labels=read_room_pkl(fs)
        xyzs, rgbs, covars, lbls = sample_block(points,labels,ds_stride=0.03,block_size=50.0,
                                                block_stride=45.0,min_pn=128,use_rescale=False,
                                                use_flip=False,use_rotate=False,covar_ds_stride=0.01,
                                                covar_nn_size=0.1,gpu_gather=True)

        for t in xrange(len(xyzs)):
            points=np.concatenate([xyzs[t],rgbs[t]],axis=1)
            save_pkl('data/Semantic3D.Net/pkl/test_presample/' + fn + '_{}.pkl'.format(t),[points,lbls[t]])


def test_presample():
    for t in xrange(17):
        points,labels=read_pkl('data/Semantic3D.Net/pkl/test_presample/MarketplaceFeldkirch_Station4_rgb_intensity-reduced_{}.pkl'.format(t))
        print points.shape
        idxs=libPointUtil.gridDownsampleGPU(points,0.1,False)
        points=points[idxs]
        output_points('test_result/{}.txt'.format(t), points)



def semantic3d_test_to_block():
    with open('cached/semantic3d_test_stems.txt','r') as f:
        lines=f.readlines()
        fns=[fn.strip('\n').split(' ')[0] for fn in lines]
        pns=[int(fn.strip('\n').split(' ')[1]) for fn in lines]


    for fn,pn in zip(fns,pns):
        all_data=[[] for _ in xrange(13)]
        for t in xrange(pn):
            fs=('data/Semantic3D.Net/pkl/test_presample/' + fn + '_{}.pkl'.format(t))
            points,labels=read_room_pkl(fs)
            xyzs, rgbs, covars, lbls = sample_block(points,labels,ds_stride=sstride,block_size=bsize,
                                                    block_stride=bstride,min_pn=min_pn,use_rescale=False,
                                                    use_flip=False,use_rotate=False,covar_ds_stride=covar_ds_stride,
                                                    covar_nn_size=covar_nn_size,gpu_gather=True)

            print 'block num {}'.format(len(xyzs))

            data =  normalize_block_hierarchy(xyzs, rgbs, covars, lbls,
                                              bsize=bsize, nr1=nr1, nr2=nr2, nr3=nr3,
                                              vc1=vc1, vc2=vc2,resample=True, jitter_color=True,
                                              resample_low=resample_ratio_low,
                                              resample_high=resample_ratio_high,
                                              max_pt_num=max_pt_num)
            for i in xrange(13):
                all_data[i]+=data[i]

        save_pkl('data/Semantic3D.Net/block/test/'+fn+'.pkl',all_data)
        print '{} done'.format(fn)


def modelnet_dataset_to_block():
    from io_util import read_model_h5
    train_list=['data/ModelNet40/ply_data_train{}.h5'.format(i) for i in xrange(5)]
    test_list=['data/ModelNet40/ply_data_test{}.h5'.format(i) for i in xrange(2)]
    # train_list2=['data/ModelNet40/ply_data_train{}.pkl'.format(i) for i in xrange(5)]
    # test_list2=['data/ModelNet40/ply_data_test{}.pkl'.format(i) for i in xrange(2)]

    for fi,filename in enumerate(train_list[:2]):
        points,labels=read_model_h5(filename)
        data = normalize_model_hierarchy(points,False)
        app_data=[]
        app_data.append(labels)
        app_data+=data
        save_pkl('data/ModelNet40/ply_data_train{}.pkl'.format(fi),app_data)
        print len(app_data)
        print '{} done'.format(fi)

    # for fi,filename in enumerate(test_list):
    #     points,labels=read_model_h5(filename)
    #     data = normalize_model_hierarchy(points,False)
    #     app_data=[]
    #     app_data.append(labels)
    #     app_data+=data
    #     save_pkl('data/ModelNet40/ply_data_test{}.pkl'.format(fi),app_data)
    #     print '{} done'.format(fi)




if __name__=="__main__":
    # semantic3d_to_block(40.0,37.5,0.03)
    # get_intensity_distribution()
    # merge_train_files()
    # semantic3d_sample_training_block()
    # merge_train_files()
    # semantic3d_test_to_block()
    modelnet_dataset_to_block()


