from io_util import *
from aug_util import *
from concurrent.futures import ProcessPoolExecutor
import random
from draw_util import get_semantic3d_class_colors
import libPointUtil


# step 0 compute min Z
def semantic3d_sample_trainset_offset_z():
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    with open('cached/semantic3d_stems.txt','r') as f:
        stems=[line.split(' ')[0] for line in f.readlines()]

    train_list=semantic3d_read_train_block_list()
    f=open('cached/semantic3d_train_offsetz.txt','w')
    for stem in stems:
        pts,lbls=[],[]
        for tfs in train_list:
            if (not tfs.startswith(stem)) or (not tfs.endswith('0.pkl')): continue
            fs='data/Semantic3D.Net/block/train/'+tfs
            points,labels=read_room_pkl(fs) # [n,6],[n,1]
            idxs=libPointUtil.gridDownsampleGPU(points,0.2,False)
            pts.append(points[idxs])
            lbls.append(labels[idxs])

        pts=np.concatenate(pts,axis=0)
        idxs=libPointUtil.gridDownsampleGPU(pts,0.1,False)
        pts=pts[idxs]

        zs=pts[:,2]
        min_z=np.min(zs)
        zs-=np.min(zs)
        plt.figure()
        plt.hist(zs,200,range=(0,20))
        plt.savefig('test_result/{}.png'.format(stem))
        plt.close()

        hist,_=np.histogram(zs,np.arange(0.0,20.0,0.1),range=(0,20))
        offset_z=np.argmax(hist)*0.1+min_z
        f.write('{} {}\n'.format(stem,offset_z))

    f.close()


def semantic3d_read_map_offset_z():
    with open('cached/semantic3d_train_offsetz.txt','r') as f:
        stem_offset_map={}
        for line in f.readlines():
            line=line.strip('\n')
            stem=line.split(' ')[0]
            offset=float(line.split(' ')[1])
            stem_offset_map[stem]=offset

    return stem_offset_map


def semantic3d_test_sample_trainset_offset_z():
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    with open('cached/semantic3d_stems.txt','r') as f:
        stems=[line.split(' ')[0] for line in f.readlines()]

    stem_offset_map=semantic3d_read_map_offset_z()

    train_list=semantic3d_read_train_block_list()
    for stem in stems:
        pts,lbls=[],[]
        for tfs in train_list:
            if (not tfs.startswith(stem)) or (not tfs.endswith('0.pkl')): continue
            fs='data/Semantic3D.Net/block/train/'+tfs
            points,labels=read_room_pkl(fs) # [n,6],[n,1]
            idxs=libPointUtil.gridDownsampleGPU(points,0.2,False)
            pts.append(points[idxs])
            lbls.append(labels[idxs])

        pts=np.concatenate(pts,axis=0)
        idxs=libPointUtil.gridDownsampleGPU(pts,0.1,False)
        pts=pts[idxs]

        zs=pts[:,2]
        zs-=stem_offset_map[stem]
        plt.figure()
        plt.hist(zs,500,range=(-25,25))
        plt.savefig('test_result/{}_offseted.png'.format(stem))
        plt.close()


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

# step 1 write big block to block/train
def semantic3d_to_block(bsize=80.0,bstride=40.0,ds_stride=0.03):
    executor=ProcessPoolExecutor(6)

    fss,fns=read_semantic3d_pkl_stems()
    for fs,fn in zip(fss,fns):
        for ri in xrange(6):
            rot_ang=np.pi/12.0*ri
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

# step 2 get all filenames in block/train
def write_train_block_list():
    with open('cached/semantic3d_train_pkl.txt','w') as f:
        import os
        for fs in os.listdir('data/Semantic3D.Net/block/train/'):
            f.write(fs+'\n')


# test for step 1
def test_big_block():
    fss=semantic3d_read_train_block_list()
    random.shuffle(fss)
    colors=get_semantic3d_class_colors()
    fss=[fs for fs in fss if fs.startswith('untermaederbrunnen_station1_xyz_intensity_rgb')]
    orientations=[[] for _ in xrange(6)]
    for t in xrange(6):
        for fs in fss:
            if fs.split('_')[-1].startswith(str(t)):
                orientations[t].append(fs)

    for i in xrange(6):
        for t,fs in enumerate(orientations[i]):
            points,labels=read_pkl('data/Semantic3D.Net/block/train/'+fs)
            idxs=libPointUtil.gridDownsampleGPU(points,0.2,False)
            points=points[idxs]
            labels=labels[idxs]
            print points.shape
            output_points('test_result/{}_{}_colors.txt'.format(i,t), points)
            output_points('test_result/{}_{}_labels.txt'.format(i,t), points, colors[labels,:])


sample_stride=0.125
block_size=10.0
block_stride=5.0
min_point_num=512
covar_sample_stride=0.05
covar_neighbor_radius=0.2
max_pt_num=10240
def normalize_semantic3d_block(xyzs,rgbs,covars,lbls,offset_z,bsize=3.0,
                               resample=False,resample_low=0.8,resample_high=1.0,
                               jitter_color=False,jitter_val=2.5,max_pt_num=10240):
    bn=len(xyzs)
    block_mins=[]

    # t=0
    for bid in xrange(bn):
        if resample:
            pt_num=len(xyzs[bid])
            random_down_ratio=np.random.uniform(resample_low,resample_high)
            idxs=np.random.choice(pt_num,int(pt_num*random_down_ratio))
            xyzs[bid]=xyzs[bid][idxs,:]
            rgbs[bid]=rgbs[bid][idxs,:]
            lbls[bid]=lbls[bid][idxs]
            covars[bid]=covars[bid][idxs,:]

        if len(xyzs[bid])>max_pt_num:
            pt_num=len(xyzs[bid])
            ratio=max_pt_num/float(len(xyzs[bid]))
            idxs=np.random.choice(pt_num,int(pt_num*ratio))
            xyzs[bid]=xyzs[bid][idxs,:]
            rgbs[bid]=rgbs[bid][idxs,:]
            lbls[bid]=lbls[bid][idxs]
            covars[bid]=covars[bid][idxs,:]

        # offset center to zero
        # !!! dont rescale here since it will affect the neighborhood size !!!
        min_xyz=np.min(xyzs[bid],axis=0,keepdims=True)
        min_xyz[:,2]=offset_z
        min_xyz[:,:2]+=bsize/2.0
        xyzs[bid]-=min_xyz
        block_mins.append(min_xyz)

        if jitter_color:
            rgbs[bid][:,:3]+=np.random.uniform(-jitter_val,jitter_val,rgbs[bid][:,:3].shape)
            rgbs[bid][:,:3]-=128
            rgbs[bid][:,:3]/=(128+jitter_val)
        else:
            rgbs[bid][:,:3]-=128
            rgbs[bid][:,:3]/=128

        # sub mean div std
        if rgbs[bid].shape[1]>3:
            rgbs[bid][:,3]-=-1164.05
            rgbs[bid][:,3]/=600.0

    return xyzs,rgbs,covars,lbls,block_mins


def semantic3d_process_block(filename,offset_z):
    points,labels=read_room_pkl(filename) # [n,6],[n,1]
    xyzs, rgbs, covars, lbls=sample_block(points,labels,sample_stride,block_size,block_stride,min_pn=min_point_num,
                                          use_rescale=True,use_flip=True,use_rotate=False,
                                          covar_ds_stride=covar_sample_stride,covar_nn_size=covar_neighbor_radius,
                                          gpu_gather=True)
    # normalize rgbs
    xyzs, rgbs, covars, lbls, block_mins=normalize_semantic3d_block(xyzs,rgbs,covars,lbls,offset_z,block_size,
                                                                    resample=True,resample_low=0.8,resample_high=1.0,
                                                                    jitter_color=True,jitter_val=2.5,max_pt_num=max_pt_num)
    return xyzs, rgbs, covars, lbls, block_mins


def semantic3d_sample_single_file_training_block(tfs):
    stem_offset_map=semantic3d_read_map_offset_z()
    stem='_'.join(tfs.split('_')[:-2])
    offset_z=stem_offset_map[stem]
    fs='data/Semantic3D.Net/block/train/'+tfs
    all_data=[[] for _ in xrange(5)]
    for i in xrange(3):
        data=semantic3d_process_block(fs,offset_z)

        for k in xrange(5):
            all_data[k]+=data[k]

    save_pkl('data/Semantic3D.Net/block/sampled/'+tfs,all_data)
    print '{} done'.format(tfs)

# step 3 process block to block/sampled
def semantic3d_sample_training_block():
    executor=ProcessPoolExecutor(8)
    train_list=semantic3d_read_train_block_list()
    futures=[executor.submit(semantic3d_sample_single_file_training_block,tfs) for tfs in train_list]
    for f in futures: f.result()


def test_process_block():
    fss=semantic3d_read_train_block_list()
    random.shuffle(fss)
    colors=get_semantic3d_class_colors()
    fss=[fs for fs in fss if fs.startswith('untermaederbrunnen_station1_xyz_intensity_rgb')]
    orientations=[[] for _ in xrange(6)]
    for t in xrange(6):
        for fs in fss:
            if fs.split('_')[-1].startswith(str(t)):
                orientations[t].append(fs)

    for i in xrange(1):
        for t,fs in enumerate(orientations[i]):
            xyzs, rgbs, covars, lbls, block_mins=read_pkl('data/Semantic3D.Net/block/sampled/'+fs)
            for k in xrange(len(xyzs)):
                # idxs=libPointUtil.gridDownsampleGPU(xyzs[k],0.2,False)
                pts=xyzs[k]#[idxs]
                lbl=lbls[k]#[idxs]
                rgb=rgbs[k]#[idxs]
                print np.min(pts,axis=0),np.max(pts,axis=0)-np.min(pts,axis=0)
                output_points('test_result/colors{}_{}_{}.txt'.format(i,t,k), pts+block_mins[k], rgb*127+128)
                output_points('test_result/labels{}_{}_{}.txt'.format(i,t,k), pts+block_mins[k], colors[lbl,:])

# step 4 merge block to block/sampled/merged
def merge_train_files():
    with open('cached/semantic3d_stems.txt','r') as f:
        stems=[line.split(' ')[0] for line in f.readlines()]
    with open('cached/semantic3d_train_pkl.txt','r') as f:
        fs=[line.strip('\n') for line in f.readlines()]

    of=open('cached/semantic3d_merged_train.txt','w')
    for s in stems:
        idx=0
        all_data=[[] for _ in xrange(4)]
        for f in fs:
            if not f.startswith(s):
                continue
            data=read_pkl('data/Semantic3D.Net/block/sampled/train/'+f)

            for i in xrange(4):
                all_data[i]+=data[i]

            if len(all_data[0])>300:
                print len(all_data[0])
                save_pkl('data/Semantic3D.Net/block/sampled/merged/'+s+'_{}.pkl'.format(idx),all_data)
                all_data=[[] for _ in xrange(4)]
                idx+=1

        if len(all_data[0])>0:
            save_pkl('data/Semantic3D.Net/block/sampled/merged/'+s+'_{}.pkl'.format(idx),all_data)
            idx+=1

        of.write('{} {}\n'.format(s,idx))
        print '{} done'.format(s)

    of.close()


def merge_test_file():
    with open('cached/semantic3d_stems.txt','r') as f:
        stems=[line.split(' ')[0] for line in f.readlines()]
    with open('cached/semantic3d_train_pkl.txt','r') as f:
        fs=[line.strip('\n') for line in f.readlines()]

    of=open('cached/semantic3d_merged_test.txt','w')
    for s in stems:
        idx=0
        all_data=[[] for _ in xrange(4)]
        for f in fs:
            if (not f.startswith(s)) or (not f.endswith('0.pkl')):
                continue
            data=read_pkl('data/Semantic3D.Net/block/sampled/train/'+f)

            for i in xrange(4):
                all_data[i]+=data[i]

            if len(all_data[0])>300:
                print len(all_data[0])
                save_pkl('data/Semantic3D.Net/block/sampled/merged_test/'+s+'_{}.pkl'.format(idx),all_data)
                all_data=[[] for _ in xrange(4)]
                idx+=1

        if len(all_data[0])>0:
            save_pkl('data/Semantic3D.Net/block/sampled/merged_test/'+s+'_{}.pkl'.format(idx),all_data)
            idx+=1

        of.write('{} {}\n'.format(s,idx))
        print '{} done'.format(s)

    of.close()

# test merged
def test_merged():

    train_list,_=get_semantic3d_block_train_list()
    _,test_list=get_semantic3d_block_test_list()
    # train_list=['data/Semantic3D.Net/block/sampled/merged/'+fn for fn in train_list]
    test_list=['data/Semantic3D.Net/block/sampled/merged_test/'+fn for fn in test_list]
    read_fn=lambda model,filename: read_pkl(filename)
    # random.shuffle(train_list)
    # total=0
    # for fs in train_list:
    #     xyzs,rgbs,covars,lbls=read_fn('',fs)
    #     # lbls=np.concatenate(lbls)
    #     # print np.max(lbls),np.min(lbls)
    #     # for i in xrange(min(len(xyzs),5)):
    #     #     print np.min(xyzs[i],axis=0),np.max(xyzs[i],axis=0)
    #     #     print np.min(rgbs[i],axis=0),np.max(rgbs[i],axis=0)
    #     total+=len(xyzs)
    # print total

    random.shuffle(train_list)
    total=0
    label_counts=np.zeros(9)
    for fs in test_list:
        xyzs,rgbs,covars,lbls=read_fn('',fs)
        lbls=np.concatenate(lbls)
        label_count,_=np.histogram(lbls,range(10))
        label_counts+=label_count
        # print np.max(lbls),np.min(lbls)
        # for i in xrange(min(len(xyzs),5)):
        #     print np.min(xyzs[i],axis=0),np.max(xyzs[i],axis=0)
        #     print np.min(rgbs[i],axis=0),np.max(rgbs[i],axis=0)
        total+=len(xyzs)
    print total
    print label_counts


# testset step 1
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

# testset step 2
def semantic3d_testset_presample_block():
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


def semantic3d_process_test_block(filename):
    points,labels=read_room_pkl(filename) # [n,6],[n,1]
    xyzs, rgbs, covars, lbls=sample_block(points,labels,sample_stride,block_size,2.5,min_pn=min_point_num,
                                          use_rescale=False,use_flip=False,use_rotate=False,
                                          covar_ds_stride=covar_sample_stride,covar_nn_size=covar_neighbor_radius,
                                          gpu_gather=True)
    # normalize rgbs
    xyzs, rgbs, covars, lbls,block_mins=normalize_semantic3d_block(xyzs,rgbs,covars,lbls,block_size,
                                                        resample=False,resample_low=0.8,resample_high=1.0,
                                                        jitter_color=False,jitter_val=2.5,max_pt_num=max_pt_num)

    return xyzs, rgbs, covars, lbls,block_mins


def semantic3d_process_test_block_with_rotate(filename,rot_ang):
    points,labels=read_room_pkl(filename) # [n,6],[n,1]

    cosval = np.cos(rot_ang)
    sinval = np.sin(rot_ang)
    rot_m = np.array([[cosval,  sinval, 0],
                      [-sinval, cosval, 0],
                      [0, 0, 1]], dtype=np.float32)
    points[:, :3]=np.dot(points[:,:3],rot_m)
    points=np.ascontiguousarray(points,np.float32)

    xyzs, rgbs, covars, lbls=sample_block(points,labels,sample_stride,block_size,2.5,min_pn=1024,
                                          use_rescale=False,use_flip=False,use_rotate=False,
                                          covar_ds_stride=covar_sample_stride,covar_nn_size=covar_neighbor_radius,
                                          gpu_gather=True)
    # normalize rgbs
    xyzs, rgbs, covars, lbls,block_mins=normalize_semantic3d_block(xyzs,rgbs,covars,lbls,block_size,
                                                        resample=False,resample_low=0.8,resample_high=1.0,
                                                        jitter_color=False,jitter_val=2.5,max_pt_num=max_pt_num)

    return xyzs, rgbs, covars, lbls,block_mins

def semantic3d_test_to_block():
    with open('cached/semantic3d_test_stems.txt','r') as f:
        lines=f.readlines()
        fns=[fn.strip('\n').split(' ')[0] for fn in lines]
        pns=[int(fn.strip('\n').split(' ')[1]) for fn in lines]


    for fn,pn in zip(fns,pns):
        all_data=[[] for _ in xrange(5)]
        for t in xrange(pn):
            fs=('data/Semantic3D.Net/pkl/test_presample/' + fn + '_{}.pkl'.format(t))
            data=semantic3d_process_test_block(fs)
            for i in xrange(5):
                all_data[i]+=data[i]

        save_pkl('data/Semantic3D.Net/block/test/'+fn+'.pkl',all_data)
        print '{} done'.format(fn)

def semantic3d_test_to_block_with_rotate():
    with open('cached/semantic3d_test_stems.txt','r') as f:
        lines=f.readlines()
        fns=[fn.strip('\n').split(' ')[0] for fn in lines]
        pns=[int(fn.strip('\n').split(' ')[1]) for fn in lines]

    for ri in xrange(1,6):
        rot_ang=np.pi/12.0*ri
        for fn,pn in zip(fns,pns):
            all_data=[[] for _ in xrange(5)]
            for t in xrange(pn):
                fs=('data/Semantic3D.Net/pkl/test_presample/' + fn + '_{}.pkl'.format(t))
                data=semantic3d_process_test_block_with_rotate(fs,rot_ang)
                for i in xrange(5):
                    all_data[i]+=data[i]

            save_pkl('data/Semantic3D.Net/block/test_{}/'.format(ri)+fn+'.pkl',all_data)
            print '{} done'.format(fn)


def semantic3d_sample_test_set():
    fns,pns=get_semantic3d_testset()
    for fn,pn in zip(fns,pns):
        points, labels = read_room_pkl('data/Semantic3D.Net/pkl/test/' + fn + '.pkl')
        idxs=libPointUtil.gridDownsampleGPU(points,0.1,False)
        points=points[idxs]
        output_points('test_result/{}_color.txt'.format(fn), points)

if __name__=="__main__":
    semantic3d_sample_training_block()