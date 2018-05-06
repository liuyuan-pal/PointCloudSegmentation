from io_util import *
from aug_util import *
import libPointUtil
from concurrent.futures import ProcessPoolExecutor
from tf_ops.graph_pooling_layer import average_downsample
import  tensorflow as tf


def sample_large_block(beg, bi, ri, rm, fs, fn, min_p, bsize, ds_stride):
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
        offseted_points = points[:, :3]-np.expand_dims(min_p, axis=0)

        x_cond = (offseted_points[:, 0] >= beg[0]) & (offseted_points[:, 0] < beg[0] + bsize)
        y_cond = (offseted_points[:, 1] >= beg[1]) & (offseted_points[:, 1] < beg[1] + bsize)
        cond = x_cond & y_cond
        pn+=np.sum(cond)
        pts.append(points[cond])
        lbls.append(labels[cond])

    if pn>1024:
        pts = np.concatenate(pts, axis=0)
        lbls = np.concatenate(lbls, axis=0)

        print 'block {} pn {}'.format(bi,pn)
        save_room_pkl('data/Semantic3D.Net/context/large_block/' + fs + '_{}_{}.pkl'.format(bi,ri), pts, lbls)


def read_pkl_stems():
    with open('cached/semantic3d_stems.txt','r') as f:
        lines=f.readlines()
        fss,fns=[],[]
        for line in lines:
            line=line.strip('\n')
            fs,fn=line.split(' ')
            fss.append(fs)
            fns.append(int(fn))

    return fss,fns


# step 1 write big block to context/large_block
def sample_large_block_multi_cpu_process(bsize=80.0,bstride=40.0,ds_stride=0.03):
    executor=ProcessPoolExecutor(6)

    fss,fns=read_pkl_stems()
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
                futures.append(executor.submit(sample_large_block, beg, bi, ri, rot_m, fs, fn, min_p, bsize, ds_stride))
                # semantic3d_sample_block(beg,bi,fs,fn,min_p,bsize,ds_stride)

            for f in futures:
                f.result()

        print 'fs {} done'.format(fs)


# step 2 downsample global point
def build_avg_ds_session(ds_size,min_coor):
    pts_pl=tf.placeholder(tf.float32,[None,7],'pts')
    xyzs,feats=tf.split(pts_pl,[3,4],axis=1)
    ds_xyzs,ds_feats=average_downsample(xyzs,feats,ds_size,min_coor)
    ds_pts=tf.concat([ds_xyzs,ds_feats],axis=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    return sess,pts_pl,ds_pts

def global_avg_downsample():
    with open('cached/semantic3d_stems.txt', 'r') as f:
        stems = [line.split(' ')[0] for line in f.readlines()]

    sess, pts_pl, ds_pts_op = build_avg_ds_session(ds_size=2.0, min_coor=3000.0)

    train_list = read_large_block_list()
    for stem in stems:
        for k in xrange(6):
            pts, lbls = [], []
            for tfs in train_list:
                if (not tfs.startswith(stem)) or (not tfs.endswith('{}.pkl'.format(k))): continue
                fs = 'data/Semantic3D.Net/context/large_block/' + tfs
                points, labels = read_room_pkl(fs)  # [n,6],[n,1]
                idxs = libPointUtil.gridDownsampleGPU(points, 0.1, False)
                pts.append(points[idxs])
                lbls.append(labels[idxs])

            # downsample
            pts = np.concatenate(pts, axis=0)
            ds_pts=sess.run(ds_pts_op,feed_dict={pts_pl:pts})

            # compute covar
            ds_xyzs=np.ascontiguousarray(ds_pts[:,:3],np.float32)
            xyzs=np.ascontiguousarray(pts[:,:3],np.float32)
            nidxs=libPointUtil.findNeighborInAnotherCPU(xyzs,ds_xyzs,4.0)
            nlens=np.ascontiguousarray([len(idxs) for idxs in nidxs],np.int32)
            nbegs=compute_nidxs_bgs(nlens)
            nidxs=np.ascontiguousarray(np.concatenate(nidxs,axis=0),dtype=np.int32)
            covars=libPointUtil.computeCovarsGPU(xyzs,nidxs,nlens,nbegs)
            if np.sum(np.isnan(covars))>0:
                print stem,k
                idxs,_=np.nonzero(np.isnan(covars))
                for idx in idxs:
                    print '{} {}'.format(idx,nlens[idx])

                exit(0)

            ds_pts=np.concatenate([ds_pts,covars],axis=1)

            # output_points('test_result/{}_{}_ds.txt'.format(stem,k),pts)
            save_pkl('data/Semantic3D.Net/context/global_avg/{}_{}.pkl'.format(stem,k),ds_pts)


def test_global_avg_covars():
    from sklearn.cluster import KMeans
    with open('cached/semantic3d_stems.txt', 'r') as f:
        stems = [line.split(' ')[0] for line in f.readlines()]

    cluster_num=5
    colors=np.random.randint(0,256,[cluster_num,3])
    for stem in stems[:2]:
        for k in xrange(2):
            ds_pts=read_pkl('data/Semantic3D.Net/context/global_avg/{}_{}.pkl'.format(stem,k))
            kmeans=KMeans(cluster_num)
            feats=ds_pts[:,-9:]
            preds=kmeans.fit_predict(feats)
            output_points('test_result/{}_{}.txt'.format(stem,k),ds_pts,colors[preds])



def global_downsample():
    with open('cached/semantic3d_stems.txt', 'r') as f:
        stems = [line.split(' ')[0] for line in f.readlines()]

    train_list = read_large_block_list()
    for stem in stems:
        for k in xrange(6):
            pts, lbls = [], []
            for tfs in train_list:
                if (not tfs.startswith(stem)) or (not tfs.endswith('{}.pkl'.format(k))): continue
                fs = 'data/Semantic3D.Net/context/large_block/' + tfs
                points, labels = read_room_pkl(fs)  # [n,6],[n,1]
                idxs = libPointUtil.gridDownsampleGPU(points, 0.1, False)
                pts.append(points[idxs])
                lbls.append(labels[idxs])

            pts = np.concatenate(pts, axis=0)
            idxs = libPointUtil.gridDownsampleGPU(pts, 2.0, False)
            pts = pts[idxs]

            # output_points('test_result/{}_{}_ds.txt'.format(stem,k),pts)
            save_pkl('data/Semantic3D.Net/context/global/{}_{}.pkl'.format(stem,k),pts)


# step 3 write list
def write_large_block_list():
    with open('cached/semantic3d_context_large_block.txt','w') as f:
        import os
        for fs in os.listdir('data/Semantic3D.Net/context/large_block/'):
            f.write(fs+'\n')


def read_large_block_list():
    with open('cached/semantic3d_context_large_block.txt', 'r') as f:
        fs = [line.strip('\n') for line in f.readlines()]
    return fs


def get_context_train_test(
        test_stems=('sg28_station4_intensity_rgb','untermaederbrunnen_station3_xyz_intensity_rgb')
):
    train_list,test_list=[],[]
    fss=read_large_block_list()
    for fs in fss:
        stem='_'.join(fs.split('_')[:-2])
        if stem in test_stems:
            if fs.endswith('0.pkl'):
                test_list.append(fs)
        else:
            train_list.append(fs)

    return train_list,test_list


# step 4 compute offset z
def compute_context_offset_z():
    # import matplotlib as mpl
    # mpl.use('Agg')
    # import matplotlib.pyplot as plt

    with open('cached/semantic3d_stems.txt','r') as f:
        stems=[line.split(' ')[0] for line in f.readlines()]

    train_list=read_large_block_list()
    f=open('cached/semantic3d_context_offsetz.txt','w')
    for stem in stems:
        pts,lbls=[],[]
        for tfs in train_list:
            if (not tfs.startswith(stem)) or (not tfs.endswith('0.pkl')): continue
            fs='data/Semantic3D.Net/context/large_block/'+tfs
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
        # plt.figure()
        # plt.hist(zs,200,range=(0,20))
        # plt.savefig('test_result/{}.png'.format(stem))
        # plt.close()

        hist,_=np.histogram(zs,np.arange(0.0,20.0,0.1),range=(0,20))
        offset_z=np.argmax(hist)*0.1+min_z
        f.write('{} {}\n'.format(stem,offset_z))

    f.close()


def get_context_offset_z():
    with open('cached/semantic3d_context_offsetz.txt','r') as f:
        stem_offset_map={}
        for line in f.readlines():
            line=line.strip('\n')
            stem=line.split(' ')[0]
            offset=float(line.split(' ')[1])
            stem_offset_map[stem]=offset

    return stem_offset_map


def test_context_offset_z():
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    with open('cached/semantic3d_stems.txt','r') as f:
        stems=[line.split(' ')[0] for line in f.readlines()]

    stem_offset_map=get_context_offset_z()

    train_list=read_large_block_list()
    for stem in stems:
        pts,lbls=[],[]
        for tfs in train_list:
            if (not tfs.startswith(stem)) or (not tfs.endswith('0.pkl')): continue
            fs='data/Semantic3D.Net/context/large_block/'+tfs
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


# step 5
def compute_context_xyzs(global_xyz, xyzs, context_len=50.0):
    squared_context_len=context_len**2
    context_xyzs=[]
    for i in xrange(len(xyzs)):
        cen_xy=np.mean(xyzs[i],axis=0,keepdims=True)[:,:2]
        mask=np.sum((global_xyz[:,:2]-cen_xy)**2,axis=1)<squared_context_len
        context_xyzs.append(global_xyz[mask])

    return context_xyzs


def compute_context_idxs(context_xyzs,xyzs):
    context_idxs=[]
    for i in xrange(len(xyzs)):
        xyz=np.ascontiguousarray(xyzs[i][:,:3],np.float32)
        context_xyz=np.ascontiguousarray(context_xyzs[i][:,:3],np.float32)
        leaf_size=min(context_xyz.shape[0]-1,15)
        idxs=libPointUtil.findNeighborInAnotherCPU(context_xyz,xyz,1,leaf_size)
        idxs=np.squeeze(np.asarray(idxs,np.int32),axis=1)
        assert idxs.shape[0]==xyz.shape[0]
        context_idxs.append(idxs)

    return context_idxs


def sample_context_block(tfs, points, labels, global_points, ds_stride, block_size, block_stride, min_pn,
                         use_rescale=False, use_flip=False, covar_ds_stride=0.03, covar_nn_size=0.1,
                         context_len=50.0):

    xyzs=np.ascontiguousarray(points[:,:3])
    rgbs=np.ascontiguousarray(points[:,3:])
    min_xyz=np.min(xyzs,axis=0,keepdims=True)
    max_xyz=np.max(xyzs,axis=0,keepdims=True)
    covar_ds_idxs=libPointUtil.gridDownsampleGPU(xyzs, covar_ds_stride, False)
    covar_ds_xyzs=np.ascontiguousarray(xyzs[covar_ds_idxs,:])

    # flip
    if use_flip:
        if random.random()<0.5:
            covar_ds_xyzs=swap_xy(covar_ds_xyzs)
            global_points=swap_xy(global_points)
            min_xyz=swap_xy(min_xyz)
            max_xyz=swap_xy(max_xyz)

        if random.random()<0.5:
            covar_ds_xyzs=flip(covar_ds_xyzs,axis=0)
            global_points=flip(global_points,axis=0)
            min_xyz[:,0],max_xyz[:,0]=-max_xyz[:,0],-min_xyz[:,0]

        if random.random()<0.5:
            covar_ds_xyzs=flip(covar_ds_xyzs,axis=1)
            global_points=flip(global_points,axis=1)
            min_xyz[:,1],max_xyz[:,1]=-max_xyz[:,1],-min_xyz[:,1]

    # rescale
    if use_rescale:
        rescale=np.random.uniform(0.9,1.1,[1,3])
        covar_ds_xyzs[:,:3]*=rescale
        global_points[:,:3]*=rescale
        min_xyz*=rescale
        max_xyz*=rescale

    ds_idxs=libPointUtil.gridDownsampleGPU(covar_ds_xyzs,ds_stride,False)

    # compute covar
    covar_nidxs=libPointUtil.findNeighborRadiusCPU(covar_ds_xyzs,ds_idxs,covar_nn_size)
    covar_nidxs_lens=np.ascontiguousarray([len(idxs) for idxs in covar_nidxs],np.int32)
    covar_nidxs_bgs=compute_nidxs_bgs(covar_nidxs_lens)
    covar_nidxs=np.ascontiguousarray(np.concatenate(covar_nidxs,axis=0),dtype=np.int32)
    covars=libPointUtil.computeCovarsGPU(covar_ds_xyzs,covar_nidxs,covar_nidxs_lens,covar_nidxs_bgs)

    xyzs=covar_ds_xyzs[ds_idxs,:]
    rgbs=rgbs[covar_ds_idxs,:][ds_idxs,:]
    lbls=labels[covar_ds_idxs][ds_idxs]

    xyzs-=min_xyz
    idxs=uniform_sample_block(xyzs,block_size,block_stride,min_pn=min_pn)
    xyzs+=min_xyz

    xyzs, rgbs, covars, lbls=fetch_subset([xyzs,rgbs,covars,lbls],idxs)

    context_xyzs=compute_context_xyzs(global_points,xyzs,context_len=context_len)

    for ci,ctx_xyz in enumerate(context_xyzs):
        if ctx_xyz.shape[0]==0:
            print '!!!! error {}'.format(tfs)
            raise RuntimeError

    context_idxs=compute_context_idxs(context_xyzs,xyzs)

    return xyzs, rgbs, covars, lbls, context_xyzs, context_idxs


def test_sample_context_block():
    train_list=read_large_block_list()
    random.shuffle(train_list)
    for tfs in ['domfountain_station2_xyz_intensity_rgb_2_2.pkl']:
        print tfs
        stem='_'.join(tfs.split('_')[:-2])
        ri=int(tfs[-5])

        # read points
        fs='data/Semantic3D.Net/context/large_block/'+tfs
        points,labels=read_room_pkl(fs) # [n,6],[n,1]

        # read global points
        global_points=read_pkl('data/Semantic3D.Net/context/global/{}_{}.pkl'.format(stem,ri))
        # rotate global points
        # output_points('test_result/gloabl.txt',global_points)
        # output_points('test_result/local.txt',points)

        xyzs, rgbs, covars, lbls, context_xyzs, context_idxs= \
            sample_context_block(tfs, points, labels, global_points, sample_stride, block_size, block_stride,
                                 min_pn=min_point_num, use_rescale=False, use_flip=False, context_len=50.0,
                                 covar_ds_stride=covar_sample_stride, covar_nn_size=covar_neighbor_radius)

        for k in xrange(len(xyzs[:10])):
            output_points('test_result/{}xys.txt'.format(k),xyzs[k],rgbs[k])
            output_points('test_result/{}context_xys.txt'.format(k),context_xyzs[k])
            colors=np.random.randint(0,255,[len(context_xyzs[k]),3])
            output_points('test_result/{}context_xys_rcolor.txt'.format(k),context_xyzs[k],colors)
            output_points('test_result/{}xys_rcolor.txt'.format(k),xyzs[k],colors[context_idxs[k],:])

        print np.mean([len(cxyzs) for cxyzs in context_xyzs])


def normalize_context_block(xyzs,rgbs,covars,lbls,ctx_xyzs,ctx_idxs,offset_z,bsize=3.0,
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
            ctx_idxs[bid]=ctx_idxs[bid][idxs]

        if len(xyzs[bid])>max_pt_num:
            pt_num=len(xyzs[bid])
            ratio=max_pt_num/float(len(xyzs[bid]))
            idxs=np.random.choice(pt_num,int(pt_num*ratio))
            xyzs[bid]=xyzs[bid][idxs,:]
            rgbs[bid]=rgbs[bid][idxs,:]
            lbls[bid]=lbls[bid][idxs]
            covars[bid]=covars[bid][idxs,:]
            ctx_idxs[bid]=ctx_idxs[bid][idxs]

        # offset center to zero
        # !!! dont rescale here since it will affect the neighborhood size !!!
        min_xyz=np.min(xyzs[bid],axis=0,keepdims=True)
        min_xyz[:,2]=offset_z
        min_xyz[:,:2]+=bsize/2.0
        xyzs[bid]-=min_xyz
        ctx_xyzs[bid][:,:3]-=min_xyz
        block_mins.append(min_xyz)

        # color
        if jitter_color:
            rgbs[bid][:,:3]+=np.random.uniform(-jitter_val,jitter_val,rgbs[bid][:,:3].shape)
            rgbs[bid][:,:3]-=128
            rgbs[bid][:,:3]/=(128+jitter_val)
        else:
            rgbs[bid][:,:3]-=128
            rgbs[bid][:,:3]/=128

        ctx_xyzs[bid][:, 3:6]-=128
        ctx_xyzs[bid][:, 3:6]/=128

        # intensity sub mean div std
        if rgbs[bid].shape[1]>3:
            rgbs[bid][:,3]-=-1164.05
            rgbs[bid][:,3]/=600.0

        ctx_xyzs[bid][:, 6]-=-1164.05
        ctx_xyzs[bid][:, 6]/=600.0

    return xyzs,rgbs,covars,lbls,ctx_xyzs,ctx_idxs,block_mins


sample_stride=0.125
block_size=10.0
block_stride=5.0
contex_len=50.0
min_point_num=512
covar_sample_stride=0.05
covar_neighbor_radius=0.5
max_pt_num=10240


def process_context_block(fs):
    stem='_'.join(fs.split('_')[:-2])
    ri=int(fs[-5])
    points,labels=read_room_pkl('data/Semantic3D.Net/context/large_block/'+fs) # [n,6],[n,1]
    global_points=read_pkl('data/Semantic3D.Net/context/global_avg/{}_{}.pkl'.format(stem,ri))

    stem2offset=get_context_offset_z()
    offset_z=stem2offset[stem]
    xyzs, rgbs, covars, lbls, ctx_xyzs, ctx_idxs= \
        sample_context_block(fs, points, labels, global_points, sample_stride,
                             block_size, block_stride, min_pn=min_point_num,
                             use_rescale=True, use_flip=True, context_len=contex_len,
                             covar_ds_stride=covar_sample_stride,
                             covar_nn_size=covar_neighbor_radius)

    # normalize rgbs
    xyzs, rgbs, covars, lbls, ctx_xyzs, ctx_idxs, block_mins= \
        normalize_context_block(xyzs,rgbs,covars,lbls,ctx_xyzs,ctx_idxs,offset_z,block_size,
                                resample=True,resample_low=0.8,resample_high=1.0,
                                jitter_color=True,jitter_val=2.5,max_pt_num=max_pt_num)
    return xyzs, rgbs, covars, lbls, ctx_xyzs, ctx_idxs, block_mins


def test_process_context_block():
    from draw_util import get_semantic3d_class_colors
    lbl_colors=get_semantic3d_class_colors()
    fss=read_large_block_list()
    random.shuffle(fss)
    for fs in fss[:1]:
        print fs
        # xyzs, rgbs, covars, lbls, ctx_xyzs, ctx_idxs, block_mins = process_context_block(fs)
        xyzs, rgbs, covars, lbls, ctx_xyzs, ctx_idxs, block_mins = read_pkl('data/Semantic3D.Net/context/block_avg/'+fs)


        for k in xrange(len(xyzs[:10])):
            print np.min(lbls[k]),np.max(lbls[k])
            ctx_xyzs[k][:,3:6]*=127
            ctx_xyzs[k][:,3:6]+=128
            rgbs[k]*=127
            rgbs[k]+=128
            xyzs[k]+=block_mins[k]
            ctx_xyzs[k][:,:3]+=block_mins[k]

            output_points('test_result/{}xyzs.txt'.format(k),xyzs[k],rgbs[k])
            output_points('test_result/{}lbls.txt'.format(k),xyzs[k],lbl_colors[lbls[k]])
            output_points('test_result/{}context_xyzs.txt'.format(k),ctx_xyzs[k])

            print len(ctx_xyzs[k])
            colors=np.random.randint(0,255,[len(ctx_xyzs[k]),3])
            output_points('test_result/{}context_xyzs_rcolor.txt'.format(k),ctx_xyzs[k],colors)
            output_points('test_result/{}xyzs_rcolor.txt'.format(k),xyzs[k],colors[ctx_idxs[k],:])
            cluster_num=5
            from sklearn.cluster import KMeans
            colors=np.random.randint(0,255,[cluster_num,3])
            kmeans=KMeans(cluster_num)
            preds=kmeans.fit_predict(covars[k])
            output_points('test_result/{}xyzs_cluster.txt'.format(k),xyzs[k],colors[preds])



def process_and_save(fs):
    all_data=[[] for _ in xrange(7)]
    for i in xrange(3):
        data=process_context_block(fs)
        for k in xrange(7):
            all_data[k]+=data[k]

    save_pkl('data/Semantic3D.Net/context/block/'+fs,all_data)
    print '{} done'.format(fs)


def process_context_block_multi_cpu_process():
    executor=ProcessPoolExecutor(8)
    train_list=semantic3d_read_train_block_list()
    futures=[executor.submit(process_and_save,tfs) for tfs in train_list]
    for f in futures: f.result()


def compute_weights():
    fss=read_large_block_list()
    labelweights=np.zeros(9)
    for fs in fss:
        xyzs, rgbs, covars, lbls, ctx_xyzs, ctx_idxs, block_mins = read_pkl('data/Semantic3D.Net/context/block/'+fs)
        # print len(lbls),fs
        if len(lbls)==0: continue
        lbls=np.concatenate(lbls,axis=0)
        hist,_=np.histogram(lbls,xrange(10))
        labelweights+=hist

    labelweights = labelweights.astype(np.float32)
    labelweights = labelweights / np.sum(labelweights)
    labelweights = 1 / np.log(1.2 + labelweights)
    print labelweights


##########################process test_set###################

def testset_test_large_block():
    with open('cached/semantic3d_test_stems.txt','r') as f:
        lines=f.readlines()
        fss=[fn.strip('\n').split(' ')[0] for fn in lines]
        fns=[int(fn.strip('\n').split(' ')[1]) for fn in lines]

    for fs,fn in zip(fss,fns):
        for fni in xrange(fn):
            points,labels=read_room_pkl('data/Semantic3D.Net/context/test_large_block/{}_{}.pkl'.format(fs,fni))
            idxs=libPointUtil.gridDownsampleGPU(points,0.5,False)
            points=points[idxs]
            output_points('test_result/{}_{}.txt'.format(fs,fni),points)


# we already have large block in context/test_large_block
# step 1 compute min_z
def testset_compute_context_offset_z():
    with open('cached/semantic3d_test_stems.txt','r') as f:
        lines=f.readlines()
        fss=[fn.strip('\n').split(' ')[0] for fn in lines]
        fns=[int(fn.strip('\n').split(' ')[1]) for fn in lines]

    f=open('cached/semantic3d_test_context_offsetz.txt','w')
    for fs,fn in zip(fss,fns):
        pts=[]
        for fni in xrange(fn):
            points, labels =read_room_pkl('data/Semantic3D.Net/context/test_large_block/{}_{}.pkl'.format(fs,fni))
            idxs=libPointUtil.gridDownsampleGPU(points,0.1,False)
            pts.append(points[idxs])

        pts=np.concatenate(pts,axis=0)
        idxs=libPointUtil.gridDownsampleGPU(pts,0.1,False)
        pts=pts[idxs]

        zs=pts[:,2]
        min_z=np.min(zs)
        zs-=np.min(zs)

        hist,_=np.histogram(zs,np.arange(0.0,20.0,0.1),range=(0,20))
        offset_z=np.argmax(hist)*0.1+min_z
        f.write('{} {}\n'.format(fs,offset_z))

    f.close()


def testset_get_context_offset_z():
    with open('cached/semantic3d_test_context_offsetz.txt','r') as f:
        stem_offset_map={}
        for line in f.readlines():
            line=line.strip('\n')
            stem=line.split(' ')[0]
            offset=float(line.split(' ')[1])
            stem_offset_map[stem]=offset

    return stem_offset_map


def testset_test_context_offset_z():
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    with open('cached/semantic3d_test_stems.txt','r') as f:
        lines=f.readlines()
        fss=[fn.strip('\n').split(' ')[0] for fn in lines]
        fns=[int(fn.strip('\n').split(' ')[1]) for fn in lines]

    stem_offset_map=testset_get_context_offset_z()
    for fs,fn in zip(fss,fns):
        pts=[]
        for fni in xrange(fn):
            points, labels =read_room_pkl('data/Semantic3D.Net/context/test_large_block/{}_{}.pkl'.format(fs,fni))
            idxs=libPointUtil.gridDownsampleGPU(points,0.1,False)
            pts.append(points[idxs])

        pts=np.concatenate(pts,axis=0)
        idxs=libPointUtil.gridDownsampleGPU(pts,0.1,False)
        pts=pts[idxs]

        zs=pts[:,2]
        zs-=stem_offset_map[fs]
        plt.figure()
        plt.hist(zs,500,range=(-25,25))
        plt.savefig('test_result/{}_offseted.png'.format(fs))
        plt.close()


# step 2 global avg
def testset_global_downsample_avg():
    with open('cached/semantic3d_test_stems.txt','r') as f:
        lines=f.readlines()
        fss=[fn.strip('\n').split(' ')[0] for fn in lines]
        fns=[int(fn.strip('\n').split(' ')[1]) for fn in lines]

    sess, pts_pl, ds_pts_op = build_avg_ds_session(ds_size=2.0, min_coor=3000.0)

    for fs,fn in zip(fss,fns):
        pts=[]
        for fni in xrange(fn):
            points, labels = read_room_pkl('data/Semantic3D.Net/context/test_large_block/{}_{}.pkl'.format(fs,fni))  # [n,6],[n,1]
            idxs = libPointUtil.gridDownsampleGPU(points, 0.1, False)
            pts.append(points[idxs])

        # downsample
        pts = np.concatenate(pts, axis=0)
        ds_pts=sess.run(ds_pts_op,feed_dict={pts_pl:pts})

        # compute covar
        ds_xyzs=np.ascontiguousarray(ds_pts[:,:3],np.float32)
        xyzs=np.ascontiguousarray(pts[:,:3],np.float32)
        nidxs=libPointUtil.findNeighborInAnotherCPU(xyzs,ds_xyzs,4.0)
        nlens=np.ascontiguousarray([len(idxs) for idxs in nidxs],np.int32)
        nbegs=compute_nidxs_bgs(nlens)
        nidxs=np.ascontiguousarray(np.concatenate(nidxs,axis=0),dtype=np.int32)
        covars=libPointUtil.computeCovarsGPU(xyzs,nidxs,nlens,nbegs)
        if np.sum(np.isnan(covars))>0:
            print fs
            idxs,_=np.nonzero(np.isnan(covars))
            for idx in idxs:
                print '{} {}'.format(idx,nlens[idx])

            exit(0)

        ds_pts=np.concatenate([ds_pts,covars],axis=1)

        # output_points('test_result/{}_{}_ds.txt'.format(stem,k),pts)
        save_pkl('data/Semantic3D.Net/context/test_global_avg/{}.pkl'.format(fs),ds_pts)


def testset_test_global_downsample():
    with open('cached/semantic3d_test_stems.txt','r') as f:
        lines=f.readlines()
        fss=[fn.strip('\n').split(' ')[0] for fn in lines]

    for fs in fss:
        points=read_pkl('data/Semantic3D.Net/context/test_global_avg/{}.pkl'.format(fs))
        output_points('test_result/{}.txt'.format(fs),points)


# step 3 sample block
def testset_process_context_block(fs,fni):
    points,labels=read_room_pkl('data/Semantic3D.Net/context/test_large_block/{}_{}.pkl'.format(fs,fni)) # [n,6],[n,1]
    global_points=read_pkl('data/Semantic3D.Net/context/test_global_avg/{}.pkl'.format(fs))

    stem2offset=testset_get_context_offset_z()
    offset_z=stem2offset[fs]
    xyzs, rgbs, covars, lbls, ctx_xyzs, ctx_idxs= \
        sample_context_block(fs, points, labels, global_points, sample_stride,
                             block_size, block_stride, min_pn=32,
                             use_rescale=False, use_flip=False, context_len=contex_len,
                             covar_ds_stride=covar_sample_stride,
                             covar_nn_size=covar_neighbor_radius)

    # normalize rgbs
    xyzs, rgbs, covars, lbls, ctx_xyzs, ctx_idxs, block_mins= \
        normalize_context_block(xyzs,rgbs,covars,lbls,ctx_xyzs,ctx_idxs,offset_z,block_size,
                                resample=False,jitter_color=False,max_pt_num=max_pt_num)
    return xyzs, rgbs, covars, lbls, ctx_xyzs, ctx_idxs, block_mins


def testset_process_and_save():
    with open('cached/semantic3d_test_stems.txt','r') as f:
        lines=f.readlines()
        fss=[fn.strip('\n').split(' ')[0] for fn in lines]
        fns=[int(fn.strip('\n').split(' ')[1]) for fn in lines]

    for fs,fn in zip(fss,fns):
        all_data=[[] for _ in xrange(7)]
        for fni in xrange(fn):
            data=testset_process_context_block(fs,fni)
            for t in xrange(7):
                all_data[t]+=data[t]

        save_pkl('data/Semantic3D.Net/context/test_block_avg/{}.pkl'.format(fs),all_data)
        print '{} done'.format(fs)


def testset_test_process_context_block():
    with open('cached/semantic3d_test_stems.txt','r') as f:
        lines=f.readlines()
        fss=[fn.strip('\n').split(' ')[0] for fn in lines]

    for fs in fss[:1]:
        xyzs, rgbs, covars, lbls, ctx_xyzs, ctx_idxs, block_mins = read_pkl('data/Semantic3D.Net/context/test_block_avg/'+fs+'.pkl')

        print len(xyzs)
        for k in np.random.choice(np.arange(len(xyzs)),10,False):
            print np.min(lbls[k]),np.max(lbls[k])
            ctx_xyzs[k][:,3:6]*=127
            ctx_xyzs[k][:,3:6]+=128
            rgbs[k]*=127
            rgbs[k]+=128
            xyzs[k]+=block_mins[k]
            ctx_xyzs[k][:,:3]+=block_mins[k]

            output_points('test_result/{}xyzs.txt'.format(k),xyzs[k],rgbs[k])
            # output_points('test_result/{}lbls.txt'.format(k),xyzs[k],lbl_colors[lbls[k]])
            output_points('test_result/{}context_xyzs.txt'.format(k),ctx_xyzs[k])

            print len(ctx_xyzs[k])
            colors=np.random.randint(0,255,[len(ctx_xyzs[k]),3])
            output_points('test_result/{}context_xyzs_rcolor.txt'.format(k),ctx_xyzs[k],colors)
            output_points('test_result/{}xyzs_rcolor.txt'.format(k),xyzs[k],colors[ctx_idxs[k],:])
            # cluster_num=5
            # from sklearn.cluster import KMeans
            # colors=np.random.randint(0,255,[cluster_num,3])
            # kmeans=KMeans(cluster_num)
            # preds=kmeans.fit_predict(covars[k])
            # output_points('test_result/{}xyzs_cluster.txt'.format(k),xyzs[k],colors[preds])


if __name__=="__main__":
    testset_test_process_context_block()
    # fss=read_large_block_list()
    # count=0
    # for fs in fss:
    #     xyzs, rgbs, covars, lbls, ctx_xyzs, ctx_idxs, block_mins = read_pkl('data/Semantic3D.Net/context/block/'+fs)
    #     count+=len(xyzs)
    #
    # print count