import numpy as np
import libPointUtil
import random
from concurrent.futures import ThreadPoolExecutor
import functools


def flip(points,axis=0):
    result_points=points[:]
    result_points[:,axis]=-result_points[:,axis]
    return result_points


def swap_xy(points):
    result_points = np.empty_like(points, dtype=np.float32)
    result_points[:,0]=points[:,1]
    result_points[:,1]=points[:,0]
    result_points[:,2:]=points[:,2:]

    return result_points


def rotate(xyz,rotation_angle):
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, -sinval, 0],
                                [sinval, cosval,  0],
                                [      0,      0, 1]],dtype=np.float32)
    xyz=np.dot(xyz,rotation_matrix)
    return xyz


def get_list(maxx,block_size,stride,resample_ratio=0.03):
    x_list=[]
    spacex=maxx-block_size
    if spacex<0: x_list.append(0)
    else:
        x_list+=list(np.arange(0,spacex,stride))
        if (spacex-int(spacex/stride)*stride)/block_size>resample_ratio:
            x_list+=list(np.arange(spacex,0,-stride))

    return x_list


def get_list_without_back_sample(maxx,block_size,stride):
    x_list=[]
    spacex=maxx-block_size
    if spacex<0: x_list.append(0)
    else:
        x_list+=list(np.arange(0,spacex,stride))
        x_list.append(spacex)

    return x_list


def uniform_sample_block(xyz, block_size=3.0, stride=1.5, min_pn=2048, normalized=True):
    assert stride<=block_size

    if not normalized:
        xyz -= np.min(xyz,axis=0,keepdims=True)

    # uniform sample
    max_xyz=np.max(xyz,axis=0,keepdims=True)
    maxx,maxy=max_xyz[0,0],max_xyz[0,1]
    beg_list=[]
    x_list=get_list_without_back_sample(maxx,block_size,stride)
    y_list=get_list_without_back_sample(maxy,block_size,stride)
    for x in x_list:
        for y in y_list:
            beg_list.append((x,y))

    idxs=[]
    for beg in beg_list:
        x_cond=(xyz[:,0]>=beg[0])&(xyz[:,0]<beg[0]+block_size)
        y_cond=(xyz[:,1]>=beg[1])&(xyz[:,1]<beg[1]+block_size)
        cond=x_cond&y_cond
        if(np.sum(cond)<min_pn):
            continue
        idxs.append((np.nonzero(cond))[0])

    return idxs


def random_rotate_sample_block(points,labels,block_size=3.0,stride=1.5,rotation_angle=0.0,min_pn=2048):
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    xyz = np.ascontiguousarray(points[:,:3], dtype=np.float32)
    rgb = np.ascontiguousarray(points[:,3:], dtype=np.float32)

    block_idxs = libPointUtil.sampleRotatedBlockGPU(xyz,stride,block_size,rotation_angle)

    # rotate back
    cosval = np.cos(-rotation_angle)
    sinval = np.sin(-rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval,  0],
                                [      0,      0, 1]],dtype=np.float32)
    xyz=np.dot(xyz,rotation_matrix)

    # append
    block_xyz_list,block_rgb_list,block_label_list=[],[],[]
    for idxs in block_idxs:
        if len(idxs)<min_pn: continue
        block_xyz_list.append(xyz[idxs,:])
        block_rgb_list.append(rgb[idxs,:])
        block_label_list.append(labels[idxs,:])

    return block_xyz_list,block_rgb_list,block_label_list


def fetch_subset(all,idxs,subsets=None):
    if subsets is None:
        subsets=[[] for _ in xrange(len(all))]

    for item,subset in zip(all,subsets):
        for idx in idxs:
            subset.append(item[idx,:])

    return subsets


def sample_block(points, labels, ds_stride, block_size, block_stride, min_pn,
                 use_rescale=False, use_flip=False, use_rotate=False,
                 covar_ds_stride=0.03, covar_nn_size=0.1):
    # import time
    # t=time.time()
    xyzs=np.ascontiguousarray(points[:,:3])
    rgbs=np.ascontiguousarray(points[:,3:])

    covar_ds_idxs=libPointUtil.gridDownsampleGPU(xyzs, covar_ds_stride, False)
    covar_ds_xyzs=np.ascontiguousarray(xyzs[covar_ds_idxs,:])
    # print 'ds1 cost {} s'.format(time.time()-t)

    # t=time.time()
    # flip
    if use_flip:
        if random.random()<0.5:
            covar_ds_xyzs=swap_xy(covar_ds_xyzs)
        if random.random()<0.5:
            covar_ds_xyzs=flip(covar_ds_xyzs,axis=0)
        if random.random()<0.5:
            covar_ds_xyzs=flip(covar_ds_xyzs,axis=1)

    # rescale
    if use_rescale:
        x_scale=np.random.uniform(0.9,1.1)
        y_scale=np.random.uniform(0.9,1.1)
        z_scale=np.random.uniform(0.9,1.1)
        covar_ds_xyzs[:,0]*=x_scale
        covar_ds_xyzs[:,1]*=y_scale
        covar_ds_xyzs[:,2]*=z_scale

    # rotate
    if use_rotate:
        if random.random()>0.3:
            angle=random.random()*np.pi/2.0
            covar_ds_xyzs=rotate(covar_ds_xyzs,angle)
    # print 'flip rescale rotate cost {} s'.format(time.time()-t)

    # t=time.time()
    ds_idxs=libPointUtil.gridDownsampleGPU(covar_ds_xyzs,ds_stride,False)
    # print 'ds2 cost {} s'.format(time.time()-t)

    # t=time.time()
    covar_nidxs=libPointUtil.findNeighborRadiusCPU(covar_ds_xyzs,ds_idxs,covar_nn_size)
    # print 'flann cost {} s'.format(time.time()-t)

    # t=time.time()
    covar_nidxs_lens=np.ascontiguousarray([len(idxs) for idxs in covar_nidxs],np.int32)
    covar_nidxs_bgs=compute_nidxs_bgs(covar_nidxs_lens)
    covar_nidxs=np.ascontiguousarray(np.concatenate(covar_nidxs,axis=0),dtype=np.int32)
    # print 'lens2bgs cost {} s'.format(time.time()-t)

    # t=time.time()
    covars=libPointUtil.computeCovarsGPU(covar_ds_xyzs,covar_nidxs,covar_nidxs_lens,covar_nidxs_bgs)
    # print 'covars cost {} s'.format(time.time()-t)

    # t=time.time()
    xyzs=covar_ds_xyzs[ds_idxs,:]
    rgbs=rgbs[covar_ds_idxs,:][ds_idxs,:]
    lbls=labels[covar_ds_idxs,:][ds_idxs,:]
    # print 'index cost {} s'.format(time.time()-t)

    # t=time.time()
    points-=np.min(points,axis=0,keepdims=True)
    idxs = uniform_sample_block(xyzs,block_size,block_stride,normalized=False,min_pn=min_pn)
    # print 'sample cost {} s'.format(time.time()-t)
    # t=time.time()
    xyzs, rgbs, covars, lbls=fetch_subset([xyzs,rgbs,covars,lbls],idxs)
    # print 'fetch cost {} s'.format(time.time()-t)

    return xyzs, rgbs, covars, lbls


def compute_nidxs_bgs(nidxs_lens):
    csum=0
    nidxs_bgs=np.empty_like(nidxs_lens,dtype=np.int32)
    for i,lval in enumerate(nidxs_lens):
        nidxs_bgs[i]=csum
        csum+=lval
    return nidxs_bgs


def compute_cidxs(nidxs_lens):
    cidxs=[]
    for i,lval in enumerate(nidxs_lens):
        for _ in xrange(lval):
            cidxs.append(i)

    return np.asarray(cidxs,np.int32)


def point2ndixs(pts,radius):
    leafsize=10 if pts.shape[0]>100 else 1
    nidxs=libPointUtil.findNeighborRadiusCPU(pts,radius,leafsize)
    nidxs_lens=[len(nidx) for nidx in nidxs]
    nidxs_lens=np.asarray(nidxs_lens,dtype=np.int32)
    nidxs=np.concatenate(nidxs,axis=0)

    nidxs_bgs=compute_nidxs_bgs(nidxs_lens)
    cindxs=compute_cidxs(nidxs_lens)

    return nidxs,nidxs_lens,nidxs_bgs,cindxs


def append(lists,items):
    for l,i in zip(lists,items):
        l.append(i)


def normalize_block(xyzs,rgbs,covars,lbls,neighbor_radius=0.1,
                    resample=False,resample_low=0.8,resample_high=0.95,
                    jitter_color=False,jitter_val=2.5):
    # import time
    bn=len(xyzs)
    nidxs,nidxs_lens,nidxs_bgs,cidxs=[],[],[],[]
    # t1,t2,t3,t4,t5=0,0,0,0,0
    for bid in xrange(bn):
        # t=time.time()
        if resample:
            pt_num=len(xyzs[bid])
            random_down_ratio=np.random.uniform(resample_low,resample_high)
            idxs=np.random.choice(pt_num,int(pt_num*random_down_ratio))
            xyzs[bid]=xyzs[bid][idxs,:]
            rgbs[bid]=rgbs[bid][idxs,:]
            lbls[bid]=lbls[bid][idxs,:]
            covars[bid]=covars[bid][idxs,:]
        # t1+=time.time()-t

        # t=time.time()
        xyzs[bid]=np.ascontiguousarray(xyzs[bid],np.float32)
        nidx, nidxs_len, nidxs_bg, cidx=point2ndixs(xyzs[bid],neighbor_radius)
        # t2+=time.time()-t

        # t=time.time()
        append([nidxs,nidxs_lens,nidxs_bgs,cidxs],[nidx, nidxs_len, nidxs_bg, cidx])
        xyzs[bid]-=np.min(xyzs[bid],axis=0,keepdims=True)
        xyzs[bid][:,:2]-=1.5    # [-1.5,1.5]
        xyzs[bid][:,:2]/=1.5    # [-1,1]
        xyzs[bid][:,2]/=np.max(xyzs[bid][:,2],axis=0,keepdims=True)/2.0 # [0,2]
        xyzs[bid][:,2]-=1.0     # [-1,1]
        # t3+=time.time()-t

        # t=time.time()
        if jitter_color:
            rgbs[bid]+=np.random.uniform(-jitter_val,jitter_val,rgbs[bid].shape)

            rgbs[bid]-=128
            rgbs[bid]/=(128+jitter_val)
        else:
            rgbs[bid]-=128
            rgbs[bid]/=128
        # t4+=time.time()-t

        # t=time.time()
        mask=lbls[bid]>12
        if np.sum(mask)>0:
            lbls[bid][mask]=12
        lbls[bid]=lbls[bid].flatten()
        # t5+=time.time()-t

    # print 'cost {} {} {} {} {} s'.format(t1/bn,t2/bn,t3/bn,t4/bn,t5/bn)
    return xyzs,rgbs,covars,lbls,nidxs,nidxs_lens,nidxs_bgs,cidxs


def normalize_single_block(xyzs,rgbs,lbls,neighbor_radius=0.1,
                        resample=False,resample_low=0.8,resample_high=0.95,
                        jitter_color=False,jitter_val=2.5):

    if resample:
        pt_num=len(xyzs)
        random_down_ratio=np.random.uniform(resample_low,resample_high)
        idxs=np.random.choice(pt_num,int(pt_num*random_down_ratio))
        xyzs=xyzs[idxs,:]
        rgbs=rgbs[idxs,:]
        lbls=lbls[idxs,:]

    xyzs=np.ascontiguousarray(xyzs,np.float32)
    nidxs, nidxs_lens, nidxs_bgs, cidxs=point2ndixs(xyzs,neighbor_radius)
    covars=libPointUtil.computeCovarsGPU(xyzs, nidxs, nidxs_lens, nidxs_bgs)

    xyzs-=np.min(xyzs,axis=0,keepdims=True)
    xyzs[:,:2]-=1.5    # [-1.5,1.5]
    xyzs[:,:2]/=1.5    # [-1,1]
    xyzs[:,2]/=np.max(xyzs[:,2],axis=0,keepdims=True)/2.0 # [0,2]
    xyzs[:,2]-=1.0     # [-1,1]

    if jitter_color:
        rgbs+=np.random.uniform(-jitter_val,jitter_val,rgbs.shape)

    rgbs-=128
    rgbs/=128

    mask=lbls>12
    if np.sum(mask)>0:
        lbls[mask]=12
    lbls=lbls.flatten()

    return xyzs,rgbs,covars,lbls,nidxs,nidxs_lens,nidxs_bgs,cidxs