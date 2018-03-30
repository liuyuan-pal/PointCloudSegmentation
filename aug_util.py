import numpy as np
import libPointUtil
import random
import time
# from concurrent.futures import ThreadPoolExecutor
# import functools


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


def uniform_sample_block_gpu(xyz, block_size=3.0, stride=1.5, min_pn=2048):
    assert stride<=block_size

    # uniform sample
    max_xyz=np.max(xyz,axis=0,keepdims=True)
    maxx,maxy=max_xyz[0,0],max_xyz[0,1]
    beg_list=[]
    x_list=get_list_without_back_sample(maxx,block_size,stride)
    y_list=get_list_without_back_sample(maxy,block_size,stride)
    for x in x_list:
        for y in y_list:
            beg_list.append((x,y))

    beg_list=np.asarray(beg_list,dtype=np.float32)
    idxs=libPointUtil.gatherBlockPointsGPU(xyz,beg_list,block_size)
    idxs=[idx for idx in idxs if len(idx)>=min_pn]
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

    # _append
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
            subset.append(item[idx])

    return subsets


def sample_block(points, labels, ds_stride, block_size, block_stride, min_pn,
                 use_rescale=False, use_flip=False, use_rotate=False,
                 covar_ds_stride=0.03, covar_nn_size=0.1,gpu_gather=False):

    xyzs=np.ascontiguousarray(points[:,:3])
    rgbs=np.ascontiguousarray(points[:,3:])
    min_xyzs=np.min(xyzs,axis=0,keepdims=True)
    max_xyzs=np.max(xyzs,axis=0,keepdims=True)

    covar_ds_idxs=libPointUtil.gridDownsampleGPU(xyzs, covar_ds_stride, False)
    covar_ds_xyzs=np.ascontiguousarray(xyzs[covar_ds_idxs,:])

    # flip
    if use_flip:
        if random.random()<0.5:
            covar_ds_xyzs=swap_xy(covar_ds_xyzs)
            min_xyzs=swap_xy(min_xyzs)
            max_xyzs=swap_xy(max_xyzs)

        if random.random()<0.5:
            covar_ds_xyzs=flip(covar_ds_xyzs,axis=0)
            min_xyzs[:,0],max_xyzs[:,0]=-max_xyzs[:,0],-min_xyzs[:,0]

        if random.random()<0.5:
            covar_ds_xyzs=flip(covar_ds_xyzs,axis=1)
            min_xyzs[:,1],max_xyzs[:,1]=-max_xyzs[:,1],-min_xyzs[:,1]

    # rescale
    if use_rescale:
        rescale=np.random.uniform(0.9,1.1,[1,3])
        covar_ds_xyzs[:,:3]*=rescale
        min_xyzs*=rescale
        max_xyzs*=rescale

    # rotate
    if use_rotate:
        if random.random()>0.3:
            angle=random.random()*np.pi/2.0
            covar_ds_xyzs=rotate(covar_ds_xyzs,angle)

    ds_idxs=libPointUtil.gridDownsampleGPU(covar_ds_xyzs,ds_stride,False)

    covar_nidxs=libPointUtil.findNeighborRadiusCPU(covar_ds_xyzs,ds_idxs,covar_nn_size)

    covar_nidxs_lens=np.ascontiguousarray([len(idxs) for idxs in covar_nidxs],np.int32)
    covar_nidxs_bgs=compute_nidxs_bgs(covar_nidxs_lens)
    covar_nidxs=np.ascontiguousarray(np.concatenate(covar_nidxs,axis=0),dtype=np.int32)

    covars=libPointUtil.computeCovarsGPU(covar_ds_xyzs,covar_nidxs,covar_nidxs_lens,covar_nidxs_bgs)

    xyzs=covar_ds_xyzs[ds_idxs,:]
    rgbs=rgbs[covar_ds_idxs,:][ds_idxs,:]
    lbls=labels[covar_ds_idxs][ds_idxs]

    if gpu_gather:
        xyzs-=min_xyzs
        idxs=uniform_sample_block_gpu(xyzs,block_size,block_stride,min_pn=min_pn)
        xyzs+=min_xyzs
    else:
        xyzs-=min_xyzs
        idxs = uniform_sample_block(xyzs,block_size,block_stride,normalized=True,min_pn=min_pn)
        xyzs+=min_xyzs

    xyzs, rgbs, covars, lbls=fetch_subset([xyzs,rgbs,covars,lbls],idxs)

    return xyzs, rgbs, covars, lbls


def sample_block_v2(points, labels, ds_stride, block_size, block_stride, min_pn,
                 use_rescale=False, use_flip=False, use_rotate=False,
                 covar_nn_size=0.1):
    xyzs=np.ascontiguousarray(points[:,:3])
    rgbs=np.ascontiguousarray(points[:,3:])
    min_xyzs=np.min(xyzs,axis=0,keepdims=True)
    max_xyzs=np.max(xyzs,axis=0,keepdims=True)

    # flip
    if use_flip:
        if random.random()<0.5:
            xyzs=swap_xy(xyzs)
            min_xyzs=swap_xy(min_xyzs)
            max_xyzs=swap_xy(max_xyzs)

        if random.random()<0.5:
            xyzs=flip(xyzs,axis=0)
            min_xyzs[:,0],max_xyzs[:,0]=-max_xyzs[:,0],-min_xyzs[:,0]

        if random.random()<0.5:
            xyzs=flip(xyzs,axis=1)
            min_xyzs[:,1],max_xyzs[:,1]=-max_xyzs[:,1],-min_xyzs[:,1]

    # rescale
    if use_rescale:
        rescale=np.random.uniform(0.9,1.1,[1,3])
        xyzs[:,:3]*=rescale
        min_xyzs*=rescale
        max_xyzs*=rescale

    # rotate
    if use_rotate:
        if random.random()>0.3:
            angle=random.random()*np.pi/2.0
            xyzs=rotate(xyzs,angle)

    ds_idxs=libPointUtil.gridDownsampleGPU(xyzs,ds_stride,False)

    covar_nidxs=libPointUtil.findNeighborRadiusGPU(xyzs,ds_idxs,covar_nn_size)

    covar_nidxs_lens=np.ascontiguousarray([len(idxs) for idxs in covar_nidxs],np.int32)
    covar_nidxs_bgs=compute_nidxs_bgs(covar_nidxs_lens)
    covar_nidxs=np.ascontiguousarray(np.concatenate(covar_nidxs,axis=0),dtype=np.int32)

    covars=libPointUtil.computeCovarsGPU(xyzs,covar_nidxs,covar_nidxs_lens,covar_nidxs_bgs)

    xyzs=xyzs[ds_idxs,:]
    rgbs=rgbs[ds_idxs,:]
    lbls=labels[ds_idxs]

    xyzs-=min_xyzs
    idxs = uniform_sample_block(xyzs,block_size,block_stride,normalized=True,min_pn=min_pn)
    xyzs+=min_xyzs
    xyzs, rgbs, covars, lbls=fetch_subset([xyzs,rgbs,covars,lbls],idxs)

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
    if pts.shape[0]>8000:
        nidxs = libPointUtil.findNeighborRadiusGPU(pts, radius, leafsize)
    else:
        nidxs=libPointUtil.findNeighborRadiusCPU(pts, radius, leafsize)
    nidxs_lens=[len(nidx) for nidx in nidxs]
    nidxs_lens=np.asarray(nidxs_lens,dtype=np.int32)
    nidxs=np.concatenate(nidxs,axis=0)

    nidxs_bgs=compute_nidxs_bgs(nidxs_lens)
    cindxs=compute_cidxs(nidxs_lens)

    return nidxs,nidxs_lens,nidxs_bgs,cindxs


def _append(lists, items):
    for l,i in zip(lists,items):
        l.append(i)


def normalize_block(xyzs,rgbs,covars,lbls,neighbor_radius=0.1,
                    resample=False,resample_low=0.8,resample_high=0.95,
                    jitter_color=False,jitter_val=2.5):
    bn=len(xyzs)
    nidxs,nidxs_lens,nidxs_bgs,cidxs=[],[],[],[]
    block_bgs,block_lens=[],[]
    for bid in xrange(bn):
        if resample:
            pt_num=len(xyzs[bid])
            random_down_ratio=np.random.uniform(resample_low,resample_high)
            idxs=np.random.choice(pt_num,int(pt_num*random_down_ratio))
            xyzs[bid]=xyzs[bid][idxs,:]
            rgbs[bid]=rgbs[bid][idxs,:]
            lbls[bid]=lbls[bid][idxs,:]
            covars[bid]=covars[bid][idxs,:]

        xyzs[bid]=np.ascontiguousarray(xyzs[bid],np.float32)
        nidx, nidxs_len, nidxs_bg, cidx=point2ndixs(xyzs[bid],neighbor_radius)

        _append([nidxs, nidxs_lens, nidxs_bgs, cidxs], [nidx, nidxs_len, nidxs_bg, cidx])

        block_bgs.append(np.min(xyzs[bid],axis=0))
        xyzs[bid]-=np.min(xyzs[bid],axis=0,keepdims=True)
        block_lens.append(np.max(xyzs[bid],axis=0))
        xyzs[bid][:,:2]-=1.5    # [-1.5,1.5]
        xyzs[bid][:,:2]/=1.5    # [-1,1]
        xyzs[bid][:,2]/=np.max(xyzs[bid][:,2],axis=0,keepdims=True)/2.0 # [0,2]
        xyzs[bid][:,2]-=1.0     # [-1,1]

        if jitter_color:
            rgbs[bid]+=np.random.uniform(-jitter_val,jitter_val,rgbs[bid].shape)

            rgbs[bid]-=128
            rgbs[bid]/=(128+jitter_val)
        else:
            rgbs[bid]-=128
            rgbs[bid]/=128

        mask=lbls[bid]>12
        if np.sum(mask)>0:
            lbls[bid][mask]=12
        lbls[bid]=lbls[bid].flatten()

    return xyzs,rgbs,covars,lbls,nidxs,nidxs_lens,nidxs_bgs,cidxs,block_bgs,block_lens


def _permutation(feats_list, idxs):
    for i in xrange(len(feats_list)):
        feats_list[i]=feats_list[i][idxs]
    return feats_list


def build_hierarchy(xyzs,feats_list,vs1,vs2):
    ###########################
    cxyz1=np.ascontiguousarray(xyzs)
    sidxs1,vlens1=libPointUtil.sortVoxelGPU(cxyz1,vs1)

    cxyz1=cxyz1[sidxs1,:]
    cxyz1=np.ascontiguousarray(cxyz1)
    dxyz1,cxyz2=libPointUtil.computeCenterDiffCPU(cxyz1,vlens1)

    feats_list=_permutation(feats_list, sidxs1)
    ############################
    cxyz2=np.ascontiguousarray(cxyz2)
    sidxs2,vlens2=libPointUtil.sortVoxelGPU(cxyz2,vs2)

    cxyz2=cxyz2[sidxs2,:]
    cxyz2=np.ascontiguousarray(cxyz2)
    dxyz2,cxyz3=libPointUtil.computeCenterDiffCPU(cxyz2,vlens2)

    sidxs1,vlens1=libPointUtil.adjustPointsMemoryCPU(vlens1,sidxs2,cxyz1.shape[0])
    dxyz1,cxyz1=_permutation([dxyz1, cxyz1], sidxs1)
    feats_list=_permutation(feats_list, sidxs1)

    return cxyz1,dxyz1,vlens1,cxyz2,dxyz2,vlens2,cxyz3,feats_list


def normalize_block_hierarchy(xyzs,rgbs,covars,lbls,bsize=3.0,
                              nr1=0.1,nr2=0.3,nr3=1.0,vc1=0.15,vc2=0.5,
                              resample=False,resample_low=0.8,resample_high=0.95,
                              jitter_color=False,jitter_val=2.5,max_pt_num=10240):
    bn=len(xyzs)
    cxyzs,dxyzs,vlens,vlens_bgs,vcidxs,\
    cidxs,nidxs,nidxs_bgs,nidxs_lens=[],[],[],[],[],[],[],[],[]
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
        min_xyz[:,:2]+=bsize/2.0
        xyzs[bid]-=min_xyz
        #xyzs[bid][:,:2]-=1.5    # [-1.5,1.5]
        block_mins.append(min_xyz)

        cxyz1, dxyz1, vlens1, cxyz2, dxyz2, vlens2, cxyz3, feats_list=\
            build_hierarchy(xyzs[bid],[rgbs[bid],lbls[bid],covars[bid]],vc1,vc2)
        rgbs[bid],lbls[bid],covars[bid]=feats_list

        # rescale to unit cube
        dxyz1/=vc1
        dxyz2/=vc2

        # bg=time.time()
        vlens_bgs1=compute_nidxs_bgs(vlens1)
        vlens_bgs2=compute_nidxs_bgs(vlens2)
        vcidxs1=compute_cidxs(vlens1)
        vcidxs2=compute_cidxs(vlens2)

        nidxs1,nidxs_lens1,nidxs_bgs1,cidxs1=point2ndixs(cxyz1,nr1)
        nidxs2,nidxs_lens2,nidxs_bgs2,cidxs2=point2ndixs(cxyz2,nr2)
        nidxs3,nidxs_lens3,nidxs_bgs3,cidxs3=point2ndixs(cxyz3,nr3)
        # t+=time.time()-bg

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

        mask=lbls[bid]>12
        if np.sum(mask)>0:
            lbls[bid][mask]=12
        lbls[bid]=lbls[bid].flatten()

        _append([cxyzs, dxyzs, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens],
                [[cxyz1,cxyz2,cxyz3],[dxyz1,dxyz2],[vlens1,vlens2],[vlens_bgs1,vlens_bgs2],[vcidxs1,vcidxs2],
                [cidxs1,cidxs2,cidxs3],[nidxs1,nidxs2,nidxs3],[nidxs_bgs1,nidxs_bgs2,nidxs_bgs3],
                [nidxs_lens1,nidxs_lens2,nidxs_lens3]])

    # print 'neighbor cost {} s'.format(t)

    return cxyzs,dxyzs,rgbs,covars,lbls,vlens,vlens_bgs,vcidxs,cidxs,nidxs,nidxs_bgs,nidxs_lens,block_mins


def normalize_model_hierarchy(xyzs,use_rotate=False,nr1=0.05,nr2=0.1,nr3=0.3,vc1=0.1,vc2=0.3):
    #xyzs [b,1024,3]
    n=xyzs.shape[0]
    cxyzs,dxyzs,vlens,vlens_bgs,vcidxs,\
    cidxs,nidxs,nidxs_bgs,nidxs_lens=[],[],[],[],[],[],[],[],[]
    covars=[]
    for i in xrange(n):
        # bg=time.time()
        if use_rotate:
            ang=np.random.uniform(0,np.pi*2)
            xyzs[i]=rotate(xyzs[i],ang)

        nidxs1=libPointUtil.findNeighborRadiusCPU(xyzs[i],nr1,15)

        nidxs_lens1=np.ascontiguousarray([len(idxs) for idxs in nidxs1],np.int32)
        nidxs_bgs1=compute_nidxs_bgs(nidxs_lens1)
        nidxs1=np.ascontiguousarray(np.concatenate(nidxs1,axis=0),dtype=np.int32)

        covar=libPointUtil.computeCovarsGPU(xyzs[i],nidxs1,nidxs_lens1,nidxs_bgs1)

        cxyz1, dxyz1, vlens1, cxyz2, dxyz2, vlens2, cxyz3, feats_list=\
            build_hierarchy(xyzs[i],[covar],vc1,vc2)
        covar=feats_list[0]
        covars.append(covar)


        # rescale to unit cube
        dxyz1/=vc1
        dxyz2/=vc2

        # bg=time.time()
        vlens_bgs1=compute_nidxs_bgs(vlens1)
        vlens_bgs2=compute_nidxs_bgs(vlens2)
        vcidxs1=compute_cidxs(vlens1)
        vcidxs2=compute_cidxs(vlens2)

        nidxs1,nidxs_lens1,nidxs_bgs1,cidxs1=point2ndixs(cxyz1,nr1)
        nidxs2,nidxs_lens2,nidxs_bgs2,cidxs2=point2ndixs(cxyz2,nr2)
        nidxs3,nidxs_lens3,nidxs_bgs3,cidxs3=point2ndixs(cxyz3,nr3)


        _append([cxyzs, dxyzs, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens],
                [[cxyz1,cxyz2,cxyz3],[dxyz1,dxyz2],[vlens1,vlens2],[vlens_bgs1,vlens_bgs2],[vcidxs1,vcidxs2],
                [cidxs1,cidxs2,cidxs3],[nidxs1,nidxs2,nidxs3],[nidxs_bgs1,nidxs_bgs2,nidxs_bgs3],
                [nidxs_lens1,nidxs_lens2,nidxs_lens3]])
        # print '{} cost {} s'.format(i,time.time()-bg)

    return cxyzs, dxyzs, covars, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens



def sample_block_scannet(points, labels, ds_stride, block_size, block_stride, min_pn,
                         use_rescale=False, use_flip=False, use_rotate=False,
                         covar_nn_size=0.1):
    xyzs=np.ascontiguousarray(points[:,:3])
    min_xyzs=np.min(xyzs,axis=0,keepdims=True)
    max_xyzs=np.max(xyzs,axis=0,keepdims=True)

    # flip
    if use_flip:
        if random.random()<0.5:
            xyzs=swap_xy(xyzs)
            min_xyzs=swap_xy(min_xyzs)
            max_xyzs=swap_xy(max_xyzs)

        if random.random()<0.5:
            xyzs=flip(xyzs,axis=0)
            min_xyzs[:,0],max_xyzs[:,0]=-max_xyzs[:,0],-min_xyzs[:,0]

        if random.random()<0.5:
            xyzs=flip(xyzs,axis=1)
            min_xyzs[:,1],max_xyzs[:,1]=-max_xyzs[:,1],-min_xyzs[:,1]

    # rescale
    if use_rescale:
        rescale=np.random.uniform(0.9,1.1,[1,3])
        xyzs[:,:3]*=rescale
        min_xyzs*=rescale
        max_xyzs*=rescale

    # rotate
    if use_rotate:
        if random.random()>0.3:
            angle=random.random()*np.pi/2.0
            xyzs=rotate(xyzs,angle)
            min_xyzs=np.min(xyzs,axis=0,keepdims=True)
            max_xyzs=np.max(xyzs,axis=0,keepdims=True)

    ds_idxs=libPointUtil.gridDownsampleGPU(xyzs,ds_stride,False)

    covar_nidxs=libPointUtil.findNeighborRadiusGPU(xyzs,ds_idxs,covar_nn_size)

    covar_nidxs_lens=np.ascontiguousarray([len(idxs) for idxs in covar_nidxs],np.int32)
    covar_nidxs_bgs=compute_nidxs_bgs(covar_nidxs_lens)
    covar_nidxs=np.ascontiguousarray(np.concatenate(covar_nidxs,axis=0),dtype=np.int32)

    covars=libPointUtil.computeCovarsGPU(xyzs,covar_nidxs,covar_nidxs_lens,covar_nidxs_bgs)

    xyzs=xyzs[ds_idxs,:]
    lbls=labels[ds_idxs]

    xyzs-=min_xyzs
    idxs = uniform_sample_block(xyzs,block_size,block_stride,normalized=True,min_pn=min_pn)
    xyzs+=min_xyzs
    xyzs, covars, lbls=fetch_subset([xyzs,covars,lbls],idxs)

    return xyzs, covars, lbls



def normalize_block_scannet(xyzs,covars,lbls,bsize=3.0,
                            nr1=0.1,nr2=0.3,nr3=1.0,vc1=0.15,vc2=0.5,
                            resample=False,resample_low=0.8,resample_high=0.95,max_pt_num=10240):
    bn=len(xyzs)
    cxyzs,dxyzs,vlens,vlens_bgs,vcidxs,\
    cidxs,nidxs,nidxs_bgs,nidxs_lens=[],[],[],[],[],[],[],[],[]
    block_mins=[]

    # t=0
    for bid in xrange(bn):
        if resample:
            pt_num=len(xyzs[bid])
            random_down_ratio=np.random.uniform(resample_low,resample_high)
            idxs=np.random.choice(pt_num,int(pt_num*random_down_ratio))
            xyzs[bid]=xyzs[bid][idxs,:]
            lbls[bid]=lbls[bid][idxs]
            covars[bid]=covars[bid][idxs,:]

        if len(xyzs[bid])>max_pt_num:
            pt_num=len(xyzs[bid])
            ratio=max_pt_num/float(len(xyzs[bid]))
            idxs=np.random.choice(pt_num,int(pt_num*ratio))
            xyzs[bid]=xyzs[bid][idxs,:]
            lbls[bid]=lbls[bid][idxs]
            covars[bid]=covars[bid][idxs,:]

        # offset center to zero
        # !!! dont rescale here since it will affect the neighborhood size !!!
        min_xyz=np.min(xyzs[bid],axis=0,keepdims=True)
        min_xyz[:,:2]+=bsize/2.0
        xyzs[bid]-=min_xyz
        #xyzs[bid][:,:2]-=1.5    # [-1.5,1.5]
        block_mins.append(min_xyz)

        cxyz1, dxyz1, vlens1, cxyz2, dxyz2, vlens2, cxyz3, feats_list=\
            build_hierarchy(xyzs[bid],[lbls[bid],covars[bid]],vc1,vc2)
        lbls[bid],covars[bid]=feats_list

        # rescale to unit cube
        dxyz1/=vc1
        dxyz2/=vc2

        # bg=time.time()
        vlens_bgs1=compute_nidxs_bgs(vlens1)
        vlens_bgs2=compute_nidxs_bgs(vlens2)
        vcidxs1=compute_cidxs(vlens1)
        vcidxs2=compute_cidxs(vlens2)

        nidxs1,nidxs_lens1,nidxs_bgs1,cidxs1=point2ndixs(cxyz1,nr1)
        nidxs2,nidxs_lens2,nidxs_bgs2,cidxs2=point2ndixs(cxyz2,nr2)
        nidxs3,nidxs_lens3,nidxs_bgs3,cidxs3=point2ndixs(cxyz3,nr3)
        # t+=time.time()-bg

        _append([cxyzs, dxyzs, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens],
                [[cxyz1,cxyz2,cxyz3],[dxyz1,dxyz2],[vlens1,vlens2],[vlens_bgs1,vlens_bgs2],[vcidxs1,vcidxs2],
                [cidxs1,cidxs2,cidxs3],[nidxs1,nidxs2,nidxs3],[nidxs_bgs1,nidxs_bgs2,nidxs_bgs3],
                [nidxs_lens1,nidxs_lens2,nidxs_lens3]])

    # print 'neighbor cost {} s'.format(t)

    return cxyzs,dxyzs,covars,lbls,vlens,vlens_bgs,vcidxs,cidxs,nidxs,nidxs_bgs,nidxs_lens,block_mins