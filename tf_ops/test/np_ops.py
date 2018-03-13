import numpy as np
from data_util import *
from functools import partial

def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def np_pool_forward(ifeats,vlens,vlens_bgs):
    pn1,fd=ifeats.shape
    pn2=vlens.shape[0]

    pfeats=np.empty([pn2,fd],np.float64)
    max_idxs=np.empty([pn2,fd],np.int)
    for i in xrange(pn2):
        cfeats=ifeats[vlens_bgs[i]:vlens_bgs[i]+vlens[i]]
        pfeats[i]=np.max(cfeats,axis=0)
        max_idxs[i]=np.argmax(cfeats,axis=0)

    return pfeats


def np_pool_backward(dpfeats,pfeats,ifeats,vlens,vlens_bgs):
    pn1,fd=ifeats.shape
    pn2=vlens.shape[0]

    difeats=np.empty([pn1,fd],np.float64)
    for i in xrange(pn2):
        cfeats=ifeats[vlens_bgs[i]:vlens_bgs[i]+vlens[i]]
        # cdfeats=difeats[vlens_bgs[i]:vlens_bgs[i]+vlens[i]]
        cdfeats=np.zeros([vlens[i],fd],np.float64)
        max_idxs=np.argmax(cfeats,axis=0)
        cdfeats[max_idxs,np.arange(max_idxs.shape[0])]=np.copy(dpfeats[i])
        difeats[vlens_bgs[i]:vlens_bgs[i] + vlens[i]]=cdfeats

    return difeats


def np_unpool_forward(ifeats,vlens,vlens_bgs,cidxs):
    pn2,fd=ifeats.shape
    pn1=np.sum(vlens)

    upfeats=np.empty([pn1,fd],np.float64)
    for i in xrange(pn2):
        upfeats[vlens_bgs[i]:vlens_bgs[i]+vlens[i],:]=ifeats[i,:]

    return upfeats


def np_unpool_backward(dupfeats,upfeats,ifeats,vlens,vlens_bgs,cidxs):
    pn2,fd=ifeats.shape
    pn1=np.sum(vlens)

    difeats=np.empty([pn2,fd],np.float64)
    for i in xrange(pn2):
        difeats[i,:]=np.sum(dupfeats[vlens_bgs[i]:vlens_bgs[i]+vlens[i],:],axis=0)

    return difeats


def np_concat_non_center_forward(ifeats,nidxs,nidxs_lens,nidxs_bgs):
    pn,ifn=ifeats.shape
    csum=nidxs.shape[0]

    ofeats=np.empty([csum-pn,ifn*2],np.float64)
    for i in xrange(pn):
        nn_bg=nidxs_bgs[i]
        nn_size=nidxs_lens[i]
        t=nn_bg-i
        for j in xrange(nn_size):
            idx=nidxs[nn_bg+j]
            if idx==i: continue
            ofeats[t,:ifn]=ifeats[i,:]
            ofeats[t,ifn:]=ifeats[idx,:]
            t+=1

    return ofeats


def np_concat_non_center_backward(sfeats,nidxs,nidxs_lens,nidxs_bgs):
    ifn=sfeats.shape[1]/2
    pn=nidxs_bgs.shape[0]

    ifeats=np.zeros([pn,ifn],np.float64)
    for i in xrange(pn):
        nn_bg=nidxs_bgs[i]
        nn_size=nidxs_lens[i]
        t=nn_bg-i
        for j in xrange(nn_size):
            idx=nidxs[nn_bg+j]
            if idx==i: continue
            ifeats[i,:]+=sfeats[t,:ifn]
            ifeats[idx,:]+=sfeats[t,ifn:]
            t+=1

    return ifeats



def test_np_pool():
    vlens=np.random.randint(3,5,[30])
    vlens_bgs=compute_nidxs_bgs(vlens)

    pn1=vlens_bgs[-1]+vlens[-1]
    ifeats=np.random.uniform(-1.0,1.0,[pn1,16])
    pfeats=np_pool_forward(ifeats,vlens,vlens_bgs)
    dpfeats=np.random.uniform(-1.0,1.0,pfeats.shape)
    difeats=np_pool_backward(dpfeats,pfeats,ifeats,vlens,vlens_bgs)

    f=lambda ifeats: np_pool_forward(ifeats,vlens,vlens_bgs)
    difeats_num=eval_numerical_gradient_array(f,ifeats,dpfeats)

    print vlens
    print ifeats
    print pfeats

    print 'diff {} '.format(np.max(difeats-difeats_num))


def test_np_unpool():
    vlens=np.random.randint(3,5,[5])
    vlens_bgs=compute_nidxs_bgs(vlens)
    cidxs=compute_cidxs(vlens)

    ifeats=np.random.uniform(-1.0,1.0,[5,4])
    upfeats=np_unpool_forward(ifeats,vlens,vlens_bgs,cidxs)
    dupfeats=np.random.uniform(-1.0,1.0,upfeats.shape)
    difeats=np_unpool_backward(dupfeats,upfeats,ifeats,vlens,vlens_bgs,cidxs)

    f=lambda ifeats: np_unpool_forward(ifeats,vlens,vlens_bgs,cidxs)
    difeats_num=eval_numerical_gradient_array(f,ifeats,dupfeats)

    print vlens
    print ifeats
    print upfeats
    print '//////////'
    print dupfeats
    print difeats


    print 'diff {} '.format(np.max(difeats-difeats_num))


def test_concat_non_center():
    import libPointUtil
    pts=np.random.uniform(-1.0,1.0,[20,3])
    pts=np.asarray(pts,dtype=np.float32)
    nidxs=libPointUtil.findNeighborRadiusGPU(pts,1.0)
    nidxs_lens=np.asarray([len(idxs) for idxs in nidxs],dtype=np.int32)
    nidxs_bgs=compute_nidxs_bgs(nidxs_lens)
    nidxs=np.concatenate(nidxs,axis=0)
    nidxs=np.asarray(nidxs,dtype=np.int32)

    pts=np.asarray(pts,np.float64)
    ofeats=np_concat_non_center_forward(pts,nidxs,nidxs_lens,nidxs_bgs)
    dofeats=np.random.uniform(-1,1,ofeats.shape)
    difeats=np_concat_non_center_backward(dofeats,nidxs,nidxs_lens,nidxs_bgs)

    f=lambda ifeats: np_concat_non_center_forward(ifeats,nidxs,nidxs_lens,nidxs_bgs)
    difeats_num=eval_numerical_gradient_array(f,pts,dofeats)

    print pts
    print ofeats

    print 'diff {} '.format(np.max(difeats-difeats_num))


if __name__=="__main__":
    test_concat_non_center()