from semantic3d_util import *
import libPointUtil

sample_stride=0.1
block_size=10.0
block_stride=5.0
min_point_num=256
radius=0.25

def normalize(xyzs,rgbs,covars,lbls,offset_z,bsize=3.0):
    bn=len(xyzs)
    block_mins=[]
    block_nidxs=[]
    block_nlens=[]
    block_nbegs=[]

    # t=0
    for bid in xrange(bn):
        min_xyz=np.min(xyzs[bid],axis=0,keepdims=True)
        min_xyz[:,2]=offset_z
        min_xyz[:,:2]+=bsize/2.0
        xyzs[bid]-=min_xyz
        block_mins.append(min_xyz)

        rgbs[bid][:,:3]-=128
        rgbs[bid][:,:3]/=128

        # sub mean div std
        if rgbs[bid].shape[1]>3:
            rgbs[bid][:,3]-=-1164.05
            rgbs[bid][:,3]/=600.0

        nidxs=libPointUtil.findNeighborRadiusCPU(xyzs[bid],radius)
        nlens=np.asarray([len(idxs) for idxs in nidxs],dtype=np.int32)
        nbegs=compute_nidxs_bgs(nlens)

        nidxs=np.concatenate(nidxs,axis=0)

        block_nidxs.append(nidxs)
        block_nlens.append(nlens)
        block_nbegs.append(nbegs)

    return xyzs, rgbs, covars, lbls, block_mins, block_nidxs, block_nlens, block_nbegs

def train_block(fs,offset_z):
    points,labels=read_pkl(fs)
    xyzs, rgbs, covars, lbls=sample_block(
        points,labels,sample_stride,block_size,block_stride,min_point_num,False,False,False,0.05,0.2,True)

    xyzs, rgbs, covars, lbls, block_mins, block_nidxs, block_nlens, block_nbegs=normalize(xyzs,rgbs,covars,lbls,offset_z,block_size)

    return xyzs, rgbs, lbls, block_mins, block_nidxs, block_nlens, block_nbegs

def one_train_file(tfs):
    stem_offset_map=semantic3d_read_map_offset_z()
    stem='_'.join(tfs.split('_')[:-2])
    offset_z=stem_offset_map[stem]
    fs='data/Semantic3D.Net/block/train/'+tfs
    data=train_block(fs,offset_z)

    save_pkl('data/Semantic3D.Net/block/sampled_dense/'+tfs,data)
    print '{} done'.format(tfs)

# step 3 process block to block/sampled
def all_train_file():
    executor=ProcessPoolExecutor(8)
    train_list=semantic3d_read_train_block_list()
    futures=[executor.submit(one_train_file,tfs) for tfs in train_list]
    for f in futures: f.result()

from semantic3d_context_util import  testset_get_context_offset_z

def test_block(fs,offset_z):
    points,labels=read_pkl(fs)
    xyzs, rgbs, covars, lbls=sample_block(
        points,labels,sample_stride,block_size,block_stride,64,False,False,False,0.05,0.2,True)

    xyzs, rgbs, covars, lbls, block_mins, block_nidxs, block_nlens, block_nbegs=normalize(xyzs,rgbs,covars,lbls,offset_z,block_size)

    return xyzs, rgbs, lbls, block_mins, block_nidxs, block_nlens, block_nbegs

def testset_process_context_block(fs,fni):
    stem2offset=testset_get_context_offset_z()
    offset_z=stem2offset[fs]
    data = test_block('data/Semantic3D.Net/pkl/test_presample/{}_{}.pkl'.format(fs,fni),offset_z)
    return data

def all_test_file():
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

        save_pkl('data/Semantic3D.Net/block/test_dense/{}.pkl'.format(fs),all_data)
        print '{} done'.format(fs)


if __name__=="__main__":
    # all_train_file()
    all_test_file()
    # test_set=['sg27_station4_intensity_rgb','bildstein_station1_xyz_intensity_rgb']
    # train_list,test_list=get_semantic3d_block_train_test_list(test_set)
    # train_list=['/data/Semantic3D.Net/'+fn for fn in train_list]
    # test_list=['/data/Semantic3D.Net/'+fn for fn in test_list]
    #
    # counts=np.zeros(9)
    # block_num=0
    # begin=time.time()
    # for tfs in train_list:
    #     data=read_pkl(tfs)
    #     block_num+=len(data[0])
    #     if len(data[2])==1:
    #         labels=data[2][0]
    #     elif len(data[0])==0:
    #         continue
    #     else:
    #         labels=np.concatenate(data[2],axis=0)
    #     count,_=np.histogram(labels,np.arange(10))
    #     counts+=count
    #
    # print 'train {} cost {}s'.format(block_num,time.time()-begin)
    # print counts
    #
    # block_num=0
    # counts=np.zeros(9)
    # begin=time.time()
    # for tfs in test_list:
    #     data=read_pkl(tfs)
    #     block_num+=len(data[0])
    #     if len(data[2])==1:
    #         labels=data[2][0]
    #     elif len(data[0])==0:
    #         continue
    #     else:
    #         labels=np.concatenate(data[2],axis=0)
    #     count,_=np.histogram(labels,np.arange(10))
    #     counts+=count
    #
    # print 'test {} cost {}s'.format(block_num,time.time()-begin)
    # print counts


