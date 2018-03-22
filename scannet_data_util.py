import cPickle
from draw_util import output_points
import numpy as np
from aug_util import sample_block_scannet,normalize_block_scannet
from io_util import save_pkl,read_pkl
import time

def overview():
    with open('/home/pal/data/ScanNet/data/scannet_test.pickle','rb') as f:
        points=cPickle.load(f)
        labels=cPickle.load(f)

    max_label=0
    min_label=100
    for i in xrange(len(labels)):
        max_label=max(np.max(labels[i]),max_label)
        min_label=min(np.min(labels[i]),min_label)

    print max_label
    print min_label


def compute_weights(labels):
    labelweights = np.zeros(21)
    for seg in labels:
        tmp, _ = np.histogram(seg, range(22))
        labelweights += tmp
    labelweights = labelweights.astype(np.float32)
    labelweights = labelweights / np.sum(labelweights)
    labelweights = 1 / np.log(1.2 + labelweights)
    return labelweights

def test_train_block():
    with open('data/ScanNet/scannet_train.pickle','rb') as f:
        points=cPickle.load(f)
        labels=cPickle.load(f)

    room_num=len(points)
    for i in xrange(3):
        bg=time.time()
        xyzs, covars, lbls=sample_block_scannet(points[i],labels[i],0.02,3.0,1.5,128,True,True,True,0.1)
        cxyzs, dxyzs, covars, lbls, vlens, vlens_bgs, vcidxs, cidxs, nidxs, nidxs_bgs, nidxs_lens, block_mins=\
            normalize_block_scannet(xyzs,covars,lbls,3.0,0.1,0.3,1.0,0.15,0.5,True,0.8,1.0,10240)
        print 'single cost {} s'.format(time.time()-bg)

        colors=np.random.randint(0,256,[21,3])
        for t in xrange(len(cxyzs)):
            output_points('test_result/{}_{}.txt'.format(i,t),cxyzs[t][0]+block_mins[t],colors[lbls[t]])
            print 'point num {} {} {}'.format(cxyzs[t][0].shape[0],cxyzs[t][1].shape[0],cxyzs[t][2].shape[0])
            print 'mean neighbor size {} {} {}'.format(
                float(nidxs[t][0].shape[0])/cxyzs[t][0].shape[0],
                float(nidxs[t][1].shape[0])/cxyzs[t][1].shape[0],
                float(nidxs[t][2].shape[0])/cxyzs[t][2].shape[0]
            )

def process_one_file(fid):
    points,labels=read_pkl('data/ScanNet/train_split_{}.pkl'.format(fid))

    room_num=len(points)
    all_data=[[] for _ in xrange(12)]
    idx=0
    bg=time.time()
    for i in xrange(room_num):
        if i%10==0:
            print 'idx {} cost {} s'.format(i,time.time()-bg)
            bg=time.time()

        for t in xrange(10):
            xyzs, covars, lbls=sample_block_scannet(points[i],labels[i],0.02,3.0,1.5,128,True,True,True,0.1)
            data = normalize_block_scannet(xyzs,covars,lbls,3.0,0.1,0.3,1.0,0.15,0.5,True,0.8,1.0,10240)

            for s in xrange(len(data)):
                all_data[s]+=data[s]

        if len(all_data[0])>300:
            save_pkl('data/ScanNet/train_{}_{}.pkl'.format(fid,idx),all_data)
            idx+=1
            all_data=[[] for _ in xrange(12)]

    if len(all_data[0])>0:
        save_pkl('data/ScanNet/train_{}_{}.pkl'.format(fid,idx),all_data)
        idx+=1


def prepare_train_block():
    from concurrent.futures import ProcessPoolExecutor
    executor=ProcessPoolExecutor(6)
    futures=[]
    for i in xrange(6):
        futures.append(executor.submit(process_one_file,i))

    for f in futures:
        f.result()

def split_train_data(split_size):
    with open('data/ScanNet/scannet_train.pickle','rb') as f:
        points=cPickle.load(f)
        labels=cPickle.load(f)

    cur_size=0
    idx=0
    print 'total size {}'.format(len(points))
    while cur_size<len(points):
        save_pkl('data/ScanNet/train_split_{}.pkl'.format(idx),[points[cur_size:cur_size+split_size],labels[cur_size:cur_size+split_size]])
        idx+=1
        cur_size+=split_size
        print 'cur size {}'.format(cur_size)

def process_test_data():
    with open('data/ScanNet/scannet_test.pickle','rb') as f:
        points=cPickle.load(f)
        labels=cPickle.load(f)

    room_num=len(points)
    bg=time.time()
    for i in xrange(room_num):
        if i%10==0:
            print 'idx {} cost {} s'.format(i,time.time()-bg)
            bg=time.time()

        xyzs, covars, lbls=sample_block_scannet(points[i],labels[i],0.02,3.0,1.5,64,False,False,False,0.1)
        data = normalize_block_scannet(xyzs,covars,lbls,3.0,0.1,0.3,1.0,0.15,0.5,False,0.8,1.0,10240)

        save_pkl('data/ScanNet/sampled_test/test_{}.pkl'.format(i),data)

if __name__=="__main__":
    import os
    with open('cached/scannet_train_filenames.txt','w') as f:
        for fn in os.listdir('data/ScanNet/sampled_train'):
            f.write('{}\n'.format(fn))
