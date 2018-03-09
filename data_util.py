# from aug_util import *
# from io_util import get_block_train_test_split,read_room_pkl,save_room_pkl,get_block_train_test_split_ds

# def downsample(points,labels,stride):
#     idxs=libPointUtil.gridDownsampleGPU(points, stride, False)
#     points=points[idxs,:]
#     labels=labels[idxs,:]
#
#     return points,labels
#
# def downsample_and_save():
#     train_list,test_list=get_block_train_test_split()
#     idx=0
#     train_list+=test_list
#     with open('cached/room_block_ds0.03_stems.txt','w') as f:
#         for _ in xrange(5):
#             for fs in train_list:
#                 points,labels=read_room_pkl('data/S3DIS/room_block_10_10/'+fs)
#                 points, labels=downsample(points,labels,0.03)
#
#                 names=fs.split('_')
#                 names[0]=str(idx)
#                 nfs='_'.join(names)
#                 f.write(nfs+'\n')
#                 ofs='data/S3DIS/room_block_10_10_ds0.03/'+nfs
#                 save_room_pkl(ofs,points,labels)
#                 idx+=1
#
#
# def get_class_num():
#     train_list,test_list=get_block_train_test_split_ds()
#
#     counts=np.zeros(13)
#     for fs in test_list:
#         points, labels = read_room_pkl('data/S3DIS/room_block_10_10_ds0.03/'+fs)
#         for i in xrange(13):
#             counts[i]+=np.sum(labels==i)
#
#     for i in xrange(13):
#         print counts[i]/counts[0]

import cPickle
import numpy as np

def save_room_pkl(filename,points,labels):
    with open(filename,'wb') as f:
        cPickle.dump([points,labels],f,protocol=2)

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
        fns=[fn.strip('\n') for fn in fns]

    for fn in fns[6:]:
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
            if pi>=3000000:
                print 'output {} part {}'.format(fn,part_id)
                pi=0
                save_room_pkl('data/Semantic3D.Net/pkl/train/'+fn+'_{}.pkl'.format(part_id),
                              np.asarray(points,np.float32),np.asarray(labels,np.int32))
                points,labels=[],[]
                part_id+=1

        if pi!=0:
            save_room_pkl('data/Semantic3D.Net/pkl/train/'+fn+'_{}.pkl'.format(part_id),
                          np.asarray(points,np.float32),np.asarray(labels,np.int32))

        print '{} done'.format(fn)

if __name__=="__main__":
    prepare_semantic3d_partition()

