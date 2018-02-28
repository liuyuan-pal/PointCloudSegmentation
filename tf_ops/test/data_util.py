import h5py
import numpy as np

def read_room_h5(room_h5_file):
    f=h5py.File(room_h5_file,'r')
    data,label = f['data'][:],f['label'][:]
    f.close()

    return data, label


def compute_nidxs_bgs(nidxs_lens):
    csum=0
    nidxs_bgs=np.empty_like(nidxs_lens)
    for i,lval in enumerate(nidxs_lens):
        nidxs_bgs[i]=csum
        csum+=lval
    return nidxs_bgs