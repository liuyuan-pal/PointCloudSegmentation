import numpy as np
from scipy.stats import mode


def compute_voxel_label_np(labels,vlens,vbegs):
    vn=vlens.shape[0]

    voxel_labels=np.empty_like(vlens)
    for i in xrange(vn):
        beg=vbegs[i]
        label,_=mode(labels[beg:beg+vlens[i]])
        voxel_labels[i]=label[0]
