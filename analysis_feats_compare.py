import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def read_mious(fn):
    with open(fn,'r') as f:
        mious=[]
        for line in f.readlines():
            if not line.startswith('mean iou'):
                continue
            miou=float(line.split(' ')[2])
            mious.append(miou)

    return mious


feats_name=['feats_stage2_pool','feats_stage2','feats_stage2_fc','feats_stage1','feats_stage1_fc','feats_stage0','feats_stage0_fc']

def ablation_figure():
    fns=['pop_'+fn+'.log' for fn in feats_name]

    plt.figure(0,figsize=(16, 12), dpi=80)
    for fi,fn in enumerate(fns):
        mious=read_mious('pointnet_10_concat_pre_compare/'+fn)
        label='all' if fi==0 else feats_name[fi-1]
        plt.plot(np.arange(len(mious)),mious,label=label)

    plt.legend()
    plt.savefig('ablation_feats_compare.png')
    plt.close()

def absense_figure():
    fns=[fn+'.log' for fn in feats_name]

    plt.figure(0,figsize=(16, 12), dpi=80)
    for fi,fn in enumerate(fns):
        mious=read_mious('pointnet_10_concat_pre_compare/'+fn)
        mious=np.asarray(mious)
        mious=0.6-mious
        plt.plot(np.arange(len(mious)),mious,label=feats_name[fi])

    plt.legend()
    plt.savefig('absence_feats_compare.png')
    plt.close()

if __name__=="__main__":
    absense_figure()
