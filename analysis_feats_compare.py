import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from io_util import get_class_names

def read_mious(fn):
    with open(fn,'r') as f:
        mious=[]
        for line in f.readlines():
            if not line.startswith('mean iou'):
                continue
            miou=float(line.split(' ')[2])
            mious.append(miou)

    return mious

def read_iou_class(fn,class_name):
    with open(fn,'r') as f:
        ious=[]
        for line in f.readlines():
            if not line.startswith('{} iou'.format(class_name)):
                continue
            iou=float(line.split(' ')[2])
            ious.append(iou)

    return ious


feats_name=['feats_stage2_pool','feats_stage2','feats_stage2_fc','feats_stage1','feats_stage1_fc','feats_stage0','feats_stage0_fc']
feats_sort_name=['feats_stage2_pool','feats_stage2_fc','feats_stage2','feats_stage1_fc','feats_stage1','feats_stage0_fc','feats_stage0']

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

def sort_figure():
    fns=['sort_'+fn+'.log' for fn in feats_sort_name[1:]]

    plt.figure(0,figsize=(16, 12), dpi=80)
    for fi,fn in enumerate(fns):
        mious=read_mious('pointnet_10_concat_pre_compare/'+fn)
        mious=np.asarray(mious)
        mious=0.6-mious
        plt.plot(np.arange(len(mious)),mious,label=feats_sort_name[fi+1])

    plt.legend()
    plt.savefig('test_result/sort_feats_compare.png')
    plt.close()

def sort_bar():
    fns=['sort_'+fn+'.log' for fn in feats_sort_name]

    mean_iou=[]
    max_iou=[]
    for fi,fn in enumerate(fns):
        if fi==0: fn='feats_stage2_pool.log'
        mious=read_mious('pointnet_10_concat_pre_compare/'+fn)
        mean_iou.append(0.6-np.mean(mious[10:]))
        max_iou.append(0.6-np.max(mious[10:]))

    plt.figure(0,figsize=(16, 12), dpi=80)
    plt.bar(np.arange(len(mean_iou)),mean_iou,)
    plt.xticks(np.arange(len(mean_iou)),feats_sort_name)
    plt.legend()
    plt.savefig('test_result/sort_feats_compare_bar_mean.png')
    plt.close()

    plt.figure(0,figsize=(16, 12), dpi=80)
    plt.bar(np.arange(len(max_iou)),max_iou,)
    plt.xticks(np.arange(len(max_iou)),feats_sort_name)
    plt.legend()
    plt.savefig('test_result/sort_feats_compare_bar_max.png')
    plt.close()

def sort_bar_classes():
    fns=['sort_'+fn+'.log' for fn in feats_sort_name]

    for cn in get_class_names():
        mean_iou=[]
        for fi,fn in enumerate(fns):
            if fi==0: fn='feats_stage2_pool.log'
            mious=read_iou_class('pointnet_10_concat_pre_compare/'+fn,cn)
            mean_iou.append(np.mean(mious[10:]))

        plt.figure(0,figsize=(16, 12), dpi=80)
        plt.bar(np.arange(len(mean_iou)),mean_iou,)
        plt.xticks(np.arange(len(mean_iou)),feats_sort_name)
        plt.legend()
        plt.savefig('test_result/sort_feats_compare_bar_mean_{}.png'.format(cn))
        plt.close()


def model_figure():
    fns=['pointnet_10_concat_pre_deconv.log','pointnet_10_concat_pre_embed.log',
         'pointnet_10_dilated_deconv.log','pointnet_10_dilated.log','pointnet_14_dilated.log','pointnet_20_baseline.log',
         'pointnet_20_baseline_v2.log',
         'pointnet_5_concat.log','pointnet_5_concat_pre_deconv.log','pointnet_5_concat_pre.log','pointnet_13_dilated_embed.log',
         'pgnet_v8.log']

    base_mious = read_mious('pointnet_10_concat_pre.log')
    for fi,fn in enumerate(fns):
        plt.figure(0,figsize=(16, 12), dpi=80)
        mious=read_mious(fn)
        mious=np.asarray(mious)
        plt.plot(np.arange(len(base_mious)),base_mious,label='pointnet_10_concat_pre')
        plt.plot(np.arange(len(mious)),mious,label=fn)
        plt.legend()
        plt.savefig('test_result/model_compare_{}.png'.format(fn))
        plt.close()



if __name__=="__main__":
    model_figure()
    # print read_iou_class('pointnet_10_concat_pre_compare/feats_stage2_pool.log','beam')