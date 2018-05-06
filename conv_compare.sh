#!/usr/bin/env bash

#epoch_num=60

#python conv_compare.py --model=pointnet --epoch_num=$epoch_num
#python conv_compare.py --model=concat_ecd --epoch_num=$epoch_num
#python conv_compare.py --model=pointnet_diff_ecd --epoch_num=$epoch_num
#python conv_compare.py --model=pointnet_concat_ecd --epoch_num=$epoch_num
#python conv_compare.py --model=anchor_conv --epoch_num=$epoch_num
#python conv_compare.py --model=mlp_anchor_conv --epoch_num=$epoch_num
#python conv_compare.py --model=mlp_anchor_conv_nonorm --epoch_num=$epoch_num

#python conv_compare.py --model=v3 --epoch_num=$epoch_num
#python conv_compare.py --model=v4 --epoch_num=$epoch_num
#python conv_compare.py --model=v5 --epoch_num=$epoch_num
#python conv_compare.py --model=v6 --epoch_num=$epoch_num
#python conv_compare.py --model=v7 --epoch_num=$epoch_num
#python conv_compare.py --model=mlp_anchor_conv_v6 --epoch_num=$epoch_num
#python conv_compare.py --model=mlp_anchor_conv_v7 --epoch_num=$epoch_num
#python conv_compare.py --model=mlp_anchor_conv_v8 --epoch_num=$epoch_num

python train_feats_compare.py --num_gpus=8 --compare_type=stage
python train_feats_compare.py --num_gpus=8 --pop_id=1 --compare_type=sort_ablation
python train_feats_compare.py --num_gpus=8 --pop_id=2 --compare_type=sort_ablation
python train_feats_compare.py --num_gpus=8 --pop_id=3 --compare_type=sort_ablation
python train_feats_compare.py --num_gpus=8 --pop_id=4 --compare_type=sort_ablation
python train_feats_compare.py --num_gpus=8 --pop_id=5 --compare_type=sort_ablation
python train_feats_compare.py --num_gpus=8 --pop_id=6 --compare_type=sort_ablation
