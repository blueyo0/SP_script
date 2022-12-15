# -*- encoding: utf-8 -*-
'''
@File    :   test_nnUNet_infer.py
@Time    :   2022/08/30 10:54:40
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   nnUNet 单模型 infer 脚本
'''

import os
import os.path as osp
import pickle
import numpy as np
from nnunet.inference.predict import predict_cases
from batchgenerators.utilities.file_and_folder_operations import subfiles, join
from compute_metrics import compute_dice_nnUNet
from label_sys import label_sys_dict, label_mapping, totalseg_cls2idx
import sys
from tqdm import tqdm

if __name__ == '__main__':
    trainer = sys.argv[1]
    dataset = sys.argv[2]
    val_fold = sys.argv[3]
    if(len(sys.argv)>4): test = bool(sys.argv[4])
    else: test = False
    model = f"{trainer}__nnUNetPlansv2.1"
    input_folder = f'/mnt/petrelfs/wanghaoyu/gmai/totalseg_tmp_data/raw_data/{dataset}/imagesTr'
    output_folder = f'/mnt/petrelfs/wanghaoyu/gmai/totalseg_result/{dataset}_ep600'
    parameter_folder = f'/mnt/lustre/wanghaoyu/runs/nnUNet/RESULTS_FOLDER/nnUNet/3d_fullres/Task558_Totalsegmentator_dataset'
    folds = (1)

    test_files = subfiles(input_folder, suffix='_0000.nii.gz', join=False)

    if(test): test_files = test_files[:2]

    input_files = [join(input_folder, tf) for tf in test_files]
    output_files = [join(output_folder, model, tf) for tf in test_files]
    predict_cases(join(parameter_folder, model), [[i] for i in input_files], output_files, folds, save_npz=False,
                    num_threads_preprocessing=2, num_threads_nifti_save=2, segs_from_prev_stage=None, do_tta=False,
                    mixed_precision=True, overwrite_existing=False, all_in_gpu=False, step_size=0.5,
                    checkpoint_name="fp32_model_ep_600")

    # 指标计算
    general_ts_root = output_folder
    general_gt_root = f"/mnt/petrelfs/wanghaoyu/gmai/totalseg_tmp_data/raw_data/{dataset}/labelsTr"
    gt_folder = f'/mnt/petrelfs/wanghaoyu/gmai/nnUNet_raw_data_base/nnUNet_raw_data/{dataset}/labelsTr'
    print(f"compute metrics of fold {val_fold}")
    general_split_root = f"/mnt/petrelfs/wanghaoyu/gmai/totalseg_tmp_data/split/{dataset}/splits_final.pkl"
    if(osp.exists(general_split_root)):
        splits = pickle.load(open(general_split_root, "rb"))
        data_list = [osp.join(general_ts_root, model, f+f'_0000.nii.gz') for f in splits[int(val_fold)]['val']] 
    else:
        data_list = output_files 
        

    compute_fn = compute_dice_nnUNet
    if(dataset=="Task558_Totalsegmentator_dataset" or dataset=="Task559_TS_test"):
        check_data_seg_path = "/mnt/petrelfs/wanghaoyu/why/liver_0_0000_fast"
        labels = sorted(os.listdir(check_data_seg_path))
        label_sys = {
            # "0": "background",
        }
        label_mapping=dict()
        for idx, label in enumerate(labels):
            label_id = idx + 1  # 所有标签从1开始排，因为0给背景类别
            label_name = label.split(".")[0]  # 标签名
            label_sys[str(label_id)] = label_name
            label_mapping[label_name] = label_name
    else: label_sys = label_sys_dict[dataset]
    print("labels:", label_sys)
    pbar = tqdm(data_list)
    res = None
    for data in pbar:
        label_path = osp.join(general_gt_root, osp.basename(data).split("_0000.nii.gz")[0]+".nii.gz")
        res = compute_fn(data, label_path, label_sys=label_sys, label_mapping=label_mapping, prev_result=res, 
                         correct_direction=False if(dataset=="Task011_BTCV") else True)
    final_dice = []
    for k, v in res.items():
        mDice = np.nanmean(v)
        final_dice.append(mDice)
        print(k, mDice)
    print("all:", np.nanmean(final_dice))
    print("result_pre_class", "\t".join([str(d) for d in final_dice]))

    print(["cls"]+[d.split('/')[-1] for d in data_list])
    for k, v in res.items():
        print(k, v)


