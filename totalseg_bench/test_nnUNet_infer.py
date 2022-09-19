# -*- encoding: utf-8 -*-
'''
@File    :   test_nnUNet_infer.py
@Time    :   2022/08/30 10:54:40
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   nnUNet 单模型 infer 脚本
'''

from pathlib import Path
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from sklearn import model_selection
from nnunet.inference.predict import predict_cases
from batchgenerators.utilities.file_and_folder_operations import subfiles, join
import shutil
import sys

if __name__ == '__main__':
    trainer = sys.argv[1]
    dataset = sys.argv[2]
    model = f"{trainer}__nnUNetPlansv2.1"
    input_folder = f'/mnt/petrelfs/wanghaoyu/gmai/totalseg_tmp_data/raw_data/{dataset}/imagesTr'
    output_folder = f'/mnt/petrelfs/wanghaoyu/gmai/totalseg_result/{dataset}'
    parameter_folder = f'/mnt/lustre/wanghaoyu/runs/nnUNet/RESULTS_FOLDER/nnUNet/3d_fullres/Task558_Totalsegmentator_dataset'
    folds = (1)

    test_files = subfiles(input_folder, suffix='_0000.nii.gz', join=False)

    # test_files = test_files[:5]

    input_files = [join(input_folder, tf) for tf in test_files]
    output_files = [join(output_folder, model, tf) for tf in test_files]
    predict_cases(join(parameter_folder, model), [[i] for i in input_files], output_files, folds, save_npz=False,
                    num_threads_preprocessing=2, num_threads_nifti_save=2, segs_from_prev_stage=None, do_tta=False,
                    mixed_precision=True, overwrite_existing=False, all_in_gpu=False, step_size=0.5)

    # 指标计算
    # gt_folder = f'/mnt/petrelfs/wanghaoyu/gmai/nnUNet_raw_data_base/nnUNet_raw_data/{dataset}/labelsTr'
    # nn_folder = join(output_folder, model)
    # print(f"compute metrics of fold {fold}")
    # general_split_root = f"/mnt/petrelfs/wanghaoyu/gmai/nnUNet_preprocessed/{dataset}/splits_final.pkl"
    # if(osp.exists(general_split_root)):
    #     splits = pickle.load(open(general_split_root, "rb"))
    #     data_list = [osp.join(general_ts_root, f+f'_0000_{mode}') for f in splits[int(fold)]['val']]
    # else:
    #     splits = []
    #     all_keys_sorted = np.sort(list([osp.basename(d).split("_0000_")[0] for d in data_list]))
    #     print("all_keys_sorted", all_keys_sorted)
    #     kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
    #     for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
    #         train_keys = np.array(all_keys_sorted)[train_idx]
    #         test_keys = np.array(all_keys_sorted)[test_idx]
    #         splits.append(OrderedDict())
    #         splits[-1]['train'] = train_keys
    #         splits[-1]['val'] = test_keys
    #     data_list = [osp.join(general_ts_root, f+f'_0000_{mode}') for f in splits[int(fold)]['val']]
    # print(f"fold_{fold}_val: ", len(data_list))
    # pbar = tqdm(data_list)
    # for data in pbar:
    #     label_path = osp.join(general_gt_root, osp.basename(data).split("_0000_")[0]+".nii.gz")
    #     res = compute_dice(data, label_path, label_sys=label_sys, label_mapping=label_mapping, prev_result=res)
    # final_dice = []
    # for k, v in res.items():
    #     mDice = np.nanmean(v)
    #     final_dice.append(mDice)
    #     print(k, mDice)
    # print("all:", np.nanmean(final_dice))

    # print(["cls"]+[d.split('/')[-1] for d in data_list])
    # for k, v in res.items():
    #     print(k, v)



