# -*- encoding: utf-8 -*-
'''
@File    :   compute_metrics.py
@Time    :   2022/08/17 19:44:51
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   只适用于TotalSeg的Dice计算
'''

from metrics import dice
import SimpleITK as sitk
import os.path as osp
import os
import glob
import sys
from tqdm import tqdm
import numpy as np

# def compute_dice(ts_root, gt_root, label_sys, label_mapping, prev_result=None, flip_axis=None):
#     final_result = prev_result
#     if(not final_result): final_result = dict()
#     for k, v in label_sys.items():
#         idx = int(k)
#         gt_img = sitk.ReadImage(osp.join(gt_root, label_mapping[v]+".nii.gz"))
#         gt_arr = sitk.GetArrayFromImage(gt_img)
#         filename = osp.join(ts_root, label_mapping[v]+".nii.gz")
#         ts_img = sitk.ReadImage(filename)
#         ts_img.SetDirection(gt_img.GetDirection())
#         ts_cls_arr = sitk.GetArrayFromImage(ts_img)
#         #if(flip_axis): ts_cls_arr = np.flip(ts_cls_arr, axis=flip_axis)
#         gt_cls_arr = np.zeros_like(gt_arr)
#         gt_cls_arr[gt_arr==idx] = 1
#         score = dice(ts_cls_arr, gt_cls_arr)
#         if(not v in final_result.keys()): final_result[v] = []
#         final_result[v].append(score)
#     return final_result

def compute_dice(ts_root, gt_root, label_sys, label_mapping, prev_result=None, flip_axis=None):
    gt_img = sitk.ReadImage(gt_root)
    gt_arr = sitk.GetArrayFromImage(gt_img)
    final_result = prev_result
    if(not final_result): final_result = dict()
    for k, v in label_sys.items():
        idx = int(k)
        filename = osp.join(ts_root, label_mapping[v]+".nii.gz")
        ts_img = sitk.ReadImage(filename)
        ts_img.SetDirection(gt_img.GetDirection())
        ts_cls_arr = sitk.GetArrayFromImage(ts_img)
        #if(flip_axis): ts_cls_arr = np.flip(ts_cls_arr, axis=flip_axis)
        gt_cls_arr = np.zeros_like(gt_arr)
        gt_cls_arr[gt_arr==idx] = 1
        score = dice(ts_cls_arr, gt_cls_arr)
        if(not v in final_result.keys()): final_result[v] = []
        final_result[v].append(score)
    return final_result


check_data_seg_path = "/mnt/petrelfs/wanghaoyu/test/liver_0_0000_fast"
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

if __name__ == "__main__":
    #ts_root = "D:\\download\\flare22\\label" # totalseg root
    #gt_root = "D:\\download\\flare22\\FLARE22_Tr_0022_gt.nii.gz"

    dataset = sys.argv[1]
    mode = sys.argv[2]
    fold = sys.argv[3]
    if(len(sys.argv)>4): test = bool(sys.argv[4])
    print("test", dataset, "mode", mode)
    # for dataset in ["AMOS", "FLARE22", "CMRxMotion", "KIPA", "LAScarQS"]:
    for dataset in [dataset]:
        general_ts_root = f"/mnt/petrelfs/wanghaoyu/test/total_pred/{dataset}"
        general_gt_root = f"/mnt/petrelfs/wanghaoyu/test/{dataset}/labelsTr"
        # label_sys = label_sys_dict[dataset]
        print(label_sys)
        res = None
        # data_list = glob.glob(osp.join(general_ts_root, f"*_{mode}"))
        print(f"compute metrics of fold {fold}")
        general_split_root = f"/mnt/petrelfs/wanghaoyu/gmai/nnUNet_preprocessed/{dataset}/splits_final.pkl"
        splits = pickle.load(open(general_split_root, "rb"))
        data_list = [osp.join(general_ts_root, f+f'_0000_{mode}') for f in splits[int(fold)]['val']]
        print(f"fold_{fold}_val: ", len(data_list))

        # test one case 
        if(test): 
            print("test mode")
            data_list = data_list[:1]
            print("shorter data_list for test: ", data_list)
        
        pabr = tqdm(data_list)
        for data in pabr:
            # print(data)
            label_path = osp.join(general_gt_root, osp.basename(data).split("_0000")[0]+".nii.gz")
            #print(osp.exists(label_path), label_path)
            res = compute_dice(data, label_path, label_sys=label_sys, label_mapping=label_mapping, prev_result=res, flip_axis=2)
        # print(res)
        final_dice = []
        for k, v in res.items():
            mDice = np.nanmean(v)
            final_dice.append(mDice)
            print(k, "\t", mDice)
        print("all:", "\t", np.nanmean(final_dice))

    #res = compute_dice(ts_root=ts_root, gt_root=gt_root, label_sys=FLARE22_sys, label_mapping=FLARE22_mapping)
    #res = compute_dice(ts_root=ts_root, gt_root=gt_root, label_sys=FLARE22_sys, label_mapping=FLARE22_mapping, prev_result=res)
    
        print(res)
    
    
