# -*- encoding: utf-8 -*-
'''
@File    :   compute_metrics.py
@Time    :   2022/08/17 19:44:51
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   计算每个数据集的Dice
'''

import SimpleITK as sitk
import nibabel as nib
import os.path as osp
import numpy as np
from metrics import dice
from label_sys import label_sys_dict, label_mapping, totalseg_cls2idx
import glob
import sys
from tqdm import tqdm
import pickle
import os
from sklearn.model_selection import KFold
from collections import OrderedDict

def compute_orientation(init_axcodes, final_axcodes):
    """
    A thin wrapper around ``nib.orientations.ornt_transform``

    :param init_axcodes: Initial orientation codes
    :param final_axcodes: Target orientation codes
    :return: orientations array, start_ornt, end_ornt
    """
    ornt_init = nib.orientations.axcodes2ornt(init_axcodes)
    ornt_fin = nib.orientations.axcodes2ornt(final_axcodes)

    ornt_transf = nib.orientations.ornt_transform(ornt_init, ornt_fin)
    
    return ornt_transf, ornt_init, ornt_fin

def do_reorientation(data_array, init_axcodes, final_axcodes):
    """
    source: https://niftynet.readthedocs.io/en/dev/_modules/niftynet/io/misc_io.html#do_reorientation
    Performs the reorientation (changing order of axes)

    :param data_array: 3D Array to reorient
    :param init_axcodes: Initial orientation
    :param final_axcodes: Target orientation
    :return data_reoriented: New data array in its reoriented form
    """
    ornt_transf, ornt_init, ornt_fin = compute_orientation(init_axcodes, final_axcodes)
    if np.array_equal(ornt_init, ornt_fin):
        return data_array

    return nib.orientations.apply_orientation(data_array, ornt_transf)


def compute_dice(ts_root, gt_root, label_sys, label_mapping, prev_result=None, correct_direction=True):
    # gt_img = sitk.ReadImage(gt_root)
    # gt_arr = sitk.GetArrayFromImage(gt_img)
    if(not osp.exists(gt_root)): print(f"FileNotFound for {gt_root}"); return prev_result # do nothing
    gt_img = nib.load(gt_root)
    gt_img = nib.as_closest_canonical(gt_img)
    gt_arr = gt_img.get_fdata()

    final_result = prev_result
    if(not final_result): final_result = dict()
    for k, v in label_sys.items():
        # multi-input / multi-output preprocess
        if(k.startswith("agg:")):
            idx = [int(i) for i in k[4:].split(",")]
        else:
            idx = [int(k)]
        ts_cls_name = label_mapping[v]
        if(ts_cls_name.startswith("agg:")):
            ts_cls_name = [i for i in ts_cls_name[4:].split(",")]
        else:
            ts_cls_name = [ts_cls_name]
        # get single-class one-hot mask
        ts_cls_arr = np.zeros_like(gt_arr)
        for sub_ts_file in ts_cls_name:
            filename = osp.join(ts_root, sub_ts_file+".nii.gz")
            if(not osp.exists(filename)): print(f"FileNotFound for {filename}"); continue
            sub_ts_img = nib.load(filename)
            sub_ts_img = nib.as_closest_canonical(sub_ts_img)
            sub_ts_cls_arr = sub_ts_img.get_fdata()
            ts_cls_arr[sub_ts_cls_arr==1] = 1

        gt_cls_arr = np.zeros_like(gt_arr)
        for i in idx:   
            gt_cls_arr[gt_arr==i] = 1
        
        score = dice(ts_cls_arr, gt_cls_arr)
        if(not v in final_result.keys()): final_result[v] = []
        final_result[v].append(score)
    return final_result

def compute_dice_nnUNet(ts_root, gt_root, label_sys, label_mapping, prev_result=None, correct_direction=True):
    # gt_img = sitk.ReadImage(gt_root)
    # gt_arr = sitk.GetArrayFromImage(gt_img)
    if(not osp.exists(gt_root)): print(f"FileNotFound for {gt_root}"); return prev_result # do nothing
    if(not osp.exists(ts_root)): print(f"FileNotFound for {ts_root}"); return prev_result # do nothing
    gt_img = nib.load(gt_root)
    if(correct_direction): 
        gt_img = nib.as_closest_canonical(gt_img)
    gt_arr = gt_img.get_fdata()
    ts_img = nib.load(ts_root)
    if(correct_direction): 
        ts_img = nib.as_closest_canonical(ts_img)
    ts_arr = ts_img.get_fdata()
    # if(correct_direction): 
    #     ts_arr = do_reorientation(ts_arr, tuple(nib.aff2axcodes(ts_img.affine)), tuple(nib.aff2axcodes(gt_img.affine)))

    final_result = prev_result
    if(not final_result): final_result = dict()
    for k, v in label_sys.items():
        # multi-input / multi-output preprocess
        if(k.startswith("agg:")):
            idx = [int(i) for i in k[4:].split(",")]
        else:
            idx = [int(k)]
        ts_cls_name = label_mapping[v]
        if(ts_cls_name.startswith("agg:")):
            ts_cls_name = [i for i in ts_cls_name[4:].split(",")]
        else:
            ts_cls_name = [ts_cls_name]
        # get single-class one-hot mask
        ts_cls_arr = np.zeros_like(gt_arr)
        for cls_name in ts_cls_name:
            ts_cls_idx = totalseg_cls2idx[cls_name]
            ts_cls_arr[ts_arr==ts_cls_idx] = 1

        gt_cls_arr = np.zeros_like(gt_arr)
        for i in idx:   
            gt_cls_arr[gt_arr==i] = 1
        
        score = dice(ts_cls_arr, gt_cls_arr)
        if(not v in final_result.keys()): final_result[v] = []
        final_result[v].append(score)
    return final_result




if __name__ == "__main__":
    test = False
    dataset = sys.argv[1]
    mode = sys.argv[2]
    fold = sys.argv[3]
    if(len(sys.argv)>4): test = bool(sys.argv[4])
    print("test", dataset, "mode", mode, "fold", fold)


    is_nnUNet = not mode in ("normal", "fast")
    correct_direction = True
    # if(dataset=="Task011_BTCV" and is_nnUNet):
    #     correct_direction = False

    # get label_sys to map the label
    general_ts_root = f"/mnt/petrelfs/wanghaoyu/gmai/totalseg_result/{dataset}/"
    general_gt_root = f"/mnt/petrelfs/wanghaoyu/gmai/totalseg_tmp_data/raw_data/{dataset}/labelsTr"
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
    print(label_sys)
    res = None

    # search for eval data
    if(is_nnUNet):
        compute_fn = compute_dice_nnUNet
        # [TO-DO] nnUNet infer result
        data_list = glob.glob(osp.join(general_ts_root, f"{mode}__nnUNetPlansv2.1", "*.nii.gz"))
    else:    
        compute_fn = compute_dice    
        data_list = glob.glob(osp.join(general_ts_root, f"*_{mode}"))
        # 读取split，并输出split指标
    assert fold in ['0', '1', '2', '3', '4', 'all']
    if(fold=='all'):
        print("compute metrics of all data")
    else:
        print(f"compute metrics of fold {fold}")
        general_split_root = f"/mnt/petrelfs/wanghaoyu/gmai/totalseg_tmp_data/split/{dataset}/splits_final.pkl"
        if(osp.exists(general_split_root)):
            splits = pickle.load(open(general_split_root, "rb"))
        else:
            print("[Warning] split_final.pkl is not found, automatic compute the split")
            splits = []
            if(is_nnUNet): all_keys_sorted = np.sort(list([osp.basename(d).split("_0000.nii.gz")[0] for d in data_list]))
            else: all_keys_sorted = np.sort(list([osp.basename(d).split("_0000_")[0] for d in data_list]))
            print("all_keys_sorted", all_keys_sorted)
            kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
            for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                train_keys = np.array(all_keys_sorted)[train_idx]
                test_keys = np.array(all_keys_sorted)[test_idx]
                splits.append(OrderedDict())
                splits[-1]['train'] = train_keys
                splits[-1]['val'] = test_keys
        if(is_nnUNet): 
            data_list = [osp.join(general_ts_root, f"{mode}__nnUNetPlansv2.1", f+f'_0000.nii.gz') for f in splits[int(fold)]['val']] 
        else: 
            data_list = [osp.join(general_ts_root, f+f'_0000_{mode}') for f in splits[int(fold)]['val']] 
        print(f"fold_{fold}_val: ", len(data_list))

    # test one case 
    if(test): 
        print("test mode")
        data_list = data_list[:1]
        print("shorter data_list for test: ", data_list)

    pbar = tqdm(data_list)
    for data in pbar:
        if(is_nnUNet): 
            label_path = osp.join(general_gt_root, osp.basename(data).split("_0000.nii.gz")[0]+".nii.gz")
        else:
            label_path = osp.join(general_gt_root, osp.basename(data).split("_0000_")[0]+".nii.gz")
        res = compute_fn(data, label_path, label_sys=label_sys, label_mapping=label_mapping, prev_result=res, correct_direction=correct_direction)
    final_dice = []
    for k, v in res.items():
        mDice = np.nanmean(v)
        final_dice.append(mDice)
        print(k, mDice)
    print("all:", np.nanmean(final_dice))

    print(["cls"]+[d.split('/')[-1] for d in data_list])
    for k, v in res.items():
        print(k, v)

    
    
