# -*- encoding: utf-8 -*-
'''
@File    :   aggerate_subdataset.py
@Time    :   2022/07/02 14:49:23
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   根据dataset id的list，将这些preprocessed后的nnUNet文件夹处理为一个大dataset
'''

'''
所有要合并的内容
- nnUNetData_plans_General_stage0 (cp files)
- nnUNetData_plans_General_stage1 (cp files)
- gt_segmentations (cp files)
- nnUNetPlansGeneral_plans_3D.pkl
    info["dataset_properties"]["all_sizes"]
    info["dataset_properties"]["all_spacings"]
    info["dataset_properties"]["size_reductions"] (ordered dict)
    info["list_of_npz_files"]
    info["original_sizes"]
    info["original_spacings"]
- dataset_properties.pkl
    info["all_sizes"]
    info["all_spacings"]
    info["size_reductions"] (ordered dict)
- dataset.json
    json_file["training"]
    json_file["name"]
'''

import pickle
import json
import shutil
from multiprocessing import Pool
from itertools import chain
from collections import OrderedDict
import os
import os.path as osp

def pickle_load(file):
    f=open(file, 'rb')
    info = pickle.load(f,encoding='bytes')
    return info

def json_load(file):
    f=open(file, 'r')
    info = json.load(f)
    return info

def agg_ordered_dict(x, y):
    return OrderedDict(chain(x.items(), y.items()))


if __name__ == "__main__":
    default_num_threads = 16
    sub_dataset_list = [
        "Task300_AbdomenSubset0",
        "Task301_AbdomenSubset1",
        "Task302_AbdomenSubset2",
        "Task303_AbdomenSubset3",
        "Task304_AbdomenSubset4",
        "Task305_AbdomenSubset5",
        "Task306_AbdomenSubset6",
        "Task307_AbdomenSubset7",
        "Task308_AbdomenSubset8",
        "Task309_AbdomenSubset9",
        "Task310_AbdomenSubset10",
        "Task154_RibFrac",
        "Task029_LITS",
        "Task022_FLARE2022",
        "Task021_KiTS2021", 
        "Task011_BTCV",
        "Task010_Colon",
        "Task009_Spleen",
        "Task008_HepaticVessel",
        "Task007_Pancreas",
        "Task003_Liver",
        # "Task020_AbdomenCT1K", #[duplicate with AbdomenSubset]
        # "Task040_KiTS", #[duplicate with KiTS2021]
    ]
    # target_dataset = "Task320_test_agg"
    target_dataset = "Task322_AbdomenCT8K"
    nnUNet_preprocessed = "/mnt/lustre/share_data/gmai/nnUNet_preprocessed"

    target_path = osp.join(nnUNet_preprocessed, target_dataset)
    target_folder_stage0 = osp.join(target_path, "nnUNetData_plans_General_stage0")
    target_folder_stage1 = osp.join(target_path, "nnUNetData_plans_General_stage1")
    target_folder_gt = osp.join(target_path, "gt_segmentations")
    to_mkdir_list = [target_path, target_folder_stage0, target_folder_stage1, target_folder_gt]
    for folder in to_mkdir_list:
        if(not osp.exists(folder)): 
            os.makedirs(folder)
    to_mv_list = dict(stage0=[], stage1=[], gt_segmentations=[])
    for sub_dataset in sub_dataset_list:
        for stage in ["stage0", "stage1"]:
            data_list = os.listdir(osp.join(nnUNet_preprocessed, sub_dataset, f"nnUNetData_plans_General_{stage}"))
            for data in data_list:
                to_mv_list[stage].append(dict(
                    src=osp.join(nnUNet_preprocessed, sub_dataset, f"nnUNetData_plans_General_{stage}", data),
                    dst=osp.join(target_path, f"nnUNetData_plans_General_{stage}", data),
                ))
        data_list = os.listdir(osp.join(nnUNet_preprocessed, sub_dataset, "gt_segmentations"))
        for data in data_list:
            to_mv_list["gt_segmentations"].append(dict(
                src=osp.join(nnUNet_preprocessed, sub_dataset, "gt_segmentations", data),
                dst=osp.join(target_path, "gt_segmentations", data),
            ))

    def move_file(args):
        src_file, dst_file = args["src"], args["dst"]
        print("cp", src_file, dst_file)
        shutil.copyfile(src_file, dst_file)
        return None   

    p = Pool(default_num_threads)
    p.map(move_file, to_mv_list["stage0"])
    p.map(move_file, to_mv_list["stage1"])
    p.map(move_file, to_mv_list["gt_segmentations"])
    p.close()
    p.join()

    ''' <<< aggre files >>>
    - nnUNetPlansGeneral_plans_3D.pkl
        info["dataset_properties"]["all_sizes"]
        info["dataset_properties"]["all_spacings"]
        info["dataset_properties"]["size_reductions"] (ordered dict)
        info["list_of_npz_files"]
        info["original_sizes"]
        info["original_spacings"]
    - dataset_properties.pkl
        info["all_sizes"]
        info["all_spacings"]
        info["size_reductions"] (ordered dict)
    - dataset.json
        json_file["training"]
        json_file["name"]
    '''
    plan_pkl_file = None
    dataset_prop_file = None
    dataset_json_file = None
    for sub_dataset in sub_dataset_list:
        plan_info = pickle_load(osp.join(nnUNet_preprocessed, sub_dataset, "nnUNetPlansGeneral_plans_3D.pkl"))
        prop_info = pickle_load(osp.join(nnUNet_preprocessed, sub_dataset, "dataset_properties.pkl"))
        json_info = json_load(osp.join(nnUNet_preprocessed, sub_dataset, "dataset.json"))
        if(not plan_pkl_file):
            plan_pkl_file = plan_info
        else:
            plan_pkl_file["dataset_properties"]["all_sizes"] += plan_info["dataset_properties"]["all_sizes"]
            plan_pkl_file["dataset_properties"]["all_spacings"] += plan_info["dataset_properties"]["all_spacings"]
            plan_pkl_file["dataset_properties"]["size_reductions"] = agg_ordered_dict(plan_pkl_file["dataset_properties"]["size_reductions"], 
                                                                                      plan_info["dataset_properties"]["size_reductions"])
            plan_pkl_file["list_of_npz_files"] += plan_info["list_of_npz_files"]
            plan_pkl_file["original_sizes"] += plan_info["original_sizes"]
            plan_pkl_file["original_spacings"] += plan_info["original_spacings"]

        if(not dataset_prop_file):
            dataset_prop_file = prop_info
        else:
            dataset_prop_file["all_sizes"] += prop_info["all_sizes"]
            dataset_prop_file["all_spacings"] += prop_info["all_spacings"]
            dataset_prop_file["size_reductions"] = agg_ordered_dict(dataset_prop_file["size_reductions"], prop_info["size_reductions"])

        if(not dataset_json_file):
            dataset_json_file = json_info
        else:
            dataset_json_file["training"] += json_info["training"]

    with open(osp.join(target_path, "nnUNetPlansGeneral_plans_3D.pkl"), "wb") as fp:
        pickle.dump(plan_pkl_file, fp)
    with open(osp.join(target_path, "dataset_properties.pkl"), "wb") as fp:
        pickle.dump(dataset_prop_file, fp)
    with open(osp.join(target_path, "dataset.json"), "w") as fp:
        json.dump(dataset_json_file, fp, indent=4)