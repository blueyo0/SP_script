import enum
from git import HEAD


# -*- encoding: utf-8 -*-
'''
@File    :   create_fake_labels.py
@Time    :   2022/06/26 23:49:26
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   为了适配nnUNet带label的plan，生成一个全是1的假seg
'''
import json
import SimpleITK as sitk
from monai import transforms
import os.path as osp
import numpy as np
import tqdm

def writeUnlabeledDatasetJson(dataset_info, filename):
    with open(filename, "w") as fp:
        json.dump(dataset_info, fp, indent=4)

def get_file_list(json_file, return_json=True):
    dataset_info = json.load(open(json_file, "r"))
    tr_list = dataset_info["training"]
    if(return_json):
        return tr_list, dataset_info
    else:
        return tr_list

if __name__ == "__main__":
    json_filename = "/mnt/cache/wanghaoyu/data/AbdomenCT5K/dataset.json"
    output_filename = "/mnt/cache/wanghaoyu/data/AbdomenCT5K/dataset.json"
    dataset_dir = "/mnt/cache/wanghaoyu/data/AbdomenCT5K"
    tr_list, dataset_info = get_file_list(json_file=json_filename)
    # load_trans = transforms.LoadImaged(keys=["image", "label"], allow_missing_keys=True)
    # save_trans = transforms.SaveImaged(keys=["image"], output_dir="/mnt/cache/wanghaoyu/data/AbdomenCT5K/labelsTr", output_postfix="label", resample=True, output_dtype=np.int16, separate_folder=False)
    # for i, case in enumerate(tr_list):
    pbar = tqdm.tqdm(tr_list)
    for i, case in enumerate(pbar):
        out_file = osp.join(dataset_dir, "labelsTr", osp.basename(case["image"]))
        if(osp.exists(out_file)): continue
        itk_img = sitk.ReadImage(osp.join(dataset_dir, case["image"]))
        img = sitk.GetArrayFromImage(itk_img).astype(np.int16)
        img[:] = 1
        out = sitk.GetImageFromArray(img)
        out.SetSpacing(itk_img.GetSpacing())
        out.SetOrigin(itk_img.GetOrigin())
        out.SetDirection(itk_img.GetDirection())
        sitk.WriteImage(out, out_file)
        print("file written", out_file)
        tr_list[i]["label"] = "labelsTr/" + osp.basename(case["image"])
        # print(tr_list[i])
    dataset_info["training"] = tr_list
    writeUnlabeledDatasetJson(dataset_info, output_filename)

