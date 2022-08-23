from git import HEAD


# -*- encoding: utf-8 -*-
'''
@File    :   json_create_openNeuro.py
@Time    :   2022/05/08 16:13:01
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   针对openNeuro数据集提取json文件
'''

import json
import glob
import os.path as osp
import os

data_root = "/mnt/lustre/share_data/gmai/dataset/preprocessed/unlabeled"

def writeUnlabeledDatasetJson(tr_case_list, filename, meta, modality="all"):
    dataset_info = meta.copy()
    dataset_info["modality"] = {"0": modality}
    dataset_info["numTraining"] = len(tr_case_list)
    dataset_info["numTest"] = 0
    dataset_info["training"] = tr_case_list
    dataset_info["test"] = {}
    with open(filename, "w") as fp:
        json.dump(dataset_info, fp, indent=4)
    pass



if __name__ == "__main__":
    dataset_path_list = glob.glob(osp.join(data_root, "ds*"))
    # dataset_path_list = glob.glob(osp.join(data_root, "ds002380"))
    for dataset_path in dataset_path_list:
        if not (osp.isdir(dataset_path)): 
            print(f"[Warning] {dataset_path} is not a dir!")
            continue
        # read dataset_description
        with open(osp.join(dataset_path, "dataset_description.json"), "r") as json_file:
            dataset_info = json.load(json_file)
            try:
                dataset_info["license"] = dataset_info.pop("License")
            except:
                dataset_info["license"] = ""
            try:
                dataset_info["name"] = dataset_info.pop("Name")
            except:
                dataset_info["name"] = ""
            try:
                dataset_info["reference"] = dataset_info.pop("ReferencesAndLinks")
            except:    
                dataset_info["reference"] = ""
            try:    
                dataset_info["release"] = dataset_info.pop("BIDSVersion")
            except:   
                dataset_info["release"] = "0.0"

            dataset_info["labels"] = {}
            dataset_info["tensorImageSize"] = "4D"
            
            nifti_data_list = glob.glob(osp.join(dataset_path, "sub*/*/*.nii*")) + glob.glob(osp.join(dataset_path, "sub*/ses*/*/*.nii*"))
            nifti_data_dict_by_modality = {}
            for data_path in nifti_data_list:
                rel_data_path = osp.relpath(data_path, dataset_path)
                img_item = {"image": rel_data_path}
                # split by modality
                modality = osp.basename(data_path).split(".")[0].split("_")[-1]
                if not (modality in nifti_data_dict_by_modality.keys()):
                    nifti_data_dict_by_modality[modality] = [img_item]
                else:
                    nifti_data_dict_by_modality[modality].append(img_item)
                # nifti_data_dict_by_modality["all"].append(img_item) 

            # output dataset.json
            for k, v in nifti_data_dict_by_modality.items():
                fname = osp.join(dataset_path, f"dataset-{k}.json")
                writeUnlabeledDatasetJson(v, fname, 
                                          meta=dataset_info, modality=k)
                print("write", fname)

    print("end")

    






