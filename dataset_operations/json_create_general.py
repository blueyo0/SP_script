# -*- encoding: utf-8 -*-
'''
@File    :   json_create_general.py
@Time    :   2022/05/14 20:29:32
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   自动判断dataset类型，并生成json
'''
import os
import os.path as osp
import glob
import json

debug_mode = False
# debug_mode = True
# file_path = "/mnt/cache/wanghaoyu/preprocess/data/uploaded_pre_path_test.txt"
# file_path = "/mnt/cache/wanghaoyu/preprocess/data/uploaded_pre_path_todo.txt"
file_path = "/mnt/cache/wanghaoyu/preprocess/data/Ab5K.txt"
EXTRA_LIST = {
    "TCIA": [],
    "openNeuro": ["SIMON"],
}
dataset2modality = {
    "CT-Covid-19-August2020" : "CT",
    "IXI" : "MR",
    "STOIC": "CT",
    "Colin3T7T": "MR",
    "TCGA-LUSC": "CT",
    "AbdomenCT5K": "CT"
}


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

def process_openneuro(dataset_path):
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


def process_tcia(dataset_path):
    dataset_info = dict()
    dataset_info["license"] = ""    
    dataset_info["name"] = osp.basename(dataset_path)
    dataset_info["reference"] = ""
    dataset_info["release"] = "0.0"
    dataset_info["labels"] = {}
    dataset_info["tensorImageSize"] = "4D"
    
    nifti_data_list = glob.glob(osp.join(dataset_path, "*/*.nii*"))
    nifti_data_dict_by_modality = {}
    for data_path in nifti_data_list:
        rel_data_path = osp.relpath(data_path, dataset_path)
        img_item = {"image": rel_data_path}
        # split by modality
        modality = osp.basename(osp.dirname(data_path))
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

def process_unknown(dataset_path, meta):
    dataset_info = dict()
    dataset_info["license"] = ""    
    dataset_info["name"] = osp.basename(dataset_path)
    dataset_info["reference"] = ""
    dataset_info["release"] = "0.0"
    dataset_info["labels"] = {}
    dataset_info["tensorImageSize"] = "4D"
    # 对于不认识的数据集，直接深度搜索所有nifti，然后输出为meta中提供的模态\
    nifti_data_dict_by_modality = {}
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if(f.endswith(".nii") or f.endswith(".nii.gz")):
                rel_data_path = osp.relpath(osp.join(root, f), dataset_path)
                img_item = {"image": rel_data_path}
                # split by modality
                modality = meta["modality"]
                if not (modality in nifti_data_dict_by_modality.keys()):
                    nifti_data_dict_by_modality[modality] = [img_item]
                else:
                    nifti_data_dict_by_modality[modality].append(img_item)
    # output dataset.json
    for k, v in nifti_data_dict_by_modality.items():
        fname = osp.join(dataset_path, f"dataset-{k}.json")
        writeUnlabeledDatasetJson(v, fname, 
                                    meta=dataset_info, modality=k)
        print("write", fname)   

if __name__ == "__main__":
    storage_path_file = open(file_path, "r")
    storage_path_list = storage_path_file.readlines()
    storage_path_list = [s.strip('\n') for s in storage_path_list]

    for s_path in storage_path_list:
        if not osp.isdir(s_path):
            print("[MISS]", s_path)
            continue
        # create json
        sub_folders = os.listdir(s_path)
        if("CT" in sub_folders and "MR" in sub_folders)  or osp.basename(s_path) in EXTRA_LIST["TCIA"]:
            print("TCIA:", s_path)
            if(not debug_mode): process_tcia(s_path)
        elif(osp.basename(s_path).startswith("ds") or osp.basename(s_path) in EXTRA_LIST["openNeuro"]):
            print("OPENNEURO:", s_path)
            if(not debug_mode): process_openneuro(s_path)
        else:
            print("UNKNOWN:", s_path)
            meta=dict(modality=dataset2modality[osp.basename(s_path)])
            if(not debug_mode): process_unknown(s_path, meta)


