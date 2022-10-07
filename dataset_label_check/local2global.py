# -*- encoding: utf-8 -*-
'''
@File    :   local2global.py
@Time    :   2022/09/22 16:15:01
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   local label mapping to global label
'''

from collections import defaultdict
from check_local import dataset_list, dataset_modality_mapping
import os.path as osp
import json
from fuzzywuzzy import process
from local2global_mapping import CUSTUM_MAPPING
from batchgenerators.utilities.file_and_folder_operations import save_json

def standardize(cls_Name):
    return cls_Name.strip().lower().replace("_", " ").replace("-", " ")

def getGlobalKey(key, dataset=None, mapping=CUSTUM_MAPPING):
    if(dataset in mapping):
        norm_keys = [standardize(k) for k in mapping[dataset].keys()]
        if(key in mapping[dataset]):
            return mapping[dataset][key]
    if(key in mapping["global"]):
        return mapping["global"][key]
    return None


def buildGlobalLabelSys(fname="curr_global_labels.txt"):
    file = open(fname, "r")
    label_sys = defaultdict(dict)
    for cls_idx, cls_info in enumerate(file.readlines()):
        modality, cls_name = [standardize(i) for i in cls_info.split("-", maxsplit=1)]
        label_sys[cls_name][modality.upper()] = cls_idx
    file.close()
    return label_sys

def buildGlobalIdx2Label(fname="curr_global_labels.txt"):
    file = open(fname, "r")
    gidx2label = defaultdict(str)
    for cls_idx, cls_info in enumerate(file.readlines()):
        gidx2label[cls_idx] = cls_info.strip()
    file.close()
    return gidx2label


def fuzzySearch(key, labels):
    return process.extract(key, labels, limit=3)
    
if __name__ == "__main__":
    global_label_sys = buildGlobalLabelSys()
    global_idx2label = buildGlobalIdx2Label()
    # for k, v in global_label_sys.items():
    #     print(k, v)

    data_dir = "/mnt/petrelfs/wanghaoyu/why/local_label"
    json_all = {}
    for dataset in dataset_list:
        json_per_dataset = {}
        json_path =  osp.join(data_dir, dataset, "dataset.json")
        json_label = json.load(open(json_path))["labels"] 
    

        print("dataset:", dataset)
        for idx, k in json_label.items():
            ori_key = k
            k = str(k)
            if(k.endswith("_L")): k = k.replace("_L", "_left")
            if(k.endswith("_R")): k = k.replace("_R", "_right")
            query_k = standardize(k)
            modality = dataset_modality_mapping[dataset]
            if(query_k=='background'): 
                json_per_dataset[idx] = 0
                continue
            fs_key = fuzzySearch(query_k, global_label_sys.keys())
            is_find = False
            if(query_k==fs_key[0][0]):
                # print("find", k)
                if(modality.upper() not in global_label_sys[k]):
                    # print(k, modality.upper(), "not found")
                    continue
                json_per_dataset[idx] = global_label_sys[query_k][modality.upper()]
                is_find = True
            else:
                # try to mapping key to global key
                g_key = getGlobalKey(query_k, dataset)
                if(g_key):
                    fs_key = fuzzySearch(g_key, global_label_sys.keys())
                    if(query_k==fs_key[0][0]):
                        if(modality.upper() not in global_label_sys[k]):
                            # print(k, modality.upper(), "not found")
                            continue
                        json_per_dataset[idx] = global_label_sys[query_k][modality.upper()]
                        is_find = True
            if(not is_find): 
                print("[ATTEN]", k, modality.upper(), "->", fs_key)
                json_per_dataset[idx] = ori_key+" not found"
            else:
                l_idx = json_per_dataset[idx]
                g_idx = global_label_sys[query_k][modality.upper()]
                g_key = global_idx2label[int(g_idx)]
                print(f"{ori_key}({l_idx})", "=", f"{g_key}({g_idx})")
        
        print("json:", json_per_dataset)
        json_all[dataset] = json_per_dataset
    save_json(json_all, osp.join("/mnt/cache/wanghaoyu/SP_script/dataset_label_check/global_label_sys.json"))

