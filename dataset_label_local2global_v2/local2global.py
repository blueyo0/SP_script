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
import pandas as pd
import numpy as np

def standardize(cls_Name):
    return cls_Name.strip().lower().replace("_", " ").replace("-", " ")

def getGlobalKey(key, dataset=None, mapping=CUSTUM_MAPPING):
    key = str(key)
    dataset = str(dataset)
    if(dataset in mapping):
        # norm_keys = [standardize(k) for k in mapping[dataset].keys()]
        if(key in mapping[dataset]):
            trans_key = mapping[dataset][key]
            if(trans_key in mapping["global"]): trans_key = mapping["global"][trans_key]
            return trans_key
    if(key in mapping["global"]):
        return mapping["global"][key]
    return None


def buildGlobalLabelSys(fname="data/full_labels_final.csv"):
    file = pd.read_csv(fname)
    label_sys = defaultdict(dict)
    for meta_info in file.itertuples():
        cls_idx, cls_info = meta_info[1], meta_info[2]
        modality, cls_name = [standardize(i) for i in cls_info.split("-", maxsplit=1)]
        label_sys[cls_name][modality.upper()] = cls_idx
    return label_sys

def buildGlobalIdx2Label(fname="data/full_labels_final.csv"):
    file = pd.read_csv(fname)
    gidx2label = defaultdict(str)
    for meta_info in file.itertuples():
        cls_idx, cls_info = meta_info[1], meta_info[2]
        gidx2label[cls_idx] = cls_info.strip()
    return gidx2label


def fuzzySearch(key, labels):
    return process.extract(key, labels, limit=3)
    
if __name__ == "__main__":
    global_label_sys = buildGlobalLabelSys()
    global_idx2label = buildGlobalIdx2Label()
    for k, v in global_label_sys.items():
        print(k, v)

    data_dir = "/mnt/petrelfs/wanghaoyu/why/local_label"
    json_all = {}
    for dataset in dataset_list:
        json_per_dataset = {}
        json_path =  osp.join(data_dir, dataset, "dataset.json")
        if(not osp.exists(json_path)):
            print(json_path, "not found")
            json_all[dataset] = "cannot find dataset.json"
            continue
        json_data = json.load(open(json_path))
        if("labels" not in json_data):
            print("skip unlabeled data:", json_path)
            json_all[dataset] = "cannot find key 'labels' in dataset.json, this dataset may be unlabeled"
            continue
        json_label = json_data["labels"] 
        print("dataset:", dataset)
        # print(json_label.values())
        for idx, k in json_label.items():
            ori_key = k
            k = str(k)
            if(k.endswith("_L")): k = k.replace("_L", "_left")
            if(k.endswith("_R")): k = k.replace("_R", "_right")
            query_k = standardize(k)
            # [WARING] multi-modality
            modality = dataset_modality_mapping[dataset][-1]
            if(query_k=='background' or query_k=='0'): 
                json_per_dataset[idx] = 0
                continue
            fs_key = fuzzySearch(query_k, global_label_sys.keys())
            is_find = False
            
            if(query_k==fs_key[0][0]): # 直接找到相同的
                # print("find", k)
                if(modality.upper() not in global_label_sys[query_k]):
                    g_key = getGlobalKey(ori_key, dataset)
                    if(g_key):
                        query_k = standardize(g_key)
                        if(modality.upper() not in global_label_sys[query_k]):
                            print(ori_key, modality.upper(), "not found")
                            json_per_dataset[idx] = "cannot find "+str(ori_key)+"("+str(modality.upper())+")"
                            continue
                        json_per_dataset[idx] = global_label_sys[query_k][modality.upper()]
                        is_find = True
                    else:
                        json_per_dataset[idx] = "cannot find "+str(ori_key)+"("+str(modality.upper())+")"
                        print("[MODALITY]", k, modality.upper(), "not found")
                        continue
                json_per_dataset[idx] = global_label_sys[query_k][modality.upper()]
                is_find = True
            else:  # 通过global映射找匹配的
                # try to mapping key to global key
                g_key = getGlobalKey(ori_key, dataset)
                if(g_key):
                    query_k = standardize(g_key)
                    if(modality.upper() not in global_label_sys[query_k]):
                        print(ori_key, modality.upper(), "not found")
                        json_per_dataset[idx] = "cannot find "+str(ori_key)+"("+str(modality.upper())+")"
                        continue
                    json_per_dataset[idx] = global_label_sys[query_k][modality.upper()]
                    is_find = True

                
            if(not is_find): 
                print("[ATTEN]", k, modality.upper(), "->", fs_key)
                json_per_dataset[idx] = "cannot find "+str(ori_key)
            else:
                l_idx = json_per_dataset[idx]
                try:
                    g_idx = global_label_sys[query_k][modality.upper()]
                except:
                    import pdb; pdb.set_trace()
                    
                g_key = global_idx2label[int(g_idx)]
                print(f"{ori_key}({l_idx})", "=", f"{g_key}({g_idx})")
        
        print("json:", json_per_dataset)
        json_all[dataset] = json_per_dataset
    # 特殊处理
    json_all["Task626_VerSe19"] = json_all["Task083_VerSe2020"]
    json_all["Task619_VESSEL2012"]["0"] = 0
       
        
    save_json(json_all, osp.join("/mnt/cache/wanghaoyu/SP_script/dataset_label_local2global_v2/data/labeled_label_sys.json"))

    json_all_data = []
    for dataset, idx_info in json_all.items():
        label_info = [np.nan]*len(global_idx2label.keys())
        if(isinstance(idx_info, str)):
            label_info[0] = idx_info
            json_all_data.append([dataset]+label_info)
        elif(isinstance(idx_info, dict)):
            label_info[0] = 1
            for idx, g_idx in idx_info.items():
                if(isinstance(g_idx, str)): continue
                label_info[g_idx] = 1
            json_all_data.append([dataset]+label_info)
        else:
            print("ERROR in", dataset, "type:", type(idx_info))
            print(idx_info)
    json_all_df = pd.DataFrame(json_all_data, columns=["dataset"]+list(global_idx2label.values()))
    json_all_df.to_csv("/mnt/cache/wanghaoyu/SP_script/dataset_label_local2global_v2/data/labeled_label_sys.csv")
