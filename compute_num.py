from typing import OrderedDict
from git import HEAD


# -*- encoding: utf-8 -*-
'''
@File    :   compute_num.py
@Time    :   2022/05/08 18:28:22
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   读取json, 统计data 数量
'''
import json
import glob
import os.path as osp
data_list_file = None

data_dir = "/mnt/lustre/share_data/gmai/dataset/preprocessed/unlabeled"
# data_list_file = "/mnt/cache/wanghaoyu/preprocess/data/uploaded_pre_path_tcia.txt"
output_dir = "/mnt/cache/wanghaoyu/preprocess/data/result"
output_prefix = ""

def output_details(data_count_dict, filepath=None):
    if(filepath): f = open(filepath, "w")
    for k, v in sorted(data_count_dict.items(), key=lambda d: d[0]):
        if(k=="all"): continue 
        print(f"[{k}] {v}")
        if(filepath): f.write(f"[{k}] {v}" + "\n")
    print(f"[ALL] {data_count_dict['all']}")
    if(filepath): f.write(f"[ALL] {data_count_dict['all']}" + "\n")

def compute_by_modality(dataset_path_list, reduce=None):
    # by modality
    data_count_dict = {"all":0}
    for dataset_path in dataset_path_list:
        json_list = glob.glob(osp.join(dataset_path, "dataset-*.json"))
        for json_file in json_list:
            info = json.load(open(json_file, 'r'))
            tr_num = info["numTraining"]
            modality = osp.basename(json_file).split("-")[-1].split(".")[0]
            if not (modality in data_count_dict.keys()):
                data_count_dict[modality] = tr_num
            else:
                data_count_dict[modality] += tr_num
            data_count_dict["all"] += tr_num
    if(reduce=="CT-MR"):
        # reduce unknown modality to MR
        new_data_count_dict = dict(CT=0, MR=0, all=data_count_dict["all"])
        for k, v in data_count_dict.items():
            if(k=="all"): continue 
            elif(k=="CT"): new_data_count_dict["CT"] = v
            else:
                new_data_count_dict["MR"] += v
        data_count_dict = new_data_count_dict
    return data_count_dict

def compute_by_dataset(dataset_path_list):
    # by dataset
    data_count_dict = {"all":0}
    for dataset_path in dataset_path_list:
        json_list = glob.glob(osp.join(dataset_path, "dataset-*.json"))
        for json_file in json_list:
            info = json.load(open(json_file, 'r'))
            tr_num = info["numTraining"]
            dataset_name = osp.basename(osp.dirname(json_file))
            if not (dataset_name in data_count_dict.keys()):
                data_count_dict[dataset_name] = tr_num
            else:
                data_count_dict[dataset_name] += tr_num
            data_count_dict["all"] += tr_num
    return data_count_dict

def compute_by_dataset_modality(dataset_path_list, reduce=None):
    # by dataset
    data_count_dict = {"all":0}
    for dataset_path in dataset_path_list:
        result_per_dataset = compute_by_modality([dataset_path], reduce=reduce)
        dataset_name = osp.basename(dataset_path)
        for k, v in result_per_dataset.items():
            if(k=="all"): continue
            data_count_dict[dataset_name+"_"+k] = v
        data_count_dict["all"] += result_per_dataset["all"]
    return data_count_dict


if __name__ == "__main__":
    if(data_list_file):
        data_list_file = open(data_list_file, "r")
        storage_path_list = data_list_file.readlines()
        dataset_path_list = [s.strip('\n') for s in storage_path_list]
        print("find valid dataset {}".format(len(dataset_path_list)))
    else:
        print("search for datasets in", data_dir)
        dataset_path_list_all = glob.glob(osp.join(data_dir, "*"))
        dataset_path_list = []
        for d in dataset_path_list_all:
            info_list = glob.glob(osp.join(d, "dataset-*.json"))
            if(len(info_list)>0): dataset_path_list.append(d)
            else: print("[OMIT]", d)
        print("find valid dataset {}/{}".format(len(dataset_path_list), len(dataset_path_list_all)))
    
    result = dict()
    result["dataset"] = compute_by_dataset(dataset_path_list)
    result["dataset_modality"] = compute_by_dataset_modality(dataset_path_list, reduce="CT-MR")
    result["modality_raw"] = compute_by_modality(dataset_path_list)
    result["modality"] = compute_by_modality(dataset_path_list, reduce="CT-MR")

    output_details(result["modality_raw"], filepath=osp.join(output_dir, output_prefix+"result_by_modality_raw.txt"))
    output_details(result["dataset"], filepath=osp.join(output_dir, output_prefix+"result_by_dataset.txt"))
    output_details(result["dataset_modality"], filepath=osp.join(output_dir, output_prefix+"result_by_dataset_modality.txt"))
    output_details(result["modality"], filepath=osp.join(output_dir, output_prefix+"result_by_modality.txt"))
