# -*- encoding: utf-8 -*-
'''
@File    :   mapping_convert.py
@Time    :   2022/08/04 15:57:29
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   根据dataset.json 和 mapping 的信息，获取
'''



import json

def get_class_from_json(json_file, mapping):
    info = json.load(open(json_file, "r"))
    # for k, v 
    # info["labels"]


if __name__ == "__main__":
    mapping_file ="/mnt/cache/wanghaoyu/preprocess/data/joint_partial/mapping.json"
    mapping_info = json.load(open(mapping_file, "r"))
    mapping_per_dataset = {}
    for k, v in mapping_info.items():
        try:
            dataset, dataset_modality, dataset_label = k.split('-', 2)
            global_label, global_modality = v.split('-', 1)
        except:
            print(k)
            print(v)
            continue
        if(not dataset in mapping_per_dataset):
            mapping_per_dataset[dataset] = {}
        mapping_per_dataset[dataset][dataset_label] = global_label
    
    for k, v in mapping_per_dataset.items():
        print(k)
        print(v)





