# -*- encoding: utf-8 -*-
'''
@File    :   fuzzy_match.py
@Time    :   2022/11/03 20:45:38
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   模糊匹配数据集名称
'''

from check_local import dataset_list
import os.path as osp
import json
from fuzzywuzzy import process
from batchgenerators.utilities.file_and_folder_operations import save_json

def fuzzySearch(key, labels):
    return process.extract(key, labels, limit=3)

def buildGlobalDatasetList(fname="dataset_name_list.txt"):
    file = open(fname, "r")
    dataset_name_list = []
    for idx, info in enumerate(file.readlines()):
        dataset_name_list.append(info)
    file.close()
    return dataset_name_list

if __name__=="__main__":
    dname_list = buildGlobalDatasetList()
    result = dict()
    for dataset in dataset_list:
        tag = dataset.split("_")[-1]
        most_likely_dname = fuzzySearch(tag, dname_list)
        result[dataset] = most_likely_dname
    save_json(result, osp.join("/mnt/cache/wanghaoyu/SP_script/fuzzy_match_datasetname/fuzzy_dataset_name_mapping.json"))

