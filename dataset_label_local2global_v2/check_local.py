# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   20220919 10:09:36
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   check的主文件
'''
import os.path as osp
import pandas as pd

meta_infos = pd.read_csv("data/datasets_with_json.csv")
dataset_list = list()
dataset_modality_mapping = dict()
for info in meta_infos.itertuples():
    dataset_names = info[3]
    modality_names = info[5]
    # print(modality_names)
    modality = tuple(m.strip().upper() for m in modality_names.split(','))
    for name in dataset_names.split(","):
        dataset_list.append(name.strip())
        dataset_modality_mapping[name.strip()] = modality

