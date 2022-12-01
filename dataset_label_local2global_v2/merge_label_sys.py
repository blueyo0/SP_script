# -*- encoding: utf-8 -*-
'''
@File    :   merge_label_sys.py
@Time    :   2022/11/30 10:09:16
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   将人工填写的unlabeled_label_sys.csv
             和统计的labeled_label_sys.csv合并成
             global_label_sys_final.csv
'''
import pandas as pd
from collections import defaultdict
from local2global import standardize
import numpy as np

def buildLabel2GlobalIdx(fname="data/full_labels_final.csv"):
    file = pd.read_csv(fname)
    label2gidx = dict()
    for meta_info in file.itertuples():
        cls_idx, en_cls_name, ch_cls_name = int(meta_info[1]), \
                                            standardize(meta_info[2]), \
                                            standardize(meta_info[3])
        label2gidx[en_cls_name] = cls_idx
        label2gidx[ch_cls_name] = cls_idx
    return label2gidx

def buildGlobalLabelSys(fname="data/full_labels_final.csv"):
    file = pd.read_csv(fname)
    # ch_cls_name_list = file["label-CN"].values.tolist()
    general_cls_name_list = []
    for meta_info in file.itertuples():
        en_cls_name, ch_cls_name = standardize(meta_info[2].split('-')[-1]), standardize(meta_info[3].split('-')[-1])
        modality = meta_info[2].split('-')[0].upper()
        general_cls_name = f"{modality}-{ch_cls_name}({en_cls_name})"
        general_cls_name_list.append(general_cls_name)
    return general_cls_name_list


if __name__ == "__main__":
    label_sys = buildGlobalLabelSys()
    label2gidx = buildLabel2GlobalIdx()
    global_label_sys_final = []
    labeled_info = pd.read_csv("data/labeled_label_sys.csv")
    unlabeled_info = pd.read_csv("data/unlabeled_label_sys.csv")
    todo_data = [labeled_info, unlabeled_info]
    for data in todo_data:
        # get the labels of existed index -> mapping to global label -> create global_info
        for dataset_info in data.itertuples():
            dataset_name = dataset_info[2]
            dataset_global_info = [np.nan]*len(label_sys)
            dataset_global_info[0] = int(1)
            for idx in range(3, len(dataset_info)):
                if(isinstance(dataset_info[idx], str)):
                    dataset_global_info[idx-3] = dataset_info[idx] 
                if(dataset_info[idx]==1):
                    local_label = standardize(data.columns[idx-1])
                    if(local_label not in label2gidx):
                        print("cannot find", local_label)
                        continue
                    global_idx = label2gidx[local_label]
                    dataset_global_info[global_idx] = int(1)
                    print("find", local_label)
            global_label_sys_final.append([dataset_name]+dataset_global_info)
    global_label_df = pd.DataFrame(global_label_sys_final, columns=["dataset"]+list(label_sys))
    global_label_df.to_csv("/mnt/cache/wanghaoyu/SP_script/dataset_label_local2global_v2/data/global_label_sys_final.csv")


