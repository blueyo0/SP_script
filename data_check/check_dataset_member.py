# -*- encoding: utf-8 -*-
'''
@File    :   check_dataset_member.py
@Time    :   2022/05/16 14:38:32
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   根据分工表，确认现有数据集的分工情况
'''

import os
import os.path as osp
import glob

dataset_path = "/mnt/cache/wanghaoyu/preprocess/data/dataset_name.txt"
member_path = "/mnt/cache/wanghaoyu/preprocess/data/unlabel_members.txt"

def getdirsize(dir):
    size = 0
    for root, dirs, files in os.walk(dir):
        size += sum([osp.getsize(osp.join(root, name)) for name in files])
    return size

if __name__ == "__main__":
    dataset_files = open(dataset_path, "r")
    dataset_list = dataset_files.readlines()
    dataset_list = [s.strip('\n') for s in dataset_list]

    # decode member relations
    member_files = open(member_path, "r")
    member_list =member_files.readlines()
    member_list = [s.strip('\n') for s in member_list]
    dataset2member = dict()
    for member_str in member_list:
        d, m = member_str.split("\t")
        dataset2member[d] = m

    # check for exisitence
    for dataset in dataset_list:
        if(dataset in dataset2member.keys()):
            print(dataset2member[dataset])
        else:
            print("NA")

    # check for storage size
    # size_list = []
    # for s_path in storage_path_list:
    #     if not osp.isdir(s_path): 
    #         print("[MISS]", s_path)
        # else:
        #     print("[VALID]", s_path)

    #         s_size = getdirsize(s_path) / (1024**3)
    #         size_list.append(dict(
    #             path=s_path,
    #             size=s_size
    #         ))
    # for s in size_list:
    #     print(osp.basename(s["path"]), "{:.2f}G".format(s["size"]))
    print("end")

