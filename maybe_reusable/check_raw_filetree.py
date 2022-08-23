from git import HEAD


# -*- encoding: utf-8 -*-
'''
@File    :   check_raw_filetree.py
@Time    :   2022/05/17 09:32:36
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   简单搜索文件结构
'''

import os
import os.path as osp
import glob

dataset_path = "/mnt/cache/wanghaoyu/preprocess/data/tobeprocess_raw_path.txt"
output_dir = "/mnt/cache/wanghaoyu/preprocess/data/result"

if __name__ == "__main__":
    dataset_files = open(dataset_path, "r")
    dataset_list = dataset_files.readlines()
    dataset_list = [s.strip('\n') for s in dataset_list]

    output_file = open(osp.join(output_dir, "tobeprocess_status.txt"), "w")
    for dataset in dataset_list:
        folder_depth1 = os.listdir(dataset)
        if ("imagesTr" in folder_depth1 and "labelsTr" in folder_depth1):
            print("[MSD]", dataset)
        else:
            output_file.write(""+dataset+"\n")
            for folder in folder_depth1:
                output_file.write("|___"+folder+"\n")
            output_file.write("\n")
    output_file.close()  




