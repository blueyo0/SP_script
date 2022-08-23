# -*- encoding: utf-8 -*-
'''
@File    :   create_subfolder.py
@Time    :   2022/06/29 22:16:54
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   将总数据集进行拆分和复制，默认不考虑多模态（假设每个image只有id_0000.nii.gz）
'''
import os.path as osp
import json
import copy
import shutil
import os
from multiprocessing import Pool

dataset_id = 350

prefix = "AbdomenTinySubset"
json_path = "/mnt/cache/wanghaoyu/data/RAW_DATA/Task096_Ab_tiny"
clip_freq = 2

# prefix = "AbdomenSubset"
# json_path = "/mnt/cache/wanghaoyu/data/RAW_DATA/Task099_AbdomenCT5K"
# clip_freq = 500

output_path = "/mnt/cache/wanghaoyu/data/RAW_DATA/"
default_num_threads = 16

if __name__ == "__main__":
    with open(osp.join(json_path, "dataset.json"), "r") as json_file:
        dataset_info = json.load(json_file)
        total_data_num = len(dataset_info["training"])
        for i in range(total_data_num//clip_freq + 1):
            sub_dataset_info = copy.deepcopy(dataset_info)
            up_bound = (i+1)*clip_freq
            if(up_bound > total_data_num-1): up_bound = total_data_num
            sub_dataset_info["training"] = sub_dataset_info["training"][i*clip_freq:up_bound]
            sub_dataset_info["test"] = sub_dataset_info["test"][:5]
            sub_dataset_info["numTraining"] = len(sub_dataset_info["training"])
            sub_dataset_info["numTest"] = len(sub_dataset_info["test"])
            sub_dataset_info["name"] = f"{prefix}{i}"

            # create dataset_subfolder
            output_subfolder = osp.join(output_path, f"Task{dataset_id+i}_{prefix}{i}")
            if(not osp.exists(output_subfolder)): 
                os.makedirs(output_subfolder)
            if(not osp.exists(osp.join(output_subfolder, "imagesTr"))): 
                os.makedirs(osp.join(output_subfolder, "imagesTr"))
            if(not osp.exists(osp.join(output_subfolder, "labelsTr"))): 
                os.makedirs(osp.join(output_subfolder, "labelsTr"))
            if(not osp.exists(osp.join(output_subfolder, "imagesTs"))): 
                os.makedirs(osp.join(output_subfolder, "imagesTs"))

            def move_train(args):
                data_file, seg_file = args["image"], args["label"]
                # print("cp", osp.join(json_path, data_file), osp.join(output_subfolder, data_file))
                shutil.copyfile(osp.join(json_path, data_file.replace(".nii.gz", "_0000.nii.gz")), 
                                osp.join(output_subfolder, data_file.replace(".nii.gz", "_0000.nii.gz")))
                # print("cp", osp.join(json_path, seg_file), osp.join(output_subfolder, seg_file))
                shutil.copyfile(osp.join(json_path, seg_file), osp.join(output_subfolder, seg_file))
                return data_file
            def move_test(args):
                data_file = args
                # print("cp", osp.join(json_path, data_file), osp.join(output_subfolder, data_file))
                shutil.copyfile(osp.join(json_path, data_file.replace(".nii.gz", "_0000.nii.gz")), 
                                osp.join(output_subfolder, data_file.replace(".nii.gz", "_0000.nii.gz")))
                return data_file

            p = Pool(default_num_threads)
            train_ids = p.map(move_train, sub_dataset_info["training"])
            train_ids = p.map(move_test, sub_dataset_info["test"])
            p.close()
            p.join()
            with open(osp.join(output_subfolder, "dataset.json"), "w") as fp:
                json.dump(sub_dataset_info, fp, indent=4)

