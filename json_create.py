# -*- encoding: utf-8 -*-
'''
@File    :   json_create.py
@Time    :   2022/05/05 15:29:30
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   根据MSD格式的文件夹，生成dataset.json
'''

import json
import glob
import os.path as osp

dataset_info = {"description": "",
                "labels": {
                    "0": "background",
                    "1": "lesion",
                },
                "licence": "Creative Commons Attribution 4.0 International License",
                "modality": {
                    "0": "CT"
                },
                "name": "COVID-19-20",
                "reference": "",
                "release": "0.0",
                "tensorImageSize": "4D",}

data_root = "/mnt/lustre/share_data/gmai/dataset/preprocessed/labeled/COVID-19-20_v2"

if __name__ == "__main__":
    tr_case_list = []
    tr_img_list = glob.glob(osp.join(data_root, "imagesTr/*"))    
    # check for label
    for img in tr_img_list:
        fname = osp.basename(img)
        label_fname = osp.join(data_root, "labelsTr", fname)
        if osp.exists(label_fname):
            tr_case_list.append(
                {
                    "image": "./imagesTr/"+fname,
                    "label": "./labelsTr/"+fname
                }
            )
        else:
            print("Cannot find", label_fname)

    ts_case_list = []
    ts_img_list = glob.glob(osp.join(data_root, "imagesTs/*"))
    for img in ts_img_list:
        fname = osp.basename(img)
        ts_case_list.append(
            {
                "image": "./imagesTs/"+fname,
            }
        )

    dataset_info["numTraining"] = len(tr_case_list)
    dataset_info["numTest"] = len(ts_case_list)
    dataset_info["training"] = tr_case_list
    dataset_info["test"] = ts_case_list

with open(osp.join(data_root, "dataset.json"), "w") as fp:
    json.dump(dataset_info, fp, indent=4)






