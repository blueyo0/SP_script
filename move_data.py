# -*- encoding: utf-8 -*-
'''
@File    :   move_data.py
@Time    :   2022/06/28 13:58:50
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   following the 
'''


SRC_DIR = "/mnt/lustre/share_data/gmai/dataset/preprocessed/temp/Task99_AbdomenCT5K/"
LABEL_DIR = "/mnt/cache/wanghaoyu/data/RAW_DATA/Task099_AbdomenCT5K/labelsTr"
IMAGE_DIR = "/mnt/cache/wanghaoyu/data/RAW_DATA/Task099_AbdomenCT5K/imagesTr"

import os.path as osp
import glob
import shutil
import os

if __name__ == "__main__":
    image_list = glob.glob(osp.join(IMAGE_DIR, "*.nii.gz"))
    for image in image_list:
        identifier = osp.basename(image).split("_0000.nii")[0]
        src_filename = osp.join(SRC_DIR, "labelsTr", identifier+".nii.gz")
        target_filename = osp.join(LABEL_DIR, identifier+".nii.gz")
        # if(label.startswith("CPTAC-PDA")):
        #     shutil.copyfile(label, target_filename)
        if(osp.exists(target_filename)): continue
        elif(osp.exists(src_filename)):
            print("find", src_filename)
            shutil.copyfile(src_filename, target_filename)
        else:
            print("fail to find", src_filename)
            # os.remove(label)

