# -*- encoding: utf-8 -*-
'''
@File    :   compute_intensity.py
@Time    :   2022/06/06 22:53:04
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   内容说明
'''

import glob
import os.path as osp
import SimpleITK as sitk
import numpy as np
from torch import rand
import tqdm
import random
from monai import transforms

# path = "/mnt/cache/wanghaoyu/data/AbdomenCT5K/data2"
path = "/mnt/lustre/share_data/gmai/nnUNet_raw_data_base/amos22"



if __name__ == "__main__":
    load_trans = transforms.LoadImaged(keys=["image", "label"], allow_missing_keys=True,)

    files = glob.glob(osp.join(path, "*.nii.gz")) 
    # files = glob.glob(osp.join(path, "amos_0244.nii.gz")) 
    bound_list = []
    # random.shuffle(files)
    pbar = tqdm.tqdm(files)
    for f in pbar:
        try:
            image = load_trans(dict(image=f))
        except:
            print("error: ", f)
            continue
    bound = np.mean(bound_list, axis=0)
    print("end") 


