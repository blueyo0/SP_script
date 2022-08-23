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

path = "/mnt/cache/wanghaoyu/data/AbdomenCT5K/data"

def compute_stats(voxels):
    if len(voxels) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    # ignore other information
    # median = np.median(voxels)
    # mean = np.mean(voxels)
    # sd = np.std(voxels)
    # mn = np.min(voxels)
    # mx = np.max(voxels)
    median = 0
    mean = 0
    sd = 0
    mn = 0
    mx = 0
    percentile_99_5 = np.percentile(voxels, 99.5)
    percentile_00_5 = np.percentile(voxels, 00.5)
    return median, mean, sd, mn, mx, percentile_99_5, percentile_00_5



if __name__ == "__main__":
    files = glob.glob(osp.join(path, "*.nii.gz")) 
    bound_list = []
    random.shuffle(files)
    files = files[:500]
    pbar = tqdm.tqdm(files)
    for f in pbar:
        try:
            image = sitk.ReadImage(f)
            voxel = sitk.GetArrayFromImage(image)
            median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = compute_stats(voxel)
            bound_list.append([percentile_00_5, percentile_99_5])
        except:
            print("error: ", f)
            continue
    bound = np.mean(bound_list, axis=0)
    print("end") 


