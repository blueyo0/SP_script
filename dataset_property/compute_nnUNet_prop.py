# -*- encoding: utf-8 -*-
'''
@File    :   compute_nnUNet_prop.py
@Time    :   2022/11/08 18:42:52
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   分析nnUNet的imageTr内的所有数据的size和spacing
'''

import glob
import os.path as osp
import SimpleITK as sitk
import numpy as np

dataset_list = glob.glob(osp.join("/mnt/petrelfs/wanghaoyu/temp/dataset_analyze", "*"))
for dataset in dataset_list:
    print(osp.basename(dataset))
    images = glob.glob(osp.join(dataset, "imagesTr", "*.nii.gz"))
    data_prop = dict(all_spacings=[], all_sizes=[])
    for img_path in images:
        img = sitk.ReadImage(img_path)
        spacing = np.array(img.GetSpacing())
        size = np.array(img.GetSize())
        data_prop["all_spacings"].append(spacing)
        data_prop["all_sizes"].append(size)
    # print(data_prop['all_spacings'])
    # import pdb; pdb.set_trace();
    target_spacing = np.percentile(np.vstack([d[:3] for d in data_prop['all_spacings']]), 50, 0)
    target_size = np.percentile(np.vstack([d[:3] for d in data_prop['all_sizes']]), 50, 0)    
    print(target_spacing, "\t", target_size)