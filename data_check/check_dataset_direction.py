# -*- encoding: utf-8 -*-
'''
@File    :   check_dataset_direction.py
@Time    :   2022/12/21 20:55:55
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   检查数据的方向
'''

import os.path as osp
from glob import glob
import SimpleITK as sitk

# check_dir = "/mnt/petrelfs/wanghaoyu/gmai/totalseg_tmp_data/raw_data/Task559_TS_test/imagesTr"
check_dir = "/mnt/petrelfs/wanghaoyu/gmai/totalseg_tmp_data/raw_data/Task011_BTCV_rotate/imagesTr"

img_list = glob(osp.join(check_dir, "*.nii.gz"))

direct_list = []
for img_path in img_list:
    direct = sitk.ReadImage(img_path).GetDirection()
    direct_list.append(list(direct))
    print(direct, img_path)

