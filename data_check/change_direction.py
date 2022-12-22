# -*- encoding: utf-8 -*-
'''
@File    :   set_direction.py
@Time    :   2022/12/21 21:37:46
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   修改数据的方向
'''


import os.path as osp
from glob import glob
import SimpleITK as sitk
from tqdm import tqdm

dataset_list = [
    # {
    #     "in_dir" : "/mnt/petrelfs/wanghaoyu/gmai/totalseg_tmp_data/raw_data/Task011_BTCV/imagesTr",
    #     "out_dir" : "/mnt/petrelfs/wanghaoyu/gmai/totalseg_tmp_data/raw_data/Task011_BTCV_rotate/imagesTr",        
    # },
    {
        "in_dir" : "/mnt/petrelfs/wanghaoyu/gmai/totalseg_tmp_data/raw_data/Task032_AMOS22_Task1/imagesTr",
        "out_dir" : "/mnt/petrelfs/wanghaoyu/gmai/totalseg_tmp_data/raw_data/Task032_AMOS22_Task1_rotate/imagesTr",     

    }
]

for dataset in dataset_list:
    in_dir = dataset["in_dir"]
    out_dir = dataset["out_dir"]
    img_list = glob(osp.join(in_dir, "*.nii.gz"))
    for img_path in tqdm(img_list):
        img = sitk.ReadImage(img_path)
        # img.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
        # resampled_image = sitk.Resample(img, img.GetSize(), sitk.Transform(), sitk.sitkLinear,
        #                                 img.GetOrigin(), img.GetSpacing(), 
        #                                 (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0), # new direction
        #                                 0, img.GetPixelID())
        resampled_image = sitk.GetImageFromArray(sitk.GetArrayFromImage(img))
        resampled_image.SetOrigin(img.GetOrigin())
        resampled_image.SetSpacing(img.GetSpacing())
        resampled_image.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
        out_path = osp.join(out_dir, osp.basename(img_path))
        # sitk.WriteImage(img, out_path)
        sitk.WriteImage(resampled_image, out_path)
        print(resampled_image.GetDirection(), out_path)
    print("ATTEN", osp.basename(osp.dirname(in_dir)), "is handled")