# -*- encoding: utf-8 -*-
'''
@File    :   Untitled-1
@Time    :   2022/08/17 16:40:56
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   参考@伍谦的合并代码，对total_seg做合并
'''

from cgi import test
import os
import glob
import SimpleITK as sitk
import numpy as np


""" 第一步：挑一个数据构建标签映射表"""
# check_data = data_list[0]  # 挑第0个
# check_data_seg_path = os.path.join(raw_data_path, check_data, "segmentations")
# check_data_seg_path = "C:\\Users\\Lenovo\\Downloads\\KIPA\\5_0000_fast"
check_data_seg_path = "/mnt/petrelfs/wanghaoyu/gmai/nnUNet_raw_data_base/totalseg_result/Task011_BTCV/ABD_001_0000_normal"
labels = sorted(os.listdir(check_data_seg_path))
assert len(labels) == 104  # 确保这个标签104个类别都有
label_map_dict = {}  # 标签名到标签号的映射
label_id_to_name_dict = {
    "0": "background",
}

for id, label in enumerate(labels):
    label_id = id + 1  # 所有标签从1开始排，因为0给背景类别
    label_name = label.split(".")[0]  # 标签名
    label_map_dict[label_name] = label_id  # 构建标签名到label_id的映射
    label_id_to_name_dict[str(label_id)] = label_name


def ensemble_all_labels_into_one(path, label_map_dict, save_name):
    """
    将104个nii.gz标签文件合成一个
    """
    label_search_path = os.path.join(path, "*nii.gz")
    labels = glob.glob(label_search_path)
    example_label_path = labels[0]
    example_label_itk = sitk.ReadImage(example_label_path)
    example_label_array = sitk.GetArrayFromImage(example_label_itk).astype(np.int16)
    final_label = np.zeros_like(example_label_array)
    for label in labels:
        label_name = label.split("/")[-1].split(".")[0]
        label_name = os.path.basename(label).split(".")[0]
        label_id = label_map_dict[label_name]
        this_label_itk = sitk.ReadImage(label)
        this_label_array = sitk.GetArrayFromImage(this_label_itk).astype(np.int16)
        # print(this_label_array)
        this_label_array = np.flip(this_label_array, axis=2)
        index = np.where(this_label_array > 0)
        final_label[index] = label_id
        # this_label = sitk.ReadImage(label)
    # print(final_label)
    final_label = sitk.GetImageFromArray(final_label)
    final_label.SetSpacing(example_label_itk.GetSpacing())
    final_label.SetOrigin(example_label_itk.GetOrigin())
    final_label.SetDirection(example_label_itk.GetDirection())
    sitk.WriteImage(final_label, save_name)
    print("saving: {}".format(save_name))
    
if __name__ == "__main__":
    test_path_list = [
        "/mnt/petrelfs/wanghaoyu/gmai/nnUNet_raw_data_base/totalseg_result/Task030_CT_ORG/volume-104_0000_normal",
    ]
    for test_path in test_path_list:
        save_name = test_path+".nii.gz"
        ensemble_all_labels_into_one(test_path, label_map_dict, save_name=save_name)
        print("ensemble", save_name)
        # print(label_id_to_name_dict)
        
    
    