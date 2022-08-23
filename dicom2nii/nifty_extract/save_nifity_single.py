#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   save_nifty.py
@Time    :   2022/02/18 14:51:38
@Author  :   Sun Hui 
@Version :   1.0
@Contact :   bitsunhui@163.com
@License :   (C)Copyright 2022-2028, uni-medical
@Desc    :   None
'''

import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
from multiprocessing import Pool


def save_dcm_to_nii_gz(parmas):
    dcm_dir, save_dir, series_uid, dcm_name, idx, total_num = parmas
    print('{}/{}, name: {}'.format(idx, total_num, dcm_dir))

    series_reader = sitk.ImageSeriesReader()
    serie_names = series_reader.GetGDCMSeriesFileNames(dcm_dir, series_uid)
    series_reader.SetFileNames(serie_names)
    simage = series_reader.Execute()
    sitk.WriteImage(simage, save_dir)


if __name__ == '__main__':
    # dataset, hk-test
    info_csv_file_dir = r"/mnt/lustre/wanghaoyu/dataset/temp/jqn_tmp/private/unpack/Pelvic-Reference-Data/meta-info.csv" #r'/home/PJLAB/niujingqi/data_extra/TCIA/DATA/ACRIN_6657/meta-info.csv'
    save_info_csv_file_dir = os.path.join(r'/mnt/lustre/wanghaoyu/dataset/temp/jqn_tmp/private/unpack/Pelvic-Reference-Data',  'save-nifty-meta-info.csv' ) #ACRIN_6657/save-nifty-meta-info.csv'
    save_nii_dir = r'/mnt/lustre/wanghaoyu/dataset/temp/jqn_tmp/private/unpack/Pelvic-Reference-Data' 
    save_nii_dir_MR = os.path.join(save_nii_dir, 'MR')
    save_nii_dir_CT = os.path.join(save_nii_dir, 'CT')
    os.makedirs(save_nii_dir_MR, exist_ok= True)
    os.makedirs(save_nii_dir_CT, exist_ok= True)
    dataset_suffix = save_nii_dir.split('/')[-1]

    # filter conditions, re-fill according to the situation
    def condition_fn1(x):
        return True if '0.6' in str(x['Series Description (0008|103e)']).lower() else False
    min_MR_slice_num =10
    min_CT_slice_num = 50
    def condition_fn2(x):
        # print(int(x['SerieSliceNum']))
        if x['Modality (0008|0060)'] == 'MR':
            return True if int(x['SerieSliceNum']) > min_MR_slice_num else False
        elif x['Modality (0008|0060)'] == 'CT':
            return True if int(x['SerieSliceNum']) > min_CT_slice_num else False

    info_df = pd.read_csv(info_csv_file_dir)
    keep_list = []
    for i, row in info_df.iterrows():
        # print(row)
        # print()
        if  condition_fn2(row):
            keep_list.append(1)
        else:
            keep_list.append(0)
    info_df['is_keep'] = keep_list
    # print(keep_list)
    keep_info_df = info_df[info_df['is_keep'] == 1]
    keep_info_df = keep_info_df.drop(columns=['is_keep'])

    # save to nifty
    params_list = []
    nii_save_name_list = []
    total_num = len(keep_info_df)
    for i, row in keep_info_df.iterrows():
        # print(i)
        idx = i
        dcm_dir = str(row['DicomDir'])
        dcm_name = dcm_dir.split('/')[-1]
        series_uid = str(row['Series Instance UID (0020|000e)'])
        save_name = '{}-{}.nii.gz'.format(dataset_suffix, series_uid)
        save_dir = os.path.join(save_nii_dir, save_name)
        if row['Modality (0008|0060)'] == 'MR':
            save_dir = os.path.join(os.path.join(save_nii_dir, save_nii_dir_MR), save_name)
            print(os.path.join(save_nii_dir, save_nii_dir_CT))
            print(save_dir)
        if row['Modality (0008|0060)'] == 'CT':
            save_dir = os.path.join(os.path.join(save_nii_dir, save_nii_dir_CT), save_name)
            print(os.path.join(save_nii_dir, save_nii_dir_CT))
            print(save_dir)
        
        params_list.append(
            [dcm_dir, save_dir, series_uid, dcm_name, idx, total_num])
        nii_save_name_list.append(save_name)
        # print(nii_save_name_list)

    save_keep_info_df = keep_info_df.copy()
    save_keep_info_df['NIfty Save Name'] = nii_save_name_list
    save_keep_info_df.to_csv(save_info_csv_file_dir, index=False)
    pool = Pool(processes=4)
    # print(params_list)
    try:
        pool.map(save_dcm_to_nii_gz, params_list)
    except Exception as e:
        print('the error',e.__class__.__name__,e)
    pool.close()
