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
from tqdm import tqdm
import dicom2nifti
def save_dcm_to_nii_gz(parmas):
    dcm_dir, save_dir, series_uid, dcm_name, idx, total_num = parmas
    # print('{}/{}, name: {}'.format(idx, total_num, dcm_dir))
    # dcm_dir = repr(dcm_dir)
    # print(series_uid)
    dicom2nifti.convert_directory( dcm_dir, save_dir,series_uid, compression=True, reorient=True)
    # series_reader = sitk.ImageSeriesReader()
    # print('dcm_dir, series_uid',dcm_dir, series_uid)
    # serie_names = series_reader.GetGDCMSeriesFileNames(dcm_dir, series_uid)
    # series_reader.SetFileNames(serie_names)
    # simage = series_reader.Execute()
    # sitk.WriteImage(simage, save_dir)


if __name__ == '__main__':
    # dataset, hk-test
    info_csv_file_dir = '/mnt/lustre/wanghaoyu/dataset/temp/jqn_tmp/private/unpack/AAPM-RT-MAC/meta-info.csv'
    save_info_csv_file_dir = '/mnt/lustre/wanghaoyu/dataset/temp/jqn_tmp/Nifity_all/AAPM-RT-MAC/save-nifty-meta-info.csv'
    save_nii_dir = '/mnt/lustre/wanghaoyu/dataset/temp/jqn_tmp/Nifity_all/AAPM-RT-MAC'
    os.makedirs(save_nii_dir, exist_ok=True)
    dataset_suffix = 'NSCLC_Radiogenomics'
    if not os.path.exists(os.path.join(save_nii_dir, 'CT')):
        os.mkdir(os.path.join(save_nii_dir, 'CT'))
    CT_dir = os.path.join(save_nii_dir, 'CT')
    if not os.path.exists(os.path.join(save_nii_dir, 'MR')):
        os.mkdir(os.path.join(save_nii_dir, 'MR'))
    MR_dir = os.path.join(save_nii_dir, 'MR')
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
        if str(row['Modality (0008|0060)']) == 'MR':
            save_dir = MR_dir
        if str(row['Modality (0008|0060)']) == 'CT':
            save_dir = CT_dir
        params_list.append(
            [dcm_dir, save_dir, save_name, dcm_name, idx, total_num])
        nii_save_name_list.append(save_name)
        # print(nii_save_name_list)

    save_keep_info_df = keep_info_df.copy()
    save_keep_info_df['NIfty Save Name'] = nii_save_name_list
    save_keep_info_df.to_csv(save_info_csv_file_dir, index=False)
    # for i in params_list:
    #     save_dcm_to_nii_gz(i)
    pool = Pool(processes=4)
    # print(params_list)
    # try:
    pool.map(save_dcm_to_nii_gz, params_list)
    # except Exception as e:
        # print('the error',e.__class__.__name__,e)
    pool.close()
    