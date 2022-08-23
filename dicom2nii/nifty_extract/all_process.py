#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   save_metadata.py
@Time    :   2022/02/17 20:44:52
@Author  :   Sun Hui 
@Version :   1.0
@Contact :   bitsunhui@163.com
@License :   (C)Copyright 2022-2028, uni-medical
@Desc    :   None
'''

import os
import os.path as osp
import SimpleITK as sitk
import numpy as np
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool
import shutil
import sys
# https://www.dicomlibrary.com/dicom/dicom-tags/
metadata_map= defaultdict()
dicom_tag_file = os.path.join(os.path.split(os.path.realpath(__file__))[0], 
                              'common_dicom_tag.txt')
with open(dicom_tag_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        s1, *s2 = line.split(')')
        s2 = line.split('{})'.format(s1))[-1]
        tag = '|'.join(s_.strip() for s_ in s1.split('(')[-1].split(','))
        name = s2.split('\n')[0].lstrip().rstrip()
        map_name = '{} ({})'.format(name, tag)
        metadata_map[tag] = map_name

def get_series_metadata(params):
    idx, total_num, dcm_dir = params
    print('{}/{}, dir: {}'.format(idx, total_num, dcm_dir))
    
    series_reader = sitk.ImageSeriesReader()
    file_reader = sitk.ImageFileReader()
    file_reader.LoadPrivateTagsOn()
    
    series_IDs = series_reader.GetGDCMSeriesIDs(dcm_dir)
    if not series_IDs:
        print("ERROR: given directory \"{}\" does not contain a DICOM series.")
        return -1, None
    
    series_info = defaultdict(list)
    for serie in series_IDs:
        # Get the dicom filename corresponding to the current serie
        serie_names = series_reader.GetGDCMSeriesFileNames(dcm_dir, serie)
        slice_dir = serie_names[0]
        file_reader.SetFileName(slice_dir)
        file_reader.ReadImageInformation()

        series_info['DicomDir'].append(dcm_dir)
        series_info['SerieSliceNum'].append(len(serie_names))
        for tag, name in metadata_map.items():
            if file_reader.HasMetaDataKey(tag):
                value = file_reader.GetMetaData(tag)
            else:
                value = 'null'
            series_info[name].append(value)
    return 1, series_info
def save_dcm_to_nii_gz(parmas):
    dcm_dir, save_dir, series_uid, dcm_name, idx, total_num = parmas
    print('{}/{}, name: {}'.format(idx, total_num, dcm_dir))
    save_dir = osp.abspath(save_dir)
    series_reader = sitk.ImageSeriesReader()
    serie_names = series_reader.GetGDCMSeriesFileNames(dcm_dir, series_uid)
    series_reader.SetFileNames(serie_names)
    simage = series_reader.Execute()
    sitk.WriteImage(simage, save_dir) 

if __name__ == '__main__':
    # dataset, hk-test
    # dir_all = sys.argv[1] #r'/mnt/lustre/wanghaoyu/dataset/temp/jqn_tmp/private/unpack' # 
    # Nifity_out_dir = sys.argv[2] #r'/mnt/lustre/wanghaoyu/dataset/temp/jqn_tmp/Nifity_all'
    dir_all = r'/mnt/lustre/wanghaoyu/dataset/temp/why_temp/unpack' # 
    Nifity_out_dir = r'/mnt/lustre/wanghaoyu/dataset/temp/why_temp/nifti'
    dataset_list = os.listdir(dir_all)

    print(dataset_list)
    for dataset_dir in dataset_list:# saving dir 
        dcm_dataset_dir = os.path.join(dir_all ,dataset_dir)
        if os.path.isdir(dcm_dataset_dir):
            if not os.path.exists(os.path.join(dcm_dataset_dir, 'meta-info.csv')):
                info_csv_dir = dcm_dataset_dir 
                other_file_dir = os.path.join(info_csv_dir,'other_file')
                print(other_file_dir)
                os.makedirs(other_file_dir, exist_ok=True)
                os.makedirs(info_csv_dir, exist_ok=True)
                exist_dir_list = []
                i = 0
                for dir, folders, files in os.walk(dcm_dataset_dir):
                    if dir not in exist_dir_list:
                        i += 1
                        # if (len(files) > 0) and (files[0].endswith('dcm') or files[0].endswith('DCM')):
                        if len(files) > 0 and  any(file.endswith('.dcm') for file in files):
                            exist_dir_list.append(dir)           
                params_list = []
                total_num = len(exist_dir_list)
                print('total_num is', total_num)
                for dir in exist_dir_list:
                    dcm_files = os.listdir(dir)
                    for file in dcm_files:
                        if not file.endswith('.dcm'):
                            #print(os.path.join(other_file_dir, dir.split('LIDC_dcm-NC')[-1].replace('/', '_')+file))
                            shutil.move(os.path.join(dir,file),os.path.join(other_file_dir, dir.split('LIDC_dcm-NC')[-1].replace('/', '_')+file))
                for i, dcm_dir in enumerate(exist_dir_list):
                    params_list.append([i+1, total_num, dcm_dir])
                
                pool = Pool(processes=4)
                results = pool.map(get_series_metadata, params_list)
                pool.close()
                info_dict = defaultdict(list)
                for ret_code, series_info in results:
                    if ret_code == 1:
                        for key, value in series_info.items():
                            info_dict[key].extend(value)
                
                info_df = pd.DataFrame(data=info_dict, columns=list(info_dict.keys()))
                info_df.to_csv(os.path.join(info_csv_dir, 'meta-info.csv'), index=False)
            ######################################################
            #save meta_csv
            #######################################################
            print(dataset_dir)
            info_csv_file_dir = os.path.join(dcm_dataset_dir, 'meta-info.csv')
            save_info_csv_file_dir = os.path.join(Nifity_out_dir, dataset_dir, 'save-nifty-meta-info.csv' ) #ACRIN_6657/save-nifty-meta-info.csv'
            save_nii_dir = os.path.join(Nifity_out_dir, dataset_dir )
            save_nii_dir_MR = os.path.join(save_nii_dir, 'MR')
            save_nii_dir_CT = os.path.join(save_nii_dir, 'CT')
            os.makedirs(save_nii_dir_MR, exist_ok= True)
            os.makedirs(save_nii_dir_CT, exist_ok= True)
            # os.makedirs(save_nii_dir, exist_ok=True)
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
            keep_info_df = info_df[info_df['is_keep']== 1]
            keep_info_df = keep_info_df.drop(columns=['is_keep'])
            # save to nifty
            params_list = []
            nii_save_name_list = []
            total_num = len(keep_info_df)
            print('total_num', total_num)
            for i, row in keep_info_df.iterrows():
                print(i)
                # print(i)
                idx = i
                dcm_dir = str(row['DicomDir'])
                dcm_name = dcm_dir.split('/')[-1]
                series_uid = str(row['Series Instance UID (0020|000e)'])
                save_name = '{}-{}.nii.gz'.format(dataset_suffix, series_uid)
                print('row[\'Modality (0008|0060)\']',row['Modality (0008|0060)'])
                if row['Modality (0008|0060)'] == 'MR':
                    save_dir = os.path.join(os.path.join(save_nii_dir, save_nii_dir_MR), save_name)
                if row['Modality (0008|0060)'] == 'CT':
                    save_dir = os.path.join(os.path.join(save_nii_dir, save_nii_dir_CT), save_name)
                print('save_dir',save_dir)
                if not os.path.exists(save_dir):
                    params_list.append(
                        [dcm_dir, save_dir, series_uid, dcm_name, idx, total_num])
                nii_save_name_list.append(save_name)
                # print(nii_save_name_list)

            save_keep_info_df = keep_info_df.copy()
            save_keep_info_df['NIfty Save Name'] = nii_save_name_list
            save_keep_info_df.to_csv(save_info_csv_file_dir, index=False)
            
            # for param in params_list:
            #     save_dcm_to_nii_gz(param)

            pool = Pool(processes=4)
            # print(params_list)
            try:
                pool.map(save_dcm_to_nii_gz, params_list)
            except Exception as e:
                print('the error',e.__class__.__name__,e)
            pool.close()