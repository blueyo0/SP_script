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
    

if __name__ == '__main__':
    # dataset, hk-test
    dir_all = sys.argv[1] #'/mnt/lustre/wanghaoyu/dataset/temp/jqn_tmp/private/unpack'
    # dcm_dataset_dir = r'/home/PJLAB/niujingqi/data/TCIA/Data/manifest-7qRRRBGo6029235898952856192'#dcm dir which contain only dcm?
    # info_csv_dir =     '/home/PJLAB/niujingqi/data/TCIA/Data/manifest-7qRRRBGo6029235898952856192'
    # other_file_dir = os.path.join(info_csv_dir,'other_file')
    # os.makedirs(other_file_dir, exist_ok=True)
    # os.makedirs(info_csv_dir, exist_ok=True)
    # exist_dir_list = []
    # i = 0
    dataset_list = os.listdir(dir_all)
    for dataset_dir in dataset_list:
        dcm_dataset_dir = os.path.join(dir_all ,dataset_dir)
        if os.path.isdir(dcm_dataset_dir):
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
            