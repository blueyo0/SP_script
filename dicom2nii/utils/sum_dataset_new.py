from re import I
import torch
from torch.utils.data import Dataset
import os
import monai
from monai.data.dataset_summary import DatasetSummary
import nibabel
import SimpleITK as itk
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

def summary(datadir):
    datadir = datadir
    file_list = os.listdir(datadir)
    nii_list = []
    shape_list = []
    spacing_list = []
    min_intensity_list = []
    max_intensity_list = []
    i = 0
    statsFilter = itk.StatisticsImageFilter()
    for file in file_list:
        if file.endswith('.nii.gz'):
            nii_list.append(file)
    for nii in tqdm(nii_list):
        try:
            nii_file = itk.ReadImage(os.path.join(datadir,nii))
            statsFilter.Execute(nii_file)
            min_intensity = statsFilter.GetMinimum()
            max_intensity = statsFilter.GetMaximum()
            min_intensity_list.append(min_intensity)
            max_intensity_list.append(max_intensity)
            shape,spacing = itk.GetArrayFromImage(nii_file).shape,  nii_file.GetSpacing()
            shape_list.append(shape)
            spacing_list.append(spacing)
        except:
            print(nii)
    # nii_img = nii_file.get_fdata()
    return shape_list, spacing_list, min_intensity_list, max_intensity_list
Abdomen_dir = '/home/PJLAB/niujingqi/Nifty_data'  #'/home/PJLAB/niujingqi/data/NIFITY_ALL/No-Label' #'/home/PJLAB/niujingqi/data_extra/Nifty_all/No-Label' 
list_datset = os.listdir(Abdomen_dir)
modality_list  = ['MR', 'CT']
for modality in modality_list:
    for dataset_name in list_datset:
        # dataset_name = '/home/PJLAB/niujingqi/data_extra/Nifty_all/No-Label/preprocessed/CT_COLONOGRAPHY-all-CT'
        if  dataset_name == 'Abdomen' or dataset_name == 'preprocessed' or dataset_name == 'BREAST' :
            continue
        data_dir = os.path.join(Abdomen_dir, dataset_name, modality) #os.path.join(Abdomen_dir, dataset_name, modality)
        print('modality: {}, dataset: {}'.format(modality,dataset_name))
        file_list = os.listdir(data_dir)
        if file_list == []:
            continue
        shape,spacing, min_intensity_list, max_intensity_list = summary(data_dir)
        shape = np.array(shape)
        spacing = np.array(spacing)
        # print()
        # print(min(min_intensity_list) )
        # print(round(max(max_intensity_list)))
        # print(int(((np.array(min_intensity_list) + np.array(max_intensity_list))/2).mean()))
        np.set_printoptions(precision=3)
        print('[ {} , {} , {}]'.format(int(min(min_intensity_list)), round(max(max_intensity_list)), int(((np.array(min_intensity_list) + np.array(max_intensity_list))/2).mean())))
        print('[{},  {}, {} ]'.format( list(shape.min(axis=0)) , list(shape.max(axis=0)), list(shape.mean(axis=0).round())))
        print('[ {} , {} ,{} ]'.format( list(np.around(spacing.min(axis=0), 3)) , list(np.around(spacing.max(axis=0), 3)), list(np.around(spacing.mean(axis=0),3))))
        intensity_list = '[ {} , {} , {}]'.format(int(min(min_intensity_list)), round(max(max_intensity_list)), int(((np.array(min_intensity_list) + np.array(max_intensity_list))/2).mean()))
        shape_list = '[{},  {}, {} ]'.format( list(shape.min(axis=0)) , list(shape.max(axis=0)), list(shape.mean(axis=0).round()))
        spacing_list = '[ {} , {} ,{} ]'.format( list(np.around(spacing.min(axis=0), 3)) , list(np.around(spacing.max(axis=0), 3)), list(np.around(spacing.mean(axis=0),3)))
        out_path = '/home/PJLAB/niujingqi/static_dataset.txt'
        with open(out_path, 'a+') as f:
            f.write('modality : {}, dataset : {} \n'.format(modality, dataset_name))
            f.write('Intensity : {} \n'.format(intensity_list))
            f.write('shape : {} \n'.format(shape_list))
            f.write('shape: {} \n'.format(spacing_list))