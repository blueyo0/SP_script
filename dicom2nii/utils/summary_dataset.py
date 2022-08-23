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
    i = 0
    for file in file_list:
        if file.endswith('.nii.gz'):
            nii_list.append(file)
    for nii in tqdm(nii_list):
    
        nii_file = itk.ReadImage(os.path.join(datadir,nii))
        shape,spacing = itk.GetArrayFromImage(nii_file).shape,  nii_file.GetSpacing()
        shape_list.append(shape)
        spacing_list.append(spacing)
    # nii_img = nii_file.get_fdata()
    return shape_list, spacing_list

data_dir = '/home/PJLAB/niujingqi/data_extra/Nifty_all/No-Label/Abdomen/CPTAC-UCEC/CT'
shape,spacing = summary(data_dir)
shape = np.array(shape)
spacing = np.array(spacing)
print('shape : {}, spacing : {}'.format(shape.mean(axis=0), spacing.mean(axis=0)))
# print('mean shape : {}, mean spacing {}'.format(shape, spacing))
# train_dl = DataLoader(dataset, 2, False, num_workers=1)
# for shape,spacing in train_dl:
#     print(shape,spacing)
# summary = DatasetSummary(NL_dataset)
# summary.collect_meta_data()