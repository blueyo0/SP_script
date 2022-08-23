import nibabel as nib
import os
import SimpleITK as itk
import argparse
from tqdm import tqdm
# parser = argparse.ArgumentParser()
# parser.add_argument('--type', defult = 'nii')
nii_dir = r'/media/PJLAB\niujingqi/d721161f-a51e-4c8c-87d2-bb5ab04476cc/COVID-19-20_v2/COVID-19-20_v2/Train'
nii_files = os.listdir(nii_dir)
max_spacing = [-100,-100,-100]
min_spacing = [100,100,100]
max_shape = [-100,-100,-100]
min_shape = [10000,10000,10000]
for file_name in nii_files:
    file_path = os.path.join(nii_dir, file_name)
    nii_shape = nib.load(file_path).get_data().shape
    nii_spacing = itk.ReadImage(file_path).GetSpacing()
    for i , spacing in enumerate(nii_spacing):
        if max_spacing[i] < spacing:
            max_spacing[i] = spacing
        if min_spacing[i] > spacing:
            min_spacing[i] = spacing
    for i , shape in enumerate(nii_shape):
        if max_shape[i] < shape:
            max_shape[i] = shape
        if min_shape[i] > shape:
            min_shape[i] = shape
print('for spacing, the max spacing is {}, the min spacing is {}'.format( max_spacing, min_spacing))
print('for shape, the max shape is {}, the min shape is {}'.format( max_shape, min_shape))
