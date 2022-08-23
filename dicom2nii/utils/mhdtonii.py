import SimpleITK as itk
input_dir = '/home/PJLAB/niujingqi/data/STOIC/data/mha'
out_dir = '/home/PJLAB/niujingqi/data/NIFITY_ALL/STOIC'
import os
from tqdm import tqdm
subset_names = os.listdir(input_dir)
# for sub_dir in subset_names :
#     # print(sub_dir)
#     sub_path = os.path.join(input_dir, sub_dir)
#     if os.path.isdir(sub_path):
#         file_names = os.listdir(sub_path)
#         out_new_dir = os.path.join(out_dir, sub_dir)
#         if not os.path.exists(out_new_dir):
#             os.mkdir(out_new_dir)
for file_name in tqdm(subset_names):
    # print(file_name)
    if file_name.endswith('.mha'):
        try:
            print(file_name)
            if not os.path.exists(os.path.join(out_dir, file_name[:-4]) + '.nii.gz'):
                mhd_img = itk.ReadImage(os.path.join(input_dir, file_name))
                # os.remove(os.path.join(sub_path, file_name))
                itk.WriteImage(mhd_img, os.path.join(out_dir, file_name[:-4]) + '.nii.gz')
        except:
            print(file_name)