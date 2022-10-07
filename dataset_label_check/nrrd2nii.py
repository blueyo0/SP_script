# -*- encoding: utf-8 -*-
'''
@File    :   nrrd2nii.py
@Time    :   2022/09/19 15:55:40
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   nrrd to nii
'''
from tkinter import image_names
import nrrd
import nibabel as nib
import numpy as np
import glob
import os.path as osp
import os

def nrrd2nii(in_fname, out_fname=None, save=True):
    _nrrd = nrrd.read(in_fname)
    data = _nrrd[0]
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    img = nib.as_closest_canonical(img)
    if(save): 
        if out_fname is None:
            out_fname = in_fname.replace('nrrd', 'nii.gz')
        nib.save(img, out_fname)
    return data

keys = [
    "BrainStem",
    "Chiasm",
    "Mandible",
    "OpticNerve_L",
    "OpticNerve_R",
    "Parotid_L",
    "Parotid_R",
    "Submandibular_L",
    "Submandibular_R",
]

if __name__ == "__main__":
    # test_file = "/mnt/petrelfs/wanghaoyu/why/temp/HeadandNeckAutoSegmentationChallenge/0522c0001/img.nrrd"
    # test_file = "/mnt/petrelfs/wanghaoyu/why/temp/HeadandNeckAutoSegmentationChallenge/0522c0001/structures/OpticNerve_L.nrrd"
    # nrrd2nii(test_file)
    out_dir = "/mnt/petrelfs/wanghaoyu/why/temp/Task162_HeadandNeckAutoSegmentationChallenge"
    img_list = glob.glob("/mnt/petrelfs/wanghaoyu/why/temp/HeadandNeckAutoSegmentationChallenge/*/img.nrrd")
    os.makedirs(osp.join(out_dir, "imagesTr"), exist_ok=True)
    os.makedirs(osp.join(out_dir, "labelsTr"), exist_ok=True)
    
    # img_list = img_list[:1]
    # print('img_list:', img_list)
    for img_fname in img_list:
        case_id = osp.basename(osp.dirname(img_fname))
        img = nrrd2nii(img_fname, osp.join(out_dir, "imagesTr", f"{case_id}_0000.nii.gz"))
        target = np.zeros_like(img)
        for key in keys:
            if(not osp.exists(osp.join(osp.dirname(img_fname), "structures", key+".nrrd"))): continue
            gt_per_class = nrrd2nii(osp.join(osp.dirname(img_fname), "structures", key+".nrrd"), save=False)
            # gt_per_class = gt_per_class.get_fdata()
            # print(gt_per_class.shape)
            # print(target.shape)
            target[gt_per_class==1] = 1
        nib.Nifti1Image(target, np.eye(4)).to_filename(osp.join(out_dir, "labelsTr", f"{case_id}.nii.gz"))

