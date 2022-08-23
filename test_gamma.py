# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 20:24:34 2022

@author: Lenovo
"""

from batchgenerators.augmentations.color_augmentations import augment_gamma
import SimpleITK as sitk
import os
import numpy as np

def simpleGammaCorrection(data_sample, gamma, epsilon=1e-7, retain_stats_here=False):
    if retain_stats_here:
        mn = data_sample.mean()
        sd = data_sample.std()
    minm = data_sample.min()
    rnge = data_sample.max() - minm
    data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
    if retain_stats_here:
        data_sample = data_sample - data_sample.mean()
        data_sample = data_sample / (data_sample.std() + 1e-8) * sd
        data_sample = data_sample + mn
    return data_sample


if __name__ == "__main__":
    path = "D:\\dataset\\AMOS22\\imagesTr\\amos_0001.nii.gz"
    
    
    x = sitk.ReadImage(path)
    origin = x.GetOrigin()
    direction = x.GetDirection()
    spacing = x.GetSpacing()
    x = sitk.GetArrayFromImage(x)
    # print(x)

    for gamma in [0.3, 3.0, 4.0, 5,0]:   
    # for gamma in [0.1, 0.5, 0.6, 0.7, 1.0, 1.5, 2.0]:    
        out_file = f"D:\\dataset\\AMOS22\\amos_0001_gamma{gamma}.nii.gz"
        # out = augment_gamma(x, (gamma[0], gamma[1]),
        #                     False,
        #                   per_channel=False,
        #                     retain_stats=False)
        out = simpleGammaCorrection(x, gamma)
    
        # out = np.flip(x_2, 1)
        out = sitk.GetImageFromArray(out)
        out.SetOrigin(origin)
        out.SetDirection(direction)
        out.SetSpacing(spacing)
        sitk.WriteImage(out, out_file)
    print("end")
    