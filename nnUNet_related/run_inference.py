# -*- encoding: utf-8 -*-
'''
@File    :   run_inference.py
@Time    :   2022/08/30 10:53:36
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   challenge用nnUNet inference脚本，带有multi-scale test和多模型集成
'''

from pathlib import Path
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from sklearn import model_selection


if __name__ == '__main__':
    input_folder = './input'
    output_folder = './output'
    parameter_folder = './parameters/'

    model_dict = {
        "gammaV3": (0,1,2,3,4), 
        "bru2": ('all'), 
        "cmg3": ('all')
    }
    ratio_list = [1.0, 1.2]
    from nnunet.inference.predict import predict_cases
    from nnunet.inference.ensemble_predictions import merge
    from batchgenerators.utilities.file_and_folder_operations import subfiles, join
    import os
    import shutil

    test_files = subfiles(input_folder, suffix='.nii.gz', join=False)

    # output_files = [join(output_folder, i) for i in test_files]
    input_files = [join(input_folder, i) for i in test_files]

    # in the parameters folder are five models (fold_X) traines as a cross-validation. We use them as an ensemble for
    # prediction
    # folds = (0, 1, 2, 3, 4)
    # folds = (0)

    # setting this to True will make nnU-Net use test time augmentation in the form of mirroring along all axes. This
    # will increase inference time a lot at small gain, so you can turn that off
    do_tta = False

    # does inference with mixed precision. Same output, twice the speed on Turing and newer. It's free lunch!
    mixed_precision = True
    output_dir_list = []
    for model, folds in model_dict.items():
        multi_scale_output_dir_list = [join(output_folder, model, str(r)) for r in ratio_list]
        for i, r in enumerate(ratio_list):
            output_files = [join(multi_scale_output_dir_list[i], tf) for tf in test_files]
            predict_cases(join(parameter_folder, model), [[i] for i in input_files], output_files, folds, save_npz=True,
                          num_threads_preprocessing=2, num_threads_nifti_save=2, segs_from_prev_stage=None, do_tta=do_tta,
                          mixed_precision=mixed_precision, overwrite_existing=True, all_in_gpu=False, step_size=0.5, ratio=r)
        # [TO-DO] ensemble and delete multi-scale preds
        merge(multi_scale_output_dir_list, 
              join(output_folder, model, "ms"), 6, 
              override=True, postprocessing_file=None, store_npz=True)
        output_dir_list.append(join(output_folder, model, "ms"))
        for ms_dir in multi_scale_output_dir_list:
            shutil.rmtree(ms_dir)

    # [TO-DO] ensemble three preds
    merge(output_dir_list, output_folder, 6, 
          override=True, postprocessing_file=None, store_npz=False)
    for ms_dir in output_dir_list:
        shutil.rmtree(ms_dir)
