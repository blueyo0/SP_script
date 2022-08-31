from pathlib import Path
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from sklearn import model_selection


if __name__ == '__main__':
    """
    This inference script is intended to be used within a Docker container as part of the AMOS Test set submission. It
    expects to find input files (.nii.gz) in /input and will write the segmentation output to /output. Note that this
    guide draws heavily on Kits21's submission guidance, and we are grateful to the project's developers.
    
    IMPORTANT: This script performs inference using one nnU-net configuration (3d_lowres, 3d_fullres, 2d OR 
    3d_cascade_fullres). Within the /parameter folder, nnU-Net expects to find fold_X subfolders where X is the fold ID 
    (typically [0-4]). These folds CANNOT originate from different configurations. There also needs to be the plans.pkl 
    file that you find along with these fold_X folders in the corresponding nnunet training output directory.
    
    /parameters/
    ├── fold_0
    │    ├── model_final_checkpoint.model
    │    └── model_final_checkpoint.model.pkl
    ├── fold_1
    ├── ...
    ├── plans.pkl
    
    Note: nnU-Net will read the correct nnU-Net trainer class from the plans.pkl file. Thus there is no need to 
    specify it here. For the ensembling of different nnU-Net configurations (3d_lowres, 3d_fullres, ...), please refer
    to https://github.com/neheller/kits21/blob/master/examples/submission/nnUNet_submission/run_inference_ensembling.py
    
    IMPORTANT: this script performs inference using nn-UNet project, if users use other codebase, please follow
    dockerfile to install/add required packages, codes. And modify the inference code below.
    """
    #
    input_folder = './input'
    output_folder = './output'
    parameter_folder = './parameters/'

    # model_dict = {
    #     "gammaV3": (0,1,2,3,4), 
    #     "bigResUNet": (0,1,2,3,4), 
    #     "cmg3": (0)
    # }
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
