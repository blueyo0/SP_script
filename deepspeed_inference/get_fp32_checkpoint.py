# -*- encoding: utf-8 -*-
'''
@File    :   get_fp32_checkpoint.py
@Time    :   2022/11/25 10:54:39
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   提供 nnUNet ckpt 的转换
'''

import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import os.path as osp

def convert_checkpoint_to_fp32(fname):
    print("loading checkpoint", fname)
    saved_model = torch.load(fname, map_location=torch.device('cpu'))
    state_dict = get_fp32_state_dict_from_zero_checkpoint(osp.dirname(fname), tag="ds_"+osp.basename(fname))
    saved_model['state_dict'] = state_dict
    torch.save(saved_model, osp.join(osp.dirname(fname), "fp32_"+osp.basename(fname)))
    print("finish converting")

if __name__ == "__main__":
    ckpt_dir = "/nvme/wanghaoyu/nnUNet_root/RESULTS_FOLDER/nnUNet/3d_fullres/Task559_TotalSeg_wo864_nonCT_norm/BigResUNetTrainerV3_DS_113393_160_noMirror__nnUNetPlansv2.1/fold_1"
    for ep in [50, 100]:                                                                                                               
        fname = osp.join(ckpt_dir, "model_ep_{:03d}.model".format(ep))
        convert_checkpoint_to_fp32(fname)