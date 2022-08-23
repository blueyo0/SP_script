# -*- encoding: utf-8 -*-
'''
@File    :   nnUNet_ckpt_upload.py
@Time    :   2022/07/26 15:32:27
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   nnUNet lustre checkpoint检查
'''


import os
import os.path as osp
import glob
import json
import numpy as np
import torch
# import pdb


from petrel_client.client import Client
import io
def load_checkpoint_from_ceph(client, fname):
    with io.BytesIO(client.get('s3://{}'.format(fname))) as f:
        ceph_model = torch.load(f, map_location=torch.device('cpu'))
    return ceph_model
# self.client = Client(enable_mc=True)
# ceph_fname = 'nnUNet_trained_models' + fname.split('nnUNet_trained_models')[1]
# saved_model = load_checkpoint_from_ceph(self.client, ceph_fname)
# print ('[Checkpoint] Load from Ceph!')



lustre_root = "/mnt/lustre/wanghaoyu/runs/nnUNet/RESULTS_FOLDER"
ceph_root = "s3://nnUNet_trained_models"
client = Client(enable_mc=True)


if __name__ == "__main__":
    recur_list = ["model_type", "dataset", "trainer", "setting", "fold"]
    file_root = osp.join(lustre_root, "nnUNet")
    for model_type in ["2d", "3d_fullres"]:
        dataset_list = glob.glob(osp.join(file_root, model_type, "Task*"))
        for dataset in dataset_list:
            ckpt_path_list = []
            ckpt_path_list += glob.glob(osp.join(dataset, "*/*/*", "model_final_checkpoint.model"))
            ckpt_path_list += glob.glob(osp.join(dataset, "*/*", "model_final_checkpoint.model"))
            for ckpt_path in ckpt_path_list:
                ckpt_suffix = ckpt_path.rsplit(file_root, 1)[1]
                settings = ckpt_suffix.split("/")
                # if(settings[4]=="save"): continue
                # # print(ckpt_path)
                ceph_path = ceph_root+"/nnUNet"+ckpt_suffix
                try:
                    ckpt_file = load_checkpoint_from_ceph(client, ceph_path)
                    print("SUCCESS ", ckpt_path)
                except Exception as e:
                    print(e.args)
                    print("ERROR in ", ckpt_path)
                # print("ckpt_path", ckpt_path)
                # print("ceph_path", ceph_path)
                ceph_info = os.popen(f"aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 ls {ceph_path}").read()
                print(ceph_info)
                import pdb; pdb.set_trace()

    # ckpt_path_list = glob.glob(ckpt_regex[0]) \
    #                  + glob.glob()
    # print(ckpt_path_list)




    