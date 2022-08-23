from git import HEAD


# -*- encoding: utf-8 -*-
'''
@File    :   nnUNet_epoch_monitor.py
@Time    :   2022/07/09 19:38:39
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   内容说明
'''

import os
import os.path as osp
import glob
import json
import numpy as np

RESULT_FOLDER = "/mnt/lustre/wanghaoyu/runs/nnUNet/RESULTS_FOLDER"
output_path = "/mnt/cache/wanghaoyu/preprocess/data/nn_result"

# read log 
def read_log_for_epoch(path):
    log_file = open(path, 'r')
    content = log_file.readlines()
    valid_content = content[-50:]
    val = None
    for i, valid_line in enumerate(valid_content):
        if valid_line.startswith("epoch:"):
            try:
                val = valid_line.split()[-1].strip('\n')
                val = int(val)
            except:
                pass
            finally:
                print("get curr epoch from", path)
    log_file.close()
    return val




if __name__ == "__main__":
    recur_list = ["model_type", "dataset", "trainer", "fold"] # default nnUNet
    # recur_list = ["model_type", "dataset", "trainer", "setting", "fold"]
    file_root = osp.join(RESULT_FOLDER, "nnUNet")
    filter_list = ['*' for i in recur_list]
    filter_list[1] = "Task*_AMOS*"
    # log_path_list = glob.glob(osp.join(file_root, *['*' for i in recur_list], "validation_raw/summary.json"))
    log_path_list = glob.glob(osp.join(file_root, *filter_list, "training_log_*"))
    
    # reduce to particular folder
    reduced_log_file_dict = dict()
    for log_path in log_path_list:
        dir_name = osp.dirname(log_path)
        log_name = osp.basename(log_path)
        if(dir_name not in reduced_log_file_dict.keys()):
            reduced_log_file_dict[dir_name] = [log_name]
        else:
            reduced_log_file_dict[dir_name].append(log_name)
    
    # read log.txt to find result
    output_file = osp.join(output_path, "epoch_from_log.txt")
    output_file = open(output_file, "w")

    for k, v in reduced_log_file_dict.items():
        meta_info = dict()
        # raw_dir_info = osp.dirname(k)
        raw_dir_info = k
        for i in range(len(recur_list)-1, -1, -1):
            meta_key = recur_list[i]
            meta_info[meta_key] = osp.basename(raw_dir_info)
            if(meta_info[meta_key]=="UNETRTrainer3__nnUNetPlansv2.1"):
                meta_info[meta_key] = "UNTER3"
            if(meta_info[meta_key]=="nnUNetTrainerV2__nnUNetPlansv2.1"):
                meta_info[meta_key] = "UNet"
            raw_dir_info = osp.dirname(raw_dir_info)
        # start to check the log list
        final_result_list = ["None"]
        for log_path in v:
            dice = read_log_for_epoch(osp.join(k, log_path))
            if(dice): final_result_list.append(dice)
        final_result = final_result_list[-1]
        if(final_result):
            reduced_log_file_dict[k] = dict(
                dice=final_result,
                meta=meta_info,    
            )

    reduced_output_list = dict()  
    for k, v in sorted(reduced_log_file_dict.items(), key=lambda d: d[0]):
        meta = v["meta"]
        value = v["dice"]
        dataset = meta['dataset']
        model_type = meta['trainer']
        run_id = f"{dataset}_{model_type}"
        # setting = meta['setting']
        # run_id = f"{dataset}_{model_type}_{setting}"
        if(run_id in reduced_output_list.keys()):
            reduced_output_list[run_id].append(v)
        else:
            reduced_output_list[run_id] = [v]

    for k, v in sorted(reduced_output_list.items(), key=lambda d: d[0]):
        meta = v[0]["meta"]
        for key in recur_list[:-1]:
            print(meta[key], end=", ")
            output_file.write(meta[key]+", ")
        output_file.write("\n")  
        folds = [sv["meta"]["fold"] for sv in v]
    #     # settings = [sv["meta"]["setting"] for sv in v]
        dice_sum = [sv["dice"] for sv in v]
    #     dice_sum = np.array(dice_sum).transpose(1,0)
    #     print(folds)
    #     print(dice_sum)
        for f in folds: output_file.write(f+"\t")
        output_file.write('\n')
        for cls_result in dice_sum:
            # for f in cls_result: 
            output_file.write(str(cls_result)+"\t")
        output_file.write('\n')
        
    output_file.close()
    print("end")