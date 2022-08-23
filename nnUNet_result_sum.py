# -*- encoding: utf-8 -*-
'''
@File    :   nnUNet_result_sum.py
@Time    :   2022/05/23 14:03:05
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

# # read log 
# def read_log_for_dice(path):
#     log_file = open(path, 'r')
#     content = log_file.readlines()
#     valid_content = content[-30:]
#     dice_val = None
#     for i, valid_line in enumerate(valid_content):
#         if valid_line.startswith("epoch:  999"):
#             try:
#                 dice_content = valid_content[i+3]
#                 dice_val = dice_content.split(": ")[-1].strip('\n')
#             except:
#                 pass
#             finally:
#                 print("get valid dice (epoch 999) from", path)
#     log_file.close()
#     return dice_val

# read log 
def read_log_for_dice(path):
    log_file = json.load(open(path, 'r'))
    content = log_file["results"]["mean"]
    dice_val = []
    for k, v in sorted(content.items(), key=lambda d: int(d[0])):
        if(k=='0'): continue
        dice_val.append(v["Dice"])
    if(len(dice_val)==0): dice_val=None
    else: 
        res = []
        for d in dice_val: res.append(float("{:.04f}".format(d)))
        dice_val = res
    return dice_val


if __name__ == "__main__":
    recur_list = ["model_type", "dataset", "trainer", "fold"] # default nnUNet
    # recur_list = ["model_type", "dataset", "trainer", "setting", "fold"]
    file_root = osp.join(RESULT_FOLDER, "nnUNet")
    log_path_list = glob.glob(osp.join(file_root, *['*' for i in recur_list], "validation_raw/summary.json"))
    
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
    output_file = osp.join(output_path, "dice_from_log.txt")
    output_file = open(output_file, "w")

    for k, v in reduced_log_file_dict.items():
        meta_info = dict()
        raw_dir_info = osp.dirname(k)
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
            try:
                dice = read_log_for_dice(osp.join(k, log_path))
            except: 
                print("ERROR in", osp.join(k, log_path))
                dice = None
            if(dice): final_result_list.append(dice)
        final_result = final_result_list[-1]
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
        # setting = meta['setting']
        setting = "Default"
        run_id = f"{dataset}_{model_type}_{setting}"
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
        # settings = [sv["meta"]["setting"] for sv in v]
        settings = ["Default" for sv in v]
        dice_sum = [sv["dice"] for sv in v]
        dice_sum = np.array(dice_sum).transpose(1,0)
        print(folds)
        print(dice_sum)
        for f in folds: output_file.write(f+"\t")
        output_file.write('\n')
        for cls_result in dice_sum:
            for f in cls_result: output_file.write(str(f)+"\t")
            output_file.write('\n')
        
    output_file.close()
    print("end")