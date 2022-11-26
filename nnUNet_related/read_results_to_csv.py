# -*- encoding: utf-8 -*-
'''
@File    :   read_results_to_csv.py
@Time    :   2022/11/14 12:39:00
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   读取指定目录下的summary.json，以csv格式输出结果
'''

import json
import os.path as osp
import glob
import pandas as pd

in_dir = "/mnt/lustre/wanghaoyu/runs/nnUNet/RESULTS_FOLDER/nnUNet/3d_fullres/*"
out_fname = "/mnt/cache/wanghaoyu/SP_script/nnUNet_related/MSD_results.csv"
# in_dir = "/mnt/petrelfs/wanghaoyu/temp/picai_ckpt"
# out_fname = "/mnt/cache/wanghaoyu/SP_script/nnUNet_related/picai_results.csv"


res_files = glob.glob(osp.join(in_dir, "*/fold_*/validation_raw/summary.json"))
all_names = ['result_id']+['cls%d'%i for i in range(1, 200)]
df = pd.DataFrame(columns=all_names)
final_results = []
for f in res_files:
    try:
        content = json.load(open(f, "r"))
        res = content['results']['mean']
    except:
        print("error in", f)
        continue
    result_id = f.split("3d_fullres")[-1]
    dices = dict(result_id=result_id)
    for k in range(1, len(res)):
        dice = res[str(k)]['Dice']
        dices["cls%d"%k] = dice
    final_results.append(pd.Series(dices).to_frame().T)
df = pd.concat(final_results, ignore_index=True)
df.to_csv(out_fname)