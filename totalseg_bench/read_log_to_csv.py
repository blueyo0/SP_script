# -*- encoding: utf-8 -*-
'''
@File    :   read_results_to_csv.py
@Time    :   2022/11/14 12:39:00
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   读取指定目录下的所有log，以csv格式输出结果
'''

import json
import os.path as osp
import glob
import pandas as pd

in_dir = "/mnt/cache/wanghaoyu/SP_script/data/ziyan_models/"
out_fname = "/mnt/cache/wanghaoyu/SP_script/totalseg_bench/data/ziyan_results.csv"
# in_dir = "/mnt/petrelfs/wanghaoyu/temp/picai_ckpt"
# out_fname = "/mnt/cache/wanghaoyu/SP_script/nnUNet_related/picai_results.csv"


res_files = glob.glob(osp.join(in_dir, "*.log"))
all_names = ['result_id']+['cls%d'%i for i in range(1, 200)]
df = pd.DataFrame(columns=all_names)
final_results = []
for f in res_files:
    find_flag = False
    try:
        content = open(f, "r")
        for line in content.readlines():
            if(line.startswith("result_pre_class")):
                res = line.split()
                find_flag = True
    except:
        print("error in", f)
        continue
    if(not find_flag): 
        print("no result in", f)
        continue
    result_id = osp.basename(f).split(".log")[0].split("TS_test_")[-1]
    dices = dict(result_id=result_id)
    for k in range(1, len(res)):
        dice = res[k]
        dices["cls%d"%k] = dice
    final_results.append(pd.Series(dices).to_frame().T)
df = pd.concat(final_results, ignore_index=True)
df.to_csv(out_fname)