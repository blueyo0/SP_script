# -*- encoding: utf-8 -*-
'''
@File    :   dataset_json_downloader.py
@Time    :   2022/11/29 11:56:01
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   为了解析每个数据集的类别，download对应的dataset.json文件
'''

import subprocess
import os.path as osp
import pandas as pd

dataset_paths = pd.read_csv("datasets_with_json.csv")
dataset_dict = {d[3].split(",")[0]:tuple("s3://nnUNet_raw_data/{}".format(sd.strip()) for sd in d[3].split(",")) for d in dataset_paths.itertuples()}
local_dir = "/mnt/petrelfs/wanghaoyu/why/local_label"

if __name__ == "__main__":
    for paths in dataset_dict.values():
        for path in paths:
            name = osp.basename(path)
            remote_path = osp.join(path, "dataset.json")
            local_path = osp.join(local_dir, name, "dataset.json")
            if(osp.exists(local_path)):
                print("skip", remote_path)
                # pass
            else:
                print("downloading", remote_path, "to", local_path)
                process = subprocess.Popen(
                    f"mkdir -p \"{local_dir}/{name}\"", 
                    shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                cmd_out = process.stdout.read().decode('utf-8')
                process = subprocess.Popen(
                    f"aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 cp {remote_path} {local_path}", 
                    shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                cmd_out = process.stdout.read().decode('utf-8')
                if(cmd_out.startswith("fatal error")):
                    print("[WARINING] fail to download", remote_path)
0