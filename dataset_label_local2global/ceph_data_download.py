# -*- encoding: utf-8 -*-
'''
@File    :   ceph_data_download.py
@Time    :   2022/09/26 13:14:40
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   表3数据下载
'''

import subprocess
import os.path as osp
dataset_file = open("ceph_data_to_check.txt", "r")
dataset_list = [i.strip() for i in dataset_file.readlines()]


if __name__ == "__main__":
    for dataset in dataset_list:
        process = subprocess.Popen(
            f"aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 ls {dataset} | tail -n 1|cut -c 32-200", 
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        cmd_out = process.stdout.read().decode('utf-8')
        file_name = cmd_out.strip()
        data_path = dataset+file_name
        print("downloading", data_path)
        process = subprocess.Popen(
            f"mkdir -p \"/mnt/petrelfs/wanghaoyu/why/why_download/{dataset}\"", 
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        cmd_out = process.stdout.read().decode('utf-8')
        process = subprocess.Popen(
            f"aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 cp {data_path} \"/mnt/petrelfs/wanghaoyu/why/why_download/{dataset}/{file_name}\"", 
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        cmd_out = process.stdout.read().decode('utf-8')



