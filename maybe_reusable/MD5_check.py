# -*- encoding: utf-8 -*-
'''
@File    :   MD5_check.py
@Time    :   2022/06/06 17:29:28
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   检查两个文件夹里的MD5是否存在重叠
'''
import hashlib
import os.path as osp
import os
import glob

from tqdm import tqdm

def getMD5(filename):
    file = open(filename,'rb')
    md5 = hashlib.md5(file.read()).hexdigest()
    return md5

def getMD5List(dirname, output=None):
    file_list = glob.glob(osp.join(dirname, "*.nii.gz"))
    md5_results = []
    file_results = []
    if(output): out_file = open(output, "w")
    pbar = tqdm(file_list)
    for f in pbar:
        md5 = getMD5(f)
        md5_results.append(md5)
        file_results.append(f)
        if(output): out_file.write(md5+"\t"+f)
    if(output): out_file.close()
    return dict(md5=md5_results, files=file_results)


if __name__ == "__main__":
    folder1 = "/mnt/lustre/share_data/gmai/dataset/raw/labeled/AbdomenCT/imagesTr"
    folder2 = "/mnt/lustre/share_data/gmai/dataset/preprocessed/temp/AbdomenCT5K/unlabel/"
    result_folder = "/mnt/cache/wanghaoyu/preprocess/data/result"
    info1 = getMD5List(folder1, output=osp.join(result_folder, "AbdomenCT"))
    info2 = getMD5List(folder2, output=osp.join(result_folder, "FLARE22"))
    
    non_overlap_files = []
    for i in range(len(info1["md5"])):
        if(info1["md5"][i] in info2["md5"]):
            print("duplicate", info1["files"][i])
        else:
            non_overlap_files.append(info1["files"][i])

    print("useful files: ", len(non_overlap_files))