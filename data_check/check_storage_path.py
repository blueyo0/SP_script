# -*- encoding: utf-8 -*-
'''
@File    :   check_storage_path.py
@Time    :   2022/05/14 17:04:18
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   根据总表里扒下来的stroage_path文件，检查所有数据集是否存在，对应的文件夹大小
'''
import os
import os.path as osp
import glob

raw_file_path = "/mnt/cache/wanghaoyu/preprocess/data/uploaded_raw_path_todo.txt"
pre_file_path = "/mnt/cache/wanghaoyu/preprocess/data/uploaded_pre_path_todo.txt"
output_path = "/mnt/cache/wanghaoyu/preprocess/data/result"

def getdirsize(dir):
    size = 0
    for root, dirs, files in os.walk(dir):
        size += sum([osp.getsize(osp.join(root, name)) for name in files])
    return size

if __name__ == "__main__":
    storage_path_files = [open(raw_file_path, "r"), open(pre_file_path, "r")]
    storage_path_list = [sf.readlines() for sf in storage_path_files]
    storage_path_list = [[s.strip('\n') for s in sl] for sl in storage_path_list]
    # check for exisitence
    f = open(osp.join(output_path, "path_review_status.txt"), "w")
    for i in range(len(storage_path_list[0])):
        raw_path, pre_path = storage_path_list[0][i], storage_path_list[1][i]
        raw_status, pre_status = osp.isdir(raw_path), osp.isdir(pre_path)
        
        if raw_status and not pre_status:
            result_str = "MISS: pre"
        if raw_status and pre_status:
            result_str = "all valid"
        if not raw_status and pre_status:
            result_str = "MISS: raw"
        if not raw_status and not pre_status:
            result_str = "MISS: raw, pre"
        print(result_str)
        f.write(result_str+"\n")
    f.close()
    # check for storage size
    # size_list = []
    # for s_path in storage_path_list:
    #     if not osp.isdir(s_path): 
    #         print("[MISS]", s_path)
        # else:
        #     print("[VALID]", s_path)

    #         s_size = getdirsize(s_path) / (1024**3)
    #         size_list.append(dict(
    #             path=s_path,
    #             size=s_size
    #         ))
    # for s in size_list:
    #     print(osp.basename(s["path"]), "{:.2f}G".format(s["size"]))
    print("end")

