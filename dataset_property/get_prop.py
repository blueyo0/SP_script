# -*- encoding: utf-8 -*-
'''
@File    :   get_prop.py
@Time    :   2022/11/08 14:03:33
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   批量读取data中的pkl信息
'''
import pickle
import numpy as np
from glob import glob
import os.path as osp

def get_info(path):
    content = pickle.load(open(path, "rb"))
    target_spacing = np.percentile(np.vstack(content['all_spacings']), 50, 0)
    target_size = np.percentile(np.vstack(content['all_sizes']), 50, 0)
    return {
        "spacing": target_spacing,
        "size": target_size,
        "modalities": content['modalities'],
        "classes": content['all_classes'],
    }

if __name__ == "__main__":
    print("start to compute")
    files = glob("/mnt/cache/wanghaoyu/SP_script/dataset_property/data/*.pkl")    
    for f in files:
        info = get_info(f)
        print(osp.basename(f.split(".pkl")[0]))
        print("spacing:", "["+",".join(["%.2f" % (i) for i in info["spacing"]])+"]", 
              "size:", "["+",".join(["%.2f" % (i) for i in info["size"]])+"]", 
              "modality:", len(info["modalities"]), 
              "num_classes:", len(info["classes"]))
        print("["+",".join(["%.2f" % (i) for i in info["spacing"]])+"]", 
              "\t", "["+",".join(["%.2f" % (i) for i in info["size"]])+"]", 
              "\t", len(info["modalities"]), 
              "\t", len(info["classes"]))
    
    
    