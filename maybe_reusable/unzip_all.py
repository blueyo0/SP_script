import os
import os.path as osp
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import pickle
import zipfile
import hashlib
import glob

def unpack(path, unpack_root):
    with zipfile.ZipFile(path, 'r') as zf:
        os.makedirs(unpack_root, exist_ok=True)
        zf.extractall(unpack_root)
    print("unpack to", unpack_root)
    correct = True
    return correct

def unpack_dataset(dataset_path, unpack_root):
    os.makedirs(unpack_root, exist_ok=True)
    series_list = glob.glob(osp.join(dataset_path, "*"))
    for s in series_list:
        status = unpack(s, osp.join(unpack_root, osp.basename(s)))

if __name__ == "__main__":
    zip_path = "/mnt/lustre/wanghaoyu/dataset/temp/why_temp/zip"
    unpack_path = "/mnt/lustre/wanghaoyu/dataset/temp/why_temp/unpack"
    dir_list = glob.glob(osp.join(zip_path, "*"))
    print("TODO:", dir_list)
    for dataset_dir in dir_list:
        if osp.isdir(dataset_dir):
            dataset_name = osp.basename(dataset_dir)
            print("unpacking", dataset_name)
            unpack_dataset(dataset_dir, osp.join(unpack_path, dataset_name))
        else:
            print(osp.basename(dataset_dir), "is not a directory")
    print("end")
