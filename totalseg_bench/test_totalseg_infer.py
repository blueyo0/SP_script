import os
import os.path as osp
import glob
import subprocess
from tqdm import tqdm
import sys

# data_root = "/mnt/petrelfs/wanghaoyu/gmai/nnUNet_raw_data_base/nnUNet_raw_data"
# output_root = "/mnt/petrelfs/wanghaoyu/gmai/nnUNet_raw_data_base/totalseg_result"
data_root = "/mnt/petrelfs/wanghaoyu/gmai/totalseg_tmp_data/raw_data"
output_root = "/mnt/petrelfs/wanghaoyu/gmai/totalseg_result/"

# dataset_list = [
#     # "Task020_AbdomenCT1K", 
#     "Task011_BTCV", 
#     # "Task003_Liver", 
#     # "Task007_Pancreas", 
#     # "Task009_Spleen", 
#     # "Task021_KiTS2021", 
#     # "Task154_RibFrac", 
#     # "Task083_VerSe2020", 
#     # "Task030_CT_ORG", 
#     # "Task037_CHAOS_Task_3_5_Variant1", 
#     # "Task031_AMOS_MR", 
# ]

dataset_list = []
dataset_list.append(sys.argv[1])

print("dataset_list:", dataset_list)
# dataset_list = glob.glob(osp.join(data_root, "TotalSeg_Ts", "imagesTr"))
for dataset in dataset_list:
    dataset_name = dataset
    output_dir = osp.join(output_root, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    image_list = glob.glob(osp.join(data_root, dataset, "imagesTr", "*.nii.gz"))
    pbar = tqdm(image_list)
    for image in pbar:
        out_path = osp.join(output_dir, osp.basename(image).split(".nii")[0])
        # [HACK] hide the prediction of normal 1.5 mm
        # if(not osp.exists(out_path+"_normal")):
        #     process = subprocess.Popen(f"TotalSegmentator -i {image} -o {out_path}_normal", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)                                                                                                                                   
        #     cmd_out = process.stdout.read().decode('utf-8')
            # print("normal:", cmd_out)
        if(not osp.exists(out_path+"_fast")):
            process = subprocess.Popen(f"TotalSegmentator -i {image} -o {out_path}_fast --fast", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            cmd_out = process.stdout.read().decode('utf-8')
        # print(f"{i}/{len(image_list)} {image}")
