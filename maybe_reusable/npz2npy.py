
import numpy as np
import glob
import os.path as osp

# data_root = "/mnt/lustre/share_data/gmai/nnUNet_preprocessed/Task095_Ab_tiny"
data_root = "/mnt/lustre/share_data/gmai/nnUNet_preprocessed/Task040_KiTS"
# data_root = "/mnt/lustre/share_data/gmai/nnUNet_preprocessed/Task030_CT_ORG"
npz_list = glob.glob(osp.join(data_root, "*", "*.npz"))
for npz_file in npz_list:
    data = np.load(npz_file)["data"]
    np.save(npz_file[:-3] + "npy", data)
    print(data)
print(npz_list)


