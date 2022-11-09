import glob
import os.path as osp
import SimpleITK as sitk
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


path = "/mnt/petrelfs/wanghaoyu/gmai/totalseg_tmp_data/raw_data/Task558_Totalsegmentator_dataset/imagesTr"
out_path = "/mnt/cache/wanghaoyu/SP_script/dataset_property/totalseg_properties.csv"


def compute_stats(voxels):
    if len(voxels) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    median = np.median(voxels)
    mean = np.mean(voxels)
    sd = np.std(voxels)
    mn = np.min(voxels)
    mx = np.max(voxels)
    percentile_99_5 = np.percentile(voxels, 99.5)
    percentile_00_5 = np.percentile(voxels, 00.5)
    return median, mean, sd, mn, mx, percentile_99_5, percentile_00_5

files = glob.glob(osp.join(path, "*.nii.gz"))
results = defaultdict(list)
keys = [
    "case_id", "median", "mean", "sd", "mn", "mx", "percentile_99_5", "percentile_00_5"
]

for f in tqdm(files):
    img = sitk.ReadImage(f)
    arr = sitk.GetArrayFromImage(img)
    median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = compute_stats(arr)
    case_id = osp.basename(f).split(".nii")[0]
    data = [case_id, median, mean, sd, mn, mx, percentile_99_5, percentile_00_5]
    for idx, k in enumerate(keys):
        results[k].append(data[idx])

df = pd.DataFrame(results)
df.to_csv(out_path)
