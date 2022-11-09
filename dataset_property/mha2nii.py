import glob
import os.path as osp
import os
import SimpleITK as sitk

path = "/mnt/petrelfs/wanghaoyu/temp/dataset_analyze/BraTS13"
out_dir = "/mnt/petrelfs/wanghaoyu/temp/dataset_analyze/Task111_BraTS2013"
out_dir = osp.join(out_dir, "imagesTr")
os.makedirs(out_dir, exist_ok=True)

files = glob.glob(osp.join(path, "*", "VSD.Brain.XX*", "*.mha"))

for f in files:
    case_id = osp.basename(osp.dirname(osp.dirname(f)))
    modality = osp.basename(osp.dirname(f)).split("_")[-1]
    img = sitk.ReadImage(f)
    sitk.WriteImage(img, osp.join(out_dir, case_id+"_"+modality+".nii.gz"))
