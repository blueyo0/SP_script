import nibabel as nib
import glob
import os.path as osp

def modifyAndSave(fname):
    img = nib.load(fname)
    img_affine  = img.affine
    img = img.get_fdata()
    img[img!=0] = 1
    nib.Nifti1Image(img,img_affine).to_filename(fname)

if __name__ == "__main__":
    # data_dir = "/mnt/petrelfs/wanghaoyu/why/local_label/Task621_Prostate_MRI_Segmentation_Dataset/labelsTr"
    data_dir = "/mnt/petrelfs/wanghaoyu/why/local_label/Task618_Promise09/labelsTr"
    data_list = glob.glob(osp.join(data_dir, "*.nii.gz"))
    for data in data_list:
        modifyAndSave(data)
