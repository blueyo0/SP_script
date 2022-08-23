import glob
import os.path as osp
import shutil

if __name__ == "__main__":
    path = "/mnt/cache/wanghaoyu/preprocess/data/result/duplicate.txt"
    filelist = open(path, 'r').readlines()
    filelist = [f.split()[-1] for f in filelist]
    print(filelist)

    folder1 = "/mnt/lustre/share_data/gmai/dataset/raw/labeled/AbdomenCT/imagesTr"
    folder2 = "/mnt/lustre/share_data/gmai/dataset/preprocessed/temp/AbdomenCT5K/unlabel/"
    file_list1 = glob.glob(osp.join(folder1, "*.nii.gz"))
    non_overlap_list = []
    for f in file_list1:
        if(f not in filelist): non_overlap_list.append(f)
    print(len(non_overlap_list))

    for nf in non_overlap_list:
        target_file = osp.join(folder2, osp.basename(nf).replace("Case", "Ab1K"))
        print(target_file)
        shutil.copyfile(nf, target_file)