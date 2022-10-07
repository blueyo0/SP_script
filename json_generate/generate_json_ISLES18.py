from typing import Tuple
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *

def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques

def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: Tuple,
                          labels: dict, dataset_name: str, license: str = "hands off!", dataset_description: str = "",
                          dataset_reference="", dataset_release='0.0'):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = len(test_identifiers)
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i
        # {'image': "./imagesTr/%s.nii.gz" % i} for i
        in
        train_identifiers]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file))


# rename flair_pp 0000 *
# rename mprage_pp 0001 *
# rename pd_pp 0002 *
# rename t2_pp 0003 *

if __name__ == "__main__":
    data_root = "/mnt/petrelfs/wanghaoyu/why/why_download/Task165_ISLES18"
    import os.path as osp
    import os
    from glob import glob
    import shutil
    import subprocess

    os.makedirs(osp.join(data_root, "imagesTr"), exist_ok=True)
    os.makedirs(osp.join(data_root, "labelsTr"), exist_ok=True)
    os.makedirs(osp.join(data_root, "imagesTs"), exist_ok=True)

    data_list = glob(osp.join(data_root, "TRAINING", "case*"))
    for data in data_list:
        case_id = osp.basename(data)
        images = [glob(osp.join(data, f"SMIR.Brain.XX.O.CT{suffix}*"))[0] for suffix in [".", "_4DPWI", "_CBF", "_CBV", "_MTT", "_Tmax"]]
        # print("\n".join(images))
        for idx, image in enumerate(images):
            print(osp.join(image, image+".nii"), "->", osp.join(data_root, "imagesTr", "{}_{:04d}.nii".format(case_id, idx)))
            shutil.copyfile(osp.join(image, osp.basename(image)+".nii"), osp.join(data_root, "imagesTr", "{}_{:04d}.nii".format(case_id, idx)))
        labels = glob(osp.join(data, "SMIR.Brain.XX.O.OT*"))
        shutil.copyfile(osp.join(data, labels[0], osp.basename(labels[0])+".nii"), osp.join(data_root, "labelsTr", "{}.nii".format(case_id)))
        # exit()
    
    data_list = glob(osp.join(data_root, "TESTING", "case*"))
    for data in data_list:
        case_id = osp.basename(data)
        images = glob(osp.join(data, "SMIR.Brain.XX.O.CT*"))
        for idx, image in enumerate(images):
            shutil.copyfile(osp.join(image, osp.basename(image)+".nii"), osp.join(data_root, "imagesTs", "{}_{:04d}.nii".format(case_id, idx)))

    process = subprocess.Popen("gzip {}/*.nii".format(osp.join(data_root, "imagesTr")), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cmd_out = process.stdout.read().decode('utf-8')
    process = subprocess.Popen("gzip {}/*.nii".format(osp.join(data_root, "imagesTs")), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cmd_out = process.stdout.read().decode('utf-8')
    process = subprocess.Popen("gzip {}/*.nii".format(osp.join(data_root, "labelsTr")), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    cmd_out = process.stdout.read().decode('utf-8')

    generate_dataset_json(
        osp.join(data_root, "dataset.json"), 
        osp.join(data_root, "imagesTr"), 
        osp.join(data_root, "imagesTs"), 
        modalities=['DWI', '4DPWI', 'CBF', 'CBV', 'MTT', "Tmax"], 
        labels={0: "background", 1: "lesion"}, 
        dataset_name="ISLES18")
