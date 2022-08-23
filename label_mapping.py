true_labels = "Spleen-CT	Right Kidney-CT	Left Kidney-CT	Gallbladder-CT	Esophagus-CT	Liver-CT	Stomach-CT	Aorta-CT	Inferior Vena Cava-CT	Pancreas-CT	Right Adrenal-CT	Left Adrenal-CT	Portal vein, splenic vein-CT	Liver vessel-CT	Right Kidney-MRI	Left Kidney-MRI	Stomach-MRI	Pancreas-MRI	Prostate/uterus-MRI	Renal tumor-CT	Colon tumor-CT	Liver tumor-CT	Pancreatic tumor-CT	Kidney cystitis-CT	Pancreatic lesions-CT	Flexion-CT	Non-shifted-CT	Shift-CT	Segmental rib fracture-CT	Pancreatic lesions-MRI	Stomach polyp-MRI	Brain-MRI	External cerebrospinal fluid -MRI	Cerebrospinal fluid -MRI	Gray matter -MRI	White matter -MRI	Ventricles -MRI	Cerebellum -MRI	Deep gray matter -MRI	Brain stem -MRI	Brain skull-MRI	Anterior hippocampus of the brain - MRI	Posterior hippocampus of the brain - MRI	Vascular artery of the neck Wall-MRI	Cochlea-MRI	Brain tumor-CT	Brain growth-CT	Brain acute stroke lesions-MRI	Brain ischemic subacute stroke lesions-MRI	Brain necrosis-MRI	Brain edema-MRI	Non-enhancing tumor of the brain -MRI	Lung-CT	Pulmonary artery-CT	Lung airway-CT	Pulmonary artery-MRI	Lung lesion (ggo)-CT	Lung tumor-CT	Pulmonary nodules-CT	Left ventricle blood chamber-CT	Right ventricle blood chamber-CT	Left atrial blood chamber-CT	Right atrial blood chamber-CT	Left ventricle myocardium-CT	Ascending aorta-CT	coronary artery-CT	heart-CT	left anterior descending branch-CT	left circumflex artery-CT	right coronary artery-CT	Right ventricle blood chamber-MRI	Left atrial blood chamber-MRI 	Right atrial blood chamber-MRI	Left ventricle myocardium-MRI	Ascending aorta-MRI	Left atrium-MRI	myocardium-MRI	C1-CT	C2-CT	C3-CT	C4-CT	C5-CT	C6-CT	C7-CT	T1-CT	T2-CT	T3-CT	T4-CT	T5-CT	T6-CT	T7-CT	T8-CT	T9-CT	T10-CT	T11-CT	T12-CT	L1-CT	L2-CT	L3-CT	L4-CT	L5-CT	L6-CT"
true_labels = true_labels.split("\t")
true_labels = [t.strip() for t in true_labels]
# labels_info = dict()
# for label in true_labels:
#     name, modality = label.split('-')
#     if(modality)


dataset_list = [
    "Task011_BTCV",
    "Task037_CHAOS_Task_3_5_Variant1",
    "Task021_KiTS2021",
    "Task003_Liver",
    "Task005_Prostate",
    "Task006_Lung",
    "Task007_Pancreas",
    "Task008_HepaticVessel",
    "Task009_Spleen",
    "Task010_Colon",
    "Task029_LITS",
    "Task154_RibFrac",
    "Task040_KiTS",
]

modality_list = [
    "CT",  # "Task011_BTCV",
    "MRI", # "Task037_CHAOS_Task_3_5_Variant1",
    "CT",  # "Task021_KiTS2021",
    "CT",  # "Task003_Liver",
    "CT",  # "Task005_Prostate",
    "CT",  # "Task006_Lung",
    "CT",  # "Task007_Pancreas",
    "CT",  # "Task008_HepaticVessel",
    "CT",  # "Task009_Spleen",
    "CT",  # "Task010_Colon",
    "CT",  # "Task029_LITS",
    "CT",  # "Task154_RibFrac",
    "CT",  # "Task040_KiTS"
]


import json
import os.path as osp


if __name__ == "__main__":
    dataset_root = "/mnt/lustre/share_data/gmai/nnUNet_preprocessed"
    for d_idx, dataset in enumerate(dataset_list):
        file = open(osp.join(dataset_root, dataset,  "dataset.json"), "r")
        info = json.load(file)["labels"]
        modality = modality_list[d_idx]
        result_per_dataset = [""]*len(true_labels)
        for idx, label in info.items():
            idx = int(idx)
            if(idx==0): continue # skip background
            for gt_idx, gt in enumerate(true_labels):
                gt_name, gt_modality = gt.rsplit("-", 1)
                if(gt_name.lower()==label.lower()):
                    dataset_name = dataset.split('_', 1)[1]
                    result = f"{dataset_name}-{modality}-{label}: {gt}"
                    result_per_dataset[gt_idx] = result
        print("\t".join(result_per_dataset))
        # break

