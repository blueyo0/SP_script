global_mapping = {
    "arota":                        "aorta",
    "IVC":                          "inferior_vena_cava",
    "RAG":                          "adrenal_gland_right",
    "LAG":                          "adrenal_gland_left",
    "bladder":                      "urinary_bladder",
    "LV":                           "left ventricle",
    "RV":                           "right ventricle",
    "LA":                           "left atrium",
    "RA":                           "right atrial blood cavity",
    "LA Scar":                      "left atrial scars",
    "MYO":                          "myocardium",
    "inferior vena cava":           "inferior_vena_cava",
    "portal vein and splenic vein": "portal_vein_and_splenic_vein",
    "right adrenal gland":          "adrenal_gland_right",
    "left adrenal gland":           "adrenal_gland_left",
    "gall bladder":                 "gallbladder",
    "C1":                           "vertebrae_C1",
    "C2":                           "vertebrae_C2",
    "C3":                           "vertebrae_C3",
    "C4":                           "vertebrae_C4",
    "C5":                           "vertebrae_C5",
    "C6":                           "vertebrae_C6",
    "C7":                           "vertebrae_C7",
    "T1":                           "vertebrae_T1",
    "T2":                           "vertebrae_T2",
    "T3":                           "vertebrae_T3",
    "T4":                           "vertebrae_T4",
    "T5":                           "vertebrae_T5",
    "T6":                           "vertebrae_T6",
    "T7":                           "vertebrae_T7",
    "T8":                           "vertebrae_T8",
    "T9":                           "vertebrae_T9",
    "T10":                          "vertebrae_T10",
    "T11":                          "vertebrae_T11",
    "T12":                          "vertebrae_T12",
    "L1":                           "vertebrae_L1",
    "L2":                           "vertebrae_L2",
    "L3":                           "vertebrae_L3",
    "L4":                           "vertebrae_L4",
    "L5":                           "vertebrae_L5",
    "L6":                           "lumbar spine-6",
    "Head_of_femur_L":              "femur_left",
    "Head_of_femur_R":              "femur_right",
    "Brain stem":                   "brainstem",
    "edema":                        "brain_edema",
    "non-enhancing tumor":          "brain_non-enhancing_tumor",
    "enhancing tumor":              "brain_enhancing_tumor",
    "enhancing tumour":             "brain_enhancing_tumor",
}

CUSTUM_MAPPING = {
    "global": global_mapping, 
    "Task002_Heart": {
        "left atrium":   "heart_atrium_left",
    },
    "Task003_Liver": {
        "cancer": "liver_tumor"
    },
    "Task004_Hippocampus": {
        "Anterior":   "anterior hippocampus of the brain",
        "Posterior":  "posterior hippocampus of the brain",
    },
    "Task005_Prostate": {
        "PZ" : "prostate_or_uterus",
        "TZ" : "pancreatic_lesion",
    },
    "Task006_Lung": {
        "cancer": "lung_cancer"
    },
    "Task007_Pancreas": {
        "cancer": "pancreatic_lesions",
    },
    "Task008_HepaticVessel": {
        "Vessel": "liver vessel",
        "Tumour": "liver_tumor",
    },
    "Task010_Colon": {
        "colon cancer primaries": "colon_tumor",
    },
    "Task011_BTCV": {
        "right kidney": "kidney_right",
        "left kidney":  "kidney_left",
    },
    "Task013_ACDC": {
        "RV": "right ventricle",
        "LVC": "left ventricle blood cavity",
        "MLV": "myocardium",
    },
    "Task019_BraTS21": {
        "non-enhancing": "brain_non-enhancing_tumor",
        "enhancing": "brain_enhancing_tumor",
    },
    "Task021_KiTS2021": {
        "tumor":  "renal_tumor",
        "cyst":   "kidney_cyst",
    },
    "Task022_FLARE22": {
        "right kidney": "kidney_right",
        "left kidney":  "kidney_left",
    },
    "Task029_LITS": {
        "tumor": "liver_tumor"
    },
    "Task030_CT_ORG": {
        "bone": "bones",
    },
    "Task032_AMOS22_Task1": {
        "right kidney": "kidney_right",
        "left kidney":  "kidney_left",
        "prostate/uterus": "prostate_or_uterus",
        "postcava": "inferior_vena_cava",
    },
    "Task032_BraTS2018": {
        "non-enhancing": "brain_non-enhancing_tumor",
        "enhancing": "brain_enhancing_tumor",
    },
    "Task036_KiPA22": {
        "renal vein": "kidney veins",
        "renal artery": "kidney arteries",
        "tumor": "renal_tumor",
    },
    "Task037_CHAOS_Task_3_5_Variant1": {
        "right kidney": "kidney_right",
        "left kidney":  "kidney_left",
    },
    "Task040_ATM22": {
        "airway": "pulmonary airway",
    },
    "Task040_KiTS": {
        "Tumor": "renal_tumor",
    },
    "Task043_BraTS2019": {
        "non-enhancing": "brain_non-enhancing_tumor",
        "enhancing": "brain_enhancing_tumor",
    },
    "Task083_VerSe2020": {
        "1":    "C1",
        "2":    "C2",
        "3":    "C3",
        "4":    "C4",
        "5":    "C5",
        "6":    "C6",
        "7":    "C7",
        "8":    "T1",
        "9":    "T2",
        "10":   "T3",
        "11":   "T4",
        "12":   "T5",
        "13":   "T6",
        "14":   "T7",
        "15":   "T8",
        "16":   "T9",
        "17":   "T10",
        "18":   "T11",
        "19":   "T12",
        "20":   "L1",
        "21":   "L2",
        "22":   "L3",
        "23":   "L4",
        "24":   "L5",
        "25":   "L6",
        "26":   "sacrum",
        "27":   "cocygis",
        "28":   "T13"
    },
    "Task154_RibFrac": {
        "displaced_rib_fracture":       "rib_fracture",
        "non_displaced_rib_fracture":   "rib_fracture",
        "buckle_rib_fracture":          "rib_fracture",
        "segmental_rib_fracture":       "rib_fracture",
        "unidentified_rib_fracture":    "rib_fracture",
    },
    "Task161_MRBrainS13DataNii": {
        "Cortical gray matter": "gray matter",
        "Cerebrospinal fluid in the extracerebral space": "external cerebrospinal fluid",
        "Basal ganglia": "gray_matter",
        "White matter lesions": "white_matter",
    },
    "Task162_HeadandNeckAutoSegmentationChallenge": {
        "Chiasm": "optic chiasm",
        "OpticNerve_L": "left and right optic nerves",
        "OpticNerve_R": "left and right optic nerves",
        "Parotid_L": "left and right parotid glands",
        "Parotid_R": "left and right parotid glands",
        "Submandibular_L": "left and right submandibular glands",
        "Submandibular_R": "left and right submandibular glands",
    },
    "Task165_ISLES18": {
        "lesion": "ischemic_stroke_lesion"
    },
    "Task166_LongitudinalMultipleSclerosisLesionSegmentation": {
        "lesion": "multiple_sclerosis_of_the_brain"
    },
    "Task502_WMH": {
        "White_matter_hyperintensities": "white matter",
    },
    "Task503_BraTs2015": {
        "necrosis": "brain_necrosis",
    },
    "Task505_FeTA": {
        "Grey Matter":      "gray matter",
        "Deep Grey Matter": "deep gray matter",
    },
    "Task506_ISLES22_new": {
        "infarct lesions": "myocardial_infarction"
    },
    "Task507_Myops2020": {
        "left ventricular (LV) blood pool": "left ventricular blood pool",
        "LV normal myocardium":             "left ventricular myocardium",
        "LV myocardial scars":              "left ventricular myocardial scars",
        "LV myocardial edema":              "left_ventricular_myocardial_edema",
    },
    "Task510_ISLES_SISS": {
        "lesion": "sub-acute ischemic stroke lesion"
    },
    "Task514_ISLES17": {
        "infarct lesions": "ischemic_stroke_lesion"
    },
    "Task555_FeTA2022": {
        "Grey Matter" :      "gray_matter",
        "Deep Grey Matter" : "deep_gray_matter",
    },
    "Task556_FeTA2022_all": {    
        "Grey Matter" :      "gray_matter",
        "Deep Grey Matter" : "deep_gray_matter",
    },
    "Task560_WORD": {
        "right_kidney": "kidney_right",
        "left_kidney":  "kidney_left",
    },
    "Task590_Brain_PTM": {
        "matter_tracts": "white_matter"
    },
    "Task599_BraTS2021": {
        "necrosis": "brain_necrosis",
    },
    "Task601_CTSpine1K_Full": {
        "1":    "C1",
        "2":    "C2",
        "3":    "C3",
        "4":    "C4",
        "5":    "C5",
        "6":    "C6",
        "7":    "C7",
        "8":    "T1",
        "9":    "T2",
        "10":   "T3",
        "11":   "T4",
        "12":   "T5",
        "13":   "T6",
        "14":   "T7",
        "15":   "T8",
        "16":   "T9",
        "17":   "T10",
        "18":   "T11",
        "19":   "T12",
        "20":   "L1",
        "21":   "L2",
        "22":   "L3",
        "23":   "L4",
        "24":   "L5",
        "25":   "L6",
    },
    "Task602_ASC18": {
        "1": "left_atrial_cavity"
    },
    "Task603_MMWHS": {
        "1": "left ventricle blood cavity",
        "2": "right ventricle blood cavity",
        "3": "left atrial blood cavity",
        "4": "right atrial blood cavity",
        "5": "left ventricular myocardium",
        "6": "ascending aorta",
        "7": "pulmonary artery"
    },
    "Task606_orCaScore": {
        "1": "left_anterior_descending_branch",
        "2": "left_circumflex_artery",
        "3": "right_coronary_artery"
    },
    "Task611_PROMISE12": {
        "prostate lesion": "pancreatic_lesion"
    },
    "Task612_CTPelvic1k": {
        "right_hip": "hip_right",
        "left_hip": "hip_left",
    },
    "Task613_COVID-19-20": {
        "lung lesion": "ground-glass opacifications and consolidations",
    },
    "Task616_LNDb": {
        "lung node": "pulmonary_nodules",
    },
    "Task617_CAUSE07": {
        "left caudate": "caudate nucleaus",
        "right caudate": "caudate nucleaus",
    },
    "Task619_VESSEL2012": {
        "no (long) vessel" : "background",
        "(long) vessel" : "lung_vessels"
    },
    "Task620_MSseg08": {
        "multiple sclerosis" : "multiple_sclerosis_of_the_brain",
    },
    "Task624_Seg_Soft_Tissue": {
        "XXX": "soft_tissue_sarcomas",
        "XXX": "soft_tissue_sarcomas",
    },
    "Task625_EMIDEC": {
        "left myocardium":       "left ventricular myocardium",
    },
    "Task627_MRBrain18": {
        "Cortical gray matter": "gray matter",
        "Cerebrospinal fluid in the extracerebral space": "external cerebrospinal fluid",
        "Basal ganglia": "gray_matter",
        "White matter lesions": "white_matter",
        "Infarction": "brain_infarction",
    },
    "Task666_MESSEG": {
        "lesion": "multiple_sclerosis_of_the_brain",
    },
    "Task857_PICAI": {
        "fake mask": "pancreatic_lesion"
    },
    "Task888_Brats2013": {
        "necrosis": "brain_necrosis",
    },
    "Task891_RETOUCH": {
        "Pigment Epithelium Detachments": "pigment_epithelial_detachment"
    },
    "Task972_AutoCars": {
        "fake mask": "carotid_artery_vessel_wall"
    },
}