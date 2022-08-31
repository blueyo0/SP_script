FLARE22_sys = {
    # "0":  "background",
    "1":  "liver",
    "10": "esophagus",
    "11": "stomach",
    "12": "duodenum",
    "13": "left kidney",
    "2":  "right kidney",
    "3":  "spleen",
    "4":  "pancreas",
    "5":  "aorta",
    "6":  "IVC",
    "7":  "RAG",
    "8":  "LAG",
    "9":  "gallbladder"
}
LAScarQS_sys = {
    # "0": "background",
    "1": "LA"
}
AMOS_sys = {
    # "0": "background", 
    "1": "spleen", 
    "2": "right kidney", 
    "3": "left kidney", 
    "4": "gallbladder", 
    "5": "esophagus", 
    "6": "liver", 
    "7": "stomach", 
    "8": "aorta", 
    # "9": "postcava",
    "10": "pancreas", 
    "11": "RAG", 
    "12": "LAG", 
    "13": "duodenum", 
    "14": "bladder", 
    # "15": "prostate/uterus"
}

KIPA_sys = {
    #"0": "background",
    #"1": "renal vein",
    "2": "kidney",
    #"3": "renal artery",
    #"4": "tumor"
},

CMR_sys = {
   # "0": "background",
    "1": "LV",
   # "2": "MYO",
    "3": "RV"
}

Task003_Liver_sys = { 
#    "0": "background", 
   "agg:1,2": "liver", 
#    "1": "liver", 
#    "2": "cancer"
 }

Task011_BTCV_sys = {
    # "00": "background",
    "1": "spleen",
    "2": "right kidney",
    "3": "left kidney",
    "4": "gallbladder",
    "5": "esophagus",
    "6": "liver",
    "7": "stomach",
    "8": "aorta",
    "9": "inferior vena cava",
    "10": "portal vein and splenic vein",
    "11": "pancreas",
    "12": "right adrenal gland",
    "13": "left adrenal gland"
}

Task020_AbdomenCT1K_sys = {
    # "0": "background",
    "1": "liver",
    "2": "kidney",
    "3": "spleen",
    "4": "pancreas"
}


Task007_Pancreas_sys = {
#    "0": "background",
   "agg:1,2": "pancreas",
#    "1": "pancreas",
#    "2": "cancer"
 }

Task009_Spleen_sys = {
#    "0": "background",
   "1": "spleen"
 }

Task021_KiTS2021_sys = {
    # "0": "background",
    "agg:1,2,3": "kidney", 
    # "1": "kidney",
    # "2": "tumor",
    # "3": "cyst"
}

# [TO-DO] Rib label
Task154_RibFrac_sys = {

}

# [TO-DO] Verse label wrong
Task083_VerSe2020_sys = {
    # "0": "0",
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
    # "25": "L6",
    # "26": "sacrum",
    # "27": "cocygis",
    # "28": "T13"
}

Task030_CT_ORG_sys = {
    # "0": "background",
    "1": "liver",
    "2": "bladder",
    "3": "lung",
    "4": "kidney",
    # "5": "bone",
    "6": "brain"
}

Task037_CHAOS_Task_3_5_Variant1_sys = {
    # "0": "background",
    "1": "liver",
    "2": "right kidney",
    "3": "left kidney",
    "4": "spleen"
}


Task031_AMOS_MR_sys = AMOS_sys

label_mapping = {
    "lung":                         "agg:lung_lower_lobe_left,lung_lower_lobe_right,lung_middle_lobe_right,lung_upper_lobe_left,lung_upper_lobe_right",
    "liver":                        "liver",
    "esophagus":                    "esophagus",
    "stomach":                      "stomach",
    "duodenum":                     "duodenum", 
    "kidney":                       "agg:kidney_left,kidney_right",
    "left kidney":                  "kidney_left",
    "right kidney":                 "kidney_right",
    "spleen":                       "spleen",
    "pancreas":                     "pancreas",
    "aorta":                        "aorta",
    "IVC":                          "inferior_vena_cava",
    "RAG":                          "adrenal_gland_right",
    "LAG":                          "adrenal_gland_left",
    "gallbladder":                  "gallbladder",
    "bladder":                      "urinary_bladder",
    "LV":                           "heart_ventricle_left",
    "RV":                           "heart_ventricle_right",
    "LA":                           "heart_atrium_left",
    "brain":                        "brain",
    "inferior vena cava":           "inferior_vena_cava",
    "portal vein and splenic vein": "portal_vein_and_splenic_vein",
    "right adrenal gland":          "adrenal_gland_right",
    "left adrenal gland":           "adrenal_gland_left",
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
}

label_sys_dict = {
    "AMOS": AMOS_sys, 
    "FLARE22": FLARE22_sys, 
    "CMRxMotion": CMR_sys, 
    "KIPA": KIPA_sys, 
    "LAScarQS": LAScarQS_sys,
    "Task011_BTCV": Task011_BTCV_sys,
    "Task020_AbdomenCT1K": Task020_AbdomenCT1K_sys,
    "Task003_Liver": Task003_Liver_sys,
    "Task007_Pancreas": Task007_Pancreas_sys,
    "Task009_Spleen": Task009_Spleen_sys,
    "Task021_KiTS2021": Task021_KiTS2021_sys,
    "Task154_RibFrac": Task154_RibFrac_sys,
    "Task083_VerSe2020": Task083_VerSe2020_sys,
    "Task030_CT_ORG": Task030_CT_ORG_sys,
    "Task037_CHAOS_Task_3_5_Variant1": Task037_CHAOS_Task_3_5_Variant1_sys,
    "Task031_AMOS_MR": Task031_AMOS_MR_sys,
}

