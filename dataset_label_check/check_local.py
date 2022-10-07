# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   20220919 10:09:36
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   check的主文件
'''

dataset_list = [
    "Task013_ACDC", 
    "Task032_AMOS22_Task1", 
    "Task040_ATM22", 
    "Task020_AbdomenCT1K", 
    "Task602_ASC18", 
    "Task011_BTCV", 
    "Task590_Brain_PTM", 
    "Task617_CAUSE07", 
    "Task037_CHAOS_Task_3_5_Variant1", 
    "Task525_CMRxMotions", 
    "Task612_CTPelvic1k", 
    "Task601_CTSpine1K_Full", 
    "Task625_EMIDEC", 
    "Task623_FLARE21", 
    "Task022_FLARE2022", 
    "Task505_FeTA", 
    "Task892_Heart_Seg_MRI", 
    "Task036_KiPA22", 
    "Task040_KiTS", 
    "Task021_KiTS2021", 
    "Task050_LAScarQS22_task1", 
    "Task051_LAScarQS22_task2", 
    "Task614_LUNA", 
    "Task603_MMWHS", 
    "Task627_MRBrain18", 
    "Task001_BrainTumour", 
    "Task002_Heart", 
    "Task003_Liver", 
    "Task004_Hippocampus", 
    "Task005_Prostate", 
    "Task006_Lung", 
    "Task007_Pancreas", 
    "Task008_HepaticVessel", 
    "Task009_Spleen", 
    "Task010_Colon", 
    "Task507_Myops2020", 
    "Task618_Promise09", 
    "Task039_Parse22", 
    "Task622_SLIVER07", 
    "Task619_VESSEL2012", 
    "Task083_VerSe2020", 
    "Task502_WMH", 
    "Task1000_cSeg-2022", 
    "Task606_orCaScore", 
    "Task029_LITS", 
    "Task621_Prostate_MRI_Segmentation_Dataset", 
    "Task605_SegThor", 
    "Task558_Totalsegmentator_dataset", 
    "Task560_WORD", 
    "Task890_iSeg2017", 
    "Task632_AutoImplant", 
    "Task611_PROMISE12", 
    "Task1006_hvsmr_2016", 
    "Task012_BTCV_Cervix", 
    "Task162_HeadandNeckAutoSegmentationChallenge", 
    "Task161_MRBrainS13DataNii", 
    "Task512_QUBIQ2020_braingrowth_subset", 
    "Task512_QUBIQ2020_braintumor_subset", 
    "Task512_QUBIQ2020_kidney_subset", 
    "Task512_QUBIQ2020_prostate_subset", 
]

dataset_modality_mapping = {
    "Task013_ACDC":                                     "MR",  
    "Task032_AMOS22_Task1":                             "CT", 
    "Task040_ATM22":                                    "CT"    , 
    "Task020_AbdomenCT1K":                              "CT"        , 
    "Task602_ASC18":                                    "MR"    , 
    "Task011_BTCV":                                     "CT", 
    "Task590_Brain_PTM":                                "MR"        , 
    "Task617_CAUSE07":                                  "MR"    , 
    "Task037_CHAOS_Task_3_5_Variant1":                  "MR"                    , 
    "Task525_CMRxMotions":                              "MR"        , 
    "Task612_CTPelvic1k":                               "CT"        , 
    "Task601_CTSpine1K_Full":                           "CT"            , 
    "Task625_EMIDEC":                                   "MR"    , 
    "Task623_FLARE21":                                  "CT"    , 
    "Task022_FLARE2022":                                "CT"        , 
    "Task505_FeTA":                                     "MR", 
    "Task892_Heart_Seg_MRI":                            "MR"            , 
    "Task036_KiPA22":                                   "CT"    , 
    "Task040_KiTS":                                     "CT", 
    "Task021_KiTS2021":                                 "CT"    , 
    "Task050_LAScarQS22_task1":                         "MR"            , 
    "Task051_LAScarQS22_task2":                         "MR"            , 
    "Task614_LUNA":                                     "CT", 
    "Task603_MMWHS":                                    "MR"    , 
    "Task627_MRBrain18":                                "MR"        , 
    "Task001_BrainTumour":                              "MR"        , 
    "Task002_Heart":                                    "MR"    , 
    "Task003_Liver":                                    "CT"    , 
    "Task004_Hippocampus":                              "MR"        , 
    "Task005_Prostate":                                 "CT"    , 
    "Task006_Lung":                                     "CT", 
    "Task007_Pancreas":                                 "CT"    , 
    "Task008_HepaticVessel":                            "CT"            , 
    "Task009_Spleen":                                   "CT"    , 
    "Task010_Colon":                                    "CT"    , 
    "Task507_Myops2020":                                "MR"        , 
    "Task618_Promise09":                                "MR"        , 
    "Task039_Parse22":                                  "CT"    , 
    "Task622_SLIVER07":                                 "CT"    , 
    "Task619_VESSEL2012":                               "CT"        , 
    "Task083_VerSe2020":                                "CT"        , 
    "Task502_WMH":                                      "MR", 
    "Task1000_cSeg-2022":                               "MR"        , 
    "Task606_orCaScore":                                "CT"        , 
    "Task029_LITS":                                     "CT", 
    "Task621_Prostate_MRI_Segmentation_Dataset":        "MR"                                , 
    "Task605_SegThor":                                  "CT"    , 
    "Task558_Totalsegmentator_dataset":                 "CT"                    , 
    "Task560_WORD":                                     "CT", 
    "Task890_iSeg2017":                                 "MR"    , 
    "Task632_AutoImplant":                              "CT"        , 
    "Task611_PROMISE12":                                "MR"        , 
    "Task1006_hvsmr_2016":                              "MR"        , 
    "Task012_BTCV_Cervix":                              "CT"        , 
    "Task162_HeadandNeckAutoSegmentationChallenge":     "CT"                                , 
    "Task161_MRBrainS13DataNii":                        "MR"                , 
    "Task512_QUBIQ2020_braingrowth_subset":             "MR"                        , 
    "Task512_QUBIQ2020_braintumor_subset":              "MR"                        , 
    "Task512_QUBIQ2020_kidney_subset":                  "CT"                    , 
    "Task512_QUBIQ2020_prostate_subset":                "CT"                        , 
}

import os.path as osp
import glob
import nibabel as nib
import numpy as np
import json

if __name__ == "__main__":
    data_dir = "/mnt/petrelfs/wanghaoyu/why/local_label"
    
    # dataset_list = dataset_list[:1]

    file = open("/mnt/cache/wanghaoyu/SP_script/dataset_label_check/result.txt", "w")
    print("dataset_list:", dataset_list)
    file.write("dataset_list: "+str(dataset_list)+"\n")
    for dataset in dataset_list:
        json_path =  osp.join(data_dir, dataset, "dataset.json")
        label_path = osp.join(data_dir, dataset, "labelsTr")
        label_list = glob.glob(osp.join(label_path, "*.nii.gz"))
        json_label = json.load(open(json_path))["labels"] 
        json_label_keys = list(map(int, json_label.keys()))
        print("dataset:", dataset)
        print("json keys:", json_label_keys)      
        file.write("dataset: "+str(dataset)+", json keys:"+str(json_label_keys)+"\n")
        # print("label_list:", label_list)
        for label in label_list:
            gt = nib.load(label).get_fdata()
            unique_value = list(map(int, np.unique(gt)))
            print(osp.basename(label), [v in json_label_keys for v in unique_value], unique_value)
            file.write(osp.basename(label)+": "+str([v in json_label_keys for v in unique_value])+" "+str(unique_value)+"\n")

