# cd /mnt/lustre/share_data/gmai/dataset/preprocessed/unlabeled
cd /mnt/lustre/share_data/gmai/dataset/raw/unlabeled

# for DATASET in ACDC ADAM AGE19 AIDA-E_3 AIROGS AMD-OCT AMOS22.zip APTOS APTOS2019 APTOS_Cross-Country ATLAS_2 Atrial_Segmentation_Challenge Atrial_Segmentation_Challenge18 AutoImplant Blood_Cell_Images_dataset braimMRI Brain-Mri BrainPTM 2021 Brain_Tumor_Progression Brats18 Brats19 BraTS2015 BraTS_2015 BraTS2021 Breast_Cancer_cell_seg BreastMNIST BTCV CAMELYON17 Camelyon2017 Cancer_instance_Segmentation Carotid_Artery Cell_tracking19 CellTracking20 CHAOS CHASE_DB1 Chest_CT_Scan Chest_Image_Pneum Chest-X-ray Chest-X-ray-PA CMRxMotion CNI19 Continuous_Registration Continuous Registration-dcm Corona Coronahack Covid-19 COVID-19-20_v2 COVID-19_cd Covid-19-CT COVID-19-CT-CXR COVID-19-CT_SCAN_IMAGES Covid-19-Image-Dataset_datasets COVID-19_Lung_CT_Scans COVID_CT_COVID-CT COVIDxCXR-2 Cranium CREMI Crossmoda CRT-EPiggy19 CT_medical_Images CT-Med-Im CTPelvic1K CT-Scan CTSpine1K curious18 curious2019 cvcclinicdb Data-Sci-Bow DDTI Derm7pt DFUC2021 Diabetic DIARETDB DigestPath19 DR-HAGIS Drishti-GS1_files DRIVE DRIVE_Retinal Duke-OCT EAD19 EchoNet-Dynamic EMIDEC EMPIRE10 EndoCV2020 EndoSlam EndoVis15 EndoVis2019 Eye_OCT EyePACS FDG-PET-CT-Lesions FeTA fizpatrick17k FLARE FLARE22 Gleason Heart_Seg_MRI HECKTOR2021 His_Can_Det HRF HuBMAP IDRID Instance22 Intel_MobileODT IRMA iSeg ISIC18 ISIC19 ISIC20 ISIC 2016 ISIC_2016 ISLES ISLES2016 ISLES2017 ISLES22 KiPA22 kits19 kits21 KNOAP2020 Kvasir LAScarQS22 Learn2Reg Leukemia_classification LiTS2017 Liver_Organ_Tumor_seg Liver_Tumor_Seg LNDb LoDoPaB LOLA1 LUNA M2CAI16-tool MAlig_Lymph_Cls medmnist MED-NODE MESSEG Messidor Metastatic MIAS MITOS-ATYPIA-14 MM-WHS MoNuSAC20 MoNuSeg Mos_COV
# for DATASET in AbdomenCT
# for DATASET in MSD01_BrainTumour MSD02_Heart MSD03_Liver MSD04_Hippocampus MSD05_Prostate MSD06_Lung MSD07_Pancreas MSD08_HepaticVessel MSD09_Spleen MSD10_Colon Naturalistic oct2017 OCT2017_ Ocular_Dis_rec ODIR orCaScore OrganSegmentations OSIC_Pul_fib_Pro osic-pulmonary-fibrosis-progression PAD-UFES-20 PAIP2021 PALM19 PANDA Parse22 PI-CAI Pneumonia-Chest-X-ray PROMISE12 Pul_Chest_X_ray_Abno Pul_Emb_CT_Image Pulmonary_Embolism pvqa QUBIQ2021 Refuge RESECT Retina_Fundus_Image_Reg Retina-OCT-C8 RETOUCH RIADD RibFrac RibFrac2020 RSAN-PDC rsna-intracranial-hemorrhage-detection RSNA-MICCAI-BTRC RSNA-STR-Pul-Emb-Det SARAS-MESAD SARS-COV-2 Sec-Ann-Data-Sci-Bow Seg_Soft_Tissue siim-acr-pneumothorax skin_lesion Sliver SLN-Breast STACOM-SLAWT STARE SurgVisDom Task083_VerSe2020 Task501_Brain_PTM Task502_WMH Task503_BraTs2015 Task504_ATLAS Task505_FeTA Task506_WIV_01 Task508_WIV_03 Task509_BraTS Task510_ISLES_SISS Task511_ISLES_SPES Task666_MESSEG TBI Ult-Ner-Seg UW-Madison VerSE VerSe19 VinBigData VWS2021 Where_is_VALDO WMH workflow_m2cai2016
# for DATASET in ADAM AGE19 AIDA-E_3 AIROGS AMD-OCT AMOS22 APTOS APTOS2019 APTOS_Cross-Country ATLAS_2 Atrial_Segmentation_Challenge Atrial_Segmentation_Challenge18 AutoImplant Blood_Cell_Images_dataset braimMRI Brain-Mri BrainPTM2021 Brain_Tumor_Progression Brats18 Brats19 BraTS2015 BraTS_2015 BraTS2021 Breast_Cancer_cell_seg BreastMNIST BTCV Camelyon2017 Cancer_instance_Segmentation Carotid_Artery Cell_tracking19 CellTracking20 CHASE_DB1 Chest_CT_Scan Chest_Image_Pneum Chest-X-ray Chest-X-ray-PA CMRxMotion CNI19 Continuous_Registration Continuous Registration-dcm Corona Coronahack Covid-19 COVID-19-20_v2 COVID-19_cd Covid-19-CT COVID-19-CT-CXR COVID-19-CT_SCAN_IMAGES Covid-19-Image-Dataset_datasets COVID-19_Lung_CT_Scans COVID_CT_COVID-CT COVIDxCXR-2 Cranium CREMI Crossmoda CRT-EPiggy19 CT_medical_Images CT-Med-Im CTPelvic1K CT-Scan CTSpine1K curious18 curious2019 cvcclinicdb Data-Sci-Bow DDTI Derm7pt DFUC2021 Diabetic DIARETDB DigestPath19 DR-HAGIS Drishti-GS1_files DRIVE DRIVE_Retinal Duke-OCT EAD19 EchoNet-Dynamic EMIDEC EMPIRE10 EndoCV2020 EndoSlam EndoVis15 EndoVis2019 Eye_OCT EyePACS FDG-PET-CT-Lesions FeTA fizpatrick17k FLARE FLARE22 Gleason Heart_Seg_MRI HECKTOR2021 His_Can_Det HRF HuBMAP IDRID Instance22 Intel_MobileODT IRMA iSeg ISIC18 ISIC19 ISIC20 ISIC 2016 ISIC_2016 ISLES ISLES2016 ISLES2017 ISLES22 KiPA22 kits19 kits21 KNOAP2020 Kvasir LAScarQS22 Learn2Reg Leukemia_classification LiTS2017 Liver_Organ_Tumor_seg Liver_Tumor_Seg LNDb LoDoPaB LOLA1 LUNA M2CAI16-tool MAlig_Lymph_Cls medmnist MED-NODE MESSEG Messidor Metastatic MIAS MITOS-ATYPIA-14 MM-WHS MoNuSAC20 MoNuSeg Mos_COV Naturalistic oct2017 OCT2017_ Ocular_Dis_rec ODIR orCaScore OrganSegmentations OSIC_Pul_fib_Pro osic-pulmonary-fibrosis-progression PAD-UFES-20 PAIP2021 PALM19 PANDA Parse22 PICAI PI-CAI Pneumonia-Chest-X-ray PROMISE12 Pul_Chest_X_ray_Abno Pul_Emb_CT_Image Pulmonary_Embolism pvqa QUBIQ2021 Refuge RESECT Retina_Fundus_Image_Reg Retina-OCT-C8 RETOUCH RIADD RibFrac RibFrac2020 RSAN-PDC RSNA-MICCAI-BTRC RSNA-STR-Pul-Emb-Det SARAS-MESAD SARS-COV-2 Sec-Ann-Data-Sci-Bow Seg_Soft_Tissue siim-acr-pneumothorax skin_lesion Sliver SLN-Breast STACOM-SLAWT STARE SurgVisDom Task083_VerSe2020 Task501_Brain_PTM Task502_WMH Task503_BraTs2015 Task504_ATLAS Task505_FeTA Task506_WIV_01 Task508_WIV_03 Task509_BraTS Task510_ISLES_SISS Task511_ISLES_SPES Task666_MESSEG TBI Ult-Ner-Seg UW-Madison VerSE VerSe19 VinBigData VWS2021 Where_is_VALDO WMH
# for DATASET in Continuous_Registration_dcm curious2019
# for DATASET in ACRIN-HNSCC-FDG-PET-CT  DATA  Data_data  IXI  ProstateDiagnosis  QIN-BRAIN-DSC-MRI  QIN-HEADNECK  TCGA-COAD  TCGA-GBM  TCGA-LUSC  TCGA-OV  TCGA-STAD  tmp
# for DATASET in AbdomenCT  AMOS22  CHAOS           get_info.py  KiTS21           LNDb    MSD01_BrainTumour  MSD03_Liver        MSD05_Prostate  MSD07_Pancreas       MSD09_Spleen  Parse22 UW-Madison ACDC BTCV COVID-19-20_v2  ISLES2015 Liver_Tumor_Seg  LOLA11  MSD02_Heart MSD04_Hippocampus  MSD06_Lung      MSD08_HepaticVessel  MSD10_Colon
# for DATASET in  AAPM-RT-MAC ACRIN-6667 ACRIN6668 ACRIN-DSC-MR-Brain ACRIN-FLT-Breast ACRIN-FMISO-Brain ACRIN-HNSCC-FDG-PET-CT Anti-PD-1_Lung Anti-PD-1_MELANOMA Breast-Diagnosis Breast-MRI-NACT-Pilot C4KC-KiTS CC-Radiomics-Phantom-3 Colin3T7T COVID-19-AR COVID-19-NY-SBU CPTAC_CCRCC CPTAC-CM CPTAC-GBM CPTAC-HNSCC CPTAC-LUAD CPTAC-PDA CPTAC-SAR CPTAC-UCEC CT_COLONOGRAPHY CT-Covid-19-August2020 CT_Lymph_Nodes ds000030 ds000115 ds000158 ds000201 ds000202 ds000221 ds000228 ds000243 ds000258 ds001461 ds001486 ds001720 ds001734 ds001747 ds002236 ds002320 ds002345 ds002380 ds002385 ds002837 ds002843 ds002870 ds003097 ds003145 ds003346 ds003416 ds003469 ds003481 ds003499 ds003592 ds003599 ds003604 ds003653 ds003701 ds003798 ds003826 ds003949 Duke-Breast-Cancer-MRI GLIS-RT Head-Neck_Cetuximab Head-Neck-PET-CT Head-Neck-Radiomics-HN1 ISLES ISPY1 IXI knee_mri_clinical_seq LCTSC LGG-1p19qDeletion LIDC-IDRI-ALL-CT LungCT-Diagnosis Lung-PET-CT-Dx MIDRC-RICORD-1A MIDRC-RICORD-1b NLST NSCLC-Cetuximab NSCLC_Radiogenomics NSCLC-Radiomics-Genomics NSCLC-Radiomics-Interobserver1 OPC-Radiomics OrganSegmentations Pancreas-CT Pancreatic-CT-CBCT-SEG Pediatric-CT-SEG Pelvic-Reference-Data ProstateDiagnosis Prostate-MRI-US-Biopsy PROSTATEx QIN-BRAIN-DSC-MRI QIN-Breast QIN-HEADNECK REMBRANDT RIDER-Lung-PET-CT SIMON Soft-tissue-Sarcoma SPIE-AAPM-Lung-CT-Challenge STOIC TCGA-BLCA TCGA-BRCA TCGA-CESC TCGA-COAD TCGA-GBM TCGA-HNSC TCGA-KIRC TCGA-KIRP TCGA-LGG TCGA-LIHC TCGA-LUAD TCGA-LUSC TCGA-OV TCGA-STAD TCGA-UCEC TCIA-Covid-19 tmp Vestibular-Schwannoma-SEG
for DATASET in ACRIN_6667 CPTAC-PDA MIDRC-RICORD-1A MIDRC-RICORD-1B PROSTATEx TCGA-KIRC-dcm TCGA-KIRP 
do

if [ -d ./${DATASET} ]; then
echo  uploading ${DATASET} \(stage1\)
# aws --profile why s3 mb s3://${DATASET}
# aws --profile why s3 cp ./${DATASET}/ s3://${DATASET} --recursive
aws --profile why --endpoint-url=http://10.140.14.2:80 s3 cp ./${DATASET}/ s3://dataset_preprocessed_unlabeled/${DATASET} --recursive
fi
done

cd -
