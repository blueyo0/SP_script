# cd /mnt/cache/wanghaoyu/data/PREPROCESSED/${DATASET}/nnUNetData_plans_General_stage1/
# filenames=`ls Task*`
for DATASET in Task001_BrainTumour   Task002_Heart   Task003_Liver   Task004_Hippocampus   Task005_Prostate   Task006_Lung   Task007_Pancreas   Task008_HepaticVessel   Task009_Spleen   Task010_Colon   Task011_BTCV   Task013_ACDC   Task017_BTCV   Task019_BraTS21   Task020_AbdomenCT1K   Task021_KiTS2021   Task022_FLARE2022   Task022_FLARE22   Task023_FLARE22_PL   Task024_FLARE22_all   Task025_FLARE22_all_ite2   Task027_ACDC   Task029_LITS   Task030_CT_ORG   Task031_AMOS_MR   Task032_AMOS22_Task1   Task032_AMOS_CT   Task033_AMOS22_Task2   Task033_AMOS22_Task2_result2   Task033_AMOS22_task2   Task033_AMOS_Task2   Task034_Instance22   Task036_KiPA22   Task037_CHAOS_Task_3_5_Variant1   Task037_LASCarQS22_task1   Task038_LAScarQS22   Task039_Parse22   Task040_ATM22   Task040_KiTS   Task043_CrossMoDA22   Task050_LAScarQS22_task1   Task051_LAScarQS22_task2   Task083_VerSe2020   Task095_Ab_tiny   Task096_Ab_tiny   Task101_AMOS22_Task1   Task102_AMOS22_Task2   Task103_FLARE22_ite1_threshold03   Task105_FLARE22_ite1_threshold05   Task107_FLARE22_ite1_threshold07   Task109_FLARE22_ite1_threshold09   Task129_FLARE22_ite2_th09   Task154_RibFrac   Task306_AbdomenSubset6   Task320_test_agg   Task321_AbdomenCT5K   Task322_AbdomenCT8K   Task354_AbdomenTinySubset4   Task501_ISLES22_01   Task504_ATLAS   Task505_FeTA   Task506_ISLES22_new   Task507_WIV_02   Task509_WIV_01   Task510_ISLES_SISS   Task511_ISLES_SPES   Task514_ISLES17   Task522_CMR   Task555_FeTA2022   Task556_FeTA2022_all   Task597_ISLES22_adc   Task598_ISLES22_dwi   Task599_BraTS2021   Task600_CTSpine1K   Task601_CTSpine1K_Full   Task602_ASC18   Task603_MMWHS   Task605_SegThor   Task606_orCaScore   Task609_Spine1K   Task610_Cardiac1K   Task611_FullBody1K   Task615_Chest_CT_Scans_with_COVID-19   Task616_LNDb   Task706_LIDC-IDRI-ALL-CT   Task710_autoPET   Task715_RIDER-Lung-PET-CT   Task716_SPIE-AAPM-Lung-CT-Challenge 
# for DATASET in Task003_Liver 
do
echo $DATASET
for TRAINER in nnUNetTrainerV2_S4_D2_W16__nnUNetPlansv2.1   nnUNetTrainerV2_S4_D2_W32__nnUNetPlansv2.1   nnUNetTrainerV2_S4_D2_W64__nnUNetPlansv2.1   nnUNetTrainerV2_S4_D3_W32__nnUNetPlansv2.1   nnUNetTrainerV2_S4_D3_W64__nnUNetPlansv2.1   nnUNetTrainerV2_S5_D2_W16__nnUNetPlansv2.1   nnUNetTrainerV2_S5_D2_W32__nnUNetPlansv2.1   nnUNetTrainerV2_S5_D2_W64__nnUNetPlansv2.1   nnUNetTrainerV2_S5_D3_W32__nnUNetPlansv2.1   nnUNetTrainerV2_S5_D3_W64__nnUNetPlansv2.1   nnUNetTrainerV2_S6_D2_W16__nnUNetPlansv2.1   nnUNetTrainerV2_S6_D2_W32__nnUNetPlansv2.1   nnUNetTrainerV2_S6_D2_W64__nnUNetPlansv2.1   nnUNetTrainerV2_S6_D3_W32__nnUNetPlansv2.1   nnUNetTrainerV2_S6_D3_W64__nnUNetPlansv2.1
do
for FOLD in fold_0 fold_1 fold_2 fold_3 fold_4
do
eachfile="s3://nnUNet_trained_models/nnUNet/3d_fullres/$DATASET/$TRAINER/$FOLD/model_final_checkpoint.model"
check_result=`aws --profile=default --endpoint-url=http://10.140.14.254:80 s3 ls $eachfile`
if [ $? -ne 0 ]; then
    echo "NOTFOUND $eachfile"
else
    echo "FOUND $eachfile"
fi
done
done
done