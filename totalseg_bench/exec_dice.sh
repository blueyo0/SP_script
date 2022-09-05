PREFIX="/mnt/cache/wanghaoyu/SP_script/totalseg_bench"
# bash test_compute_dice_cmd.sh Task011_BTCV fast all [Atten]
# bash test_compute_dice_cmd.sh Task011_BTCV normal all
# bash test_compute_dice_cmd.sh Task031_AMOS_MR normal all [NO split]
# bash test_compute_dice_cmd.sh Task031_AMOS_MR fast all
# bash test_compute_dice_cmd.sh Task009_Spleen normal all [NO split]
# bash test_compute_dice_cmd.sh Task009_Spleen fast all
# bash test_compute_dice_cmd.sh Task003_Liver normal all
# bash test_compute_dice_cmd.sh Task003_Liver fast all
# bash test_compute_dice_cmd.sh Task030_CT_ORG normal all
# bash test_compute_dice_cmd.sh Task030_CT_ORG fast all
# bash test_compute_dice_cmd.sh Task020_AbdomenCT1K fast all "test"
# bash test_compute_dice_cmd.sh Task020_AbdomenCT1K fast all  [NO split]
# bash test_compute_dice_cmd.sh Task020_AbdomenCT1K normal all 
# bash test_compute_dice_cmd.sh Task037_CHAOS_Task_3_5_Variant1 fast all
# bash test_compute_dice_cmd.sh Task037_CHAOS_Task_3_5_Variant1 normal all
# bash test_compute_dice_cmd.sh Task007_Pancreas fast all 
# bash test_compute_dice_cmd.sh Task007_Pancreas normal all 
# bash test_compute_dice_cmd.sh Task021_KiTS2021 fast all 
# bash test_compute_dice_cmd.sh Task021_KiTS2021 normal all 

# bash test_compute_dice_cmd.sh Task037_CHAOS_Task_3_5_Variant1 fast all "test"
# bash test_compute_dice_cmd.sh Task007_Pancreas fast all "test"
# bash test_compute_dice_cmd.sh Task021_KiTS2021 fast all "test"

# bash test_compute_dice_cmd.sh Task021_KiTS2021 fast 0
# bash test_compute_dice_cmd.sh Task021_KiTS2021 normal 0
# bash test_compute_dice_cmd.sh Task007_Pancreas fast 0
# bash test_compute_dice_cmd.sh Task007_Pancreas normal 0
# bash test_compute_dice_cmd.sh Task037_CHAOS_Task_3_5_Variant1 fast 0
# bash test_compute_dice_cmd.sh Task037_CHAOS_Task_3_5_Variant1 normal 0
# bash test_compute_dice_cmd.sh Task030_CT_ORG fast 0
# bash test_compute_dice_cmd.sh Task030_CT_ORG normal 0
# bash test_compute_dice_cmd.sh Task003_Liver fast 0
# bash test_compute_dice_cmd.sh Task003_Liver normal 0

# bash test_compute_dice_cmd.sh Task558_Totalsegmentator_dataset fast 0 
# bash test_compute_dice_cmd.sh Task558_Totalsegmentator_dataset normal 0
# bash test_compute_dice_cmd.sh Task558_Totalsegmentator_dataset fast 1 
# bash test_compute_dice_cmd.sh Task558_Totalsegmentator_dataset normal 1

# bash test_compute_dice_cmd.sh Task011_BTCV fast 0 
# bash test_compute_dice_cmd.sh Task011_BTCV normal 0


# bash test_compute_dice_cmd.sh Task009_Spleen fast 0
# bash test_compute_dice_cmd.sh Task009_Spleen normal 0
# bash test_compute_dice_cmd.sh Task020_AbdomenCT1K fast 0
# bash test_compute_dice_cmd.sh Task020_AbdomenCT1K normal 0
# bash test_compute_dice_cmd.sh Task031_AMOS_MR fast 0
# bash test_compute_dice_cmd.sh Task031_AMOS_MR normal 0
# bash test_compute_dice_cmd.sh Task083_VerSe2020 fast 0 "test"
# bash test_compute_dice_cmd.sh Task083_VerSe2020 fast 0
# bash test_compute_dice_cmd.sh Task083_VerSe2020 normal 0
# bash test_compute_dice_cmd.sh Task083_VerSe2020 fast all
# bash test_compute_dice_cmd.sh Task083_VerSe2020 normal all
# bash test_compute_dice_cmd.sh Task020_AbdomenCT1K fast all
# bash test_compute_dice_cmd.sh Task020_AbdomenCT1K normal all

# bash $PREFIX/test_compute_dice_cmd.sh Task032_AMOS22_Task1 fast all
# bash $PREFIX/test_compute_dice_cmd.sh Task032_AMOS22_Task1 normal all 
# bash $PREFIX/test_compute_dice_cmd.sh Task022_FLARE22 fast all
# bash $PREFIX/test_compute_dice_cmd.sh Task022_FLARE22 normal all 

# bash $PREFIX/test_compute_nn_dice_cmd.sh Task011_BTCV BigResUNetTrainer4 0
# bash $PREFIX/test_compute_nn_dice_cmd.sh Task020_AbdomenCT1K BigResUNetTrainer4 0
# bash $PREFIX/test_compute_nn_dice_cmd.sh Task003_Liver BigResUNetTrainer4 0
# bash $PREFIX/test_compute_nn_dice_cmd.sh Task007_Pancreas BigResUNetTrainer4 0
# bash $PREFIX/test_compute_nn_dice_cmd.sh Task009_Spleen BigResUNetTrainer4 0
# bash $PREFIX/test_compute_nn_dice_cmd.sh Task021_KiTS2021 BigResUNetTrainer4 0
# # bash $PREFIX/test_compute_nn_dice_cmd.sh Task154_RibFrac BigResUNetTrainer4 0
# # bash $PREFIX/test_compute_nn_dice_cmd.sh Task083_VerSe2020 BigResUNetTrainer4 0
# # bash $PREFIX/test_compute_nn_dice_cmd.sh Task030_CT_ORG BigResUNetTrainer4 0
# bash $PREFIX/test_compute_nn_dice_cmd.sh Task037_CHAOS_Task_3_5_Variant1 BigResUNetTrainer4 0
# bash $PREFIX/test_compute_nn_dice_cmd.sh Task031_AMOS_MR BigResUNetTrainer4 0
# # bash $PREFIX/test_compute_nn_dice_cmd.sh Task558_Totalsegmentator_dataset BigResUNetTrainer4 0
# bash $PREFIX/test_compute_nn_dice_cmd.sh Task032_AMOS22_Task1 BigResUNetTrainer4 0
# bash $PREFIX/test_compute_nn_dice_cmd.sh Task022_FLARE22 BigResUNetTrainer4 0

# bash $PREFIX/test_compute_nn_dice_cmd.sh Task558_Totalsegmentator_dataset BigResUNetTrainer4 all
# bash $PREFIX/test_compute_nn_dice_cmd.sh Task558_Totalsegmentator_dataset BigResUNetTrainer1 all
# bash $PREFIX/test_compute_nn_dice_cmd.sh Task558_Totalsegmentator_dataset nnUNetTrainerV2_S6_D3_W64 all
bash $PREFIX/test_compute_nn_dice_cmd.sh Task558_Totalsegmentator_dataset nnUNetTrainerV2_S6_D3_W32 all
# bash $PREFIX/test_compute_nn_dice_cmd.sh Task558_Totalsegmentator_dataset nnUNetTrainerV2_S6_D2_W64 all
# bash $PREFIX/test_compute_nn_dice_cmd.sh Task558_Totalsegmentator_dataset nnUNetTrainerV2_S6_D2_W32 all
# bash $PREFIX/test_compute_nn_dice_cmd.sh Task558_Totalsegmentator_dataset nnUNetTrainerV2_S6_D2_W16 all
# bash $PREFIX/test_compute_nn_dice_cmd.sh Task558_Totalsegmentator_dataset nnUNetTrainerV2_S5_D3_W64 all
# bash $PREFIX/test_compute_nn_dice_cmd.sh Task558_Totalsegmentator_dataset nnUNetTrainerV2_S5_D2_W64 all
# bash $PREFIX/test_compute_nn_dice_cmd.sh Task558_Totalsegmentator_dataset nnUNetTrainerV2_S5_D3_W32 all
# bash $PREFIX/test_compute_nn_dice_cmd.sh Task558_Totalsegmentator_dataset nnUNetTrainerV2_S5_D2_W32 all
# bash $PREFIX/test_compute_nn_dice_cmd.sh Task558_Totalsegmentator_dataset nnUNetTrainerV2_S5_D2_W16 all
# bash $PREFIX/test_compute_nn_dice_cmd.sh Task558_Totalsegmentator_dataset nnUNetTrainerV2_S4_D3_W64 all
# bash $PREFIX/test_compute_nn_dice_cmd.sh Task558_Totalsegmentator_dataset nnUNetTrainerV2_S4_D3_W32 all
# bash $PREFIX/test_compute_nn_dice_cmd.sh Task558_Totalsegmentator_dataset nnUNetTrainerV2_S4_D2_W64 all
