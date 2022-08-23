# aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 cp s3://dataset_raw_labeled/AbdomenCT /mnt/lustre/share_data/gmai/dataset/raw/labeled/ 
# aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 cp s3://dataset_raw_labeled/BTCV /mnt/lustre/share_data/gmai/dataset/raw/labeled/
# aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 ls s3://dataset_preprocess_labeled/AbdomenCT
# aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 ls s3://dataset_raw_labeled/BTCV
# aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 ls s3://dataset_raw_labeled/kits21
# aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 ls s3://dataset_raw_labeled/kits19
# aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 ls s3://dataset_raw_labeled/MSD03_Liver
# aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 ls s3://dataset_raw_labeled/MSD07_Pancreas
# aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 ls s3://dataset_raw_labeled/MSD08_HepaticVessel
# aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 ls s3://dataset_raw_labeled/MSD09_Spleen
# aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 ls s3://dataset_raw_labeled/MSD10_Colon
# aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 ls s3://dataset_raw_labeled/CHAOS
# aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 ls s3://dataset_raw_labeled/FLARE22
# aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 ls s3://dataset_raw_labeled/RibFrac2020

# for DATASET in TCGA-COAD CT_COLONOGRAPHY C4KC-KiTS TCGA-LIHC TCGA-KIRC
# do
# aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 ls s3://dataset_preprocessed_unlabeled/${DATASET}
# if [ $? -ne 0 ]; then
#     echo "failed"
# else
#     echo "succeed"
#     rm -rf /mnt/lustre/share_data/gmai/dataset/preprocessed/unlabeled/${DATASET}
# fi
# done

# for DATASET in Task321_AbdomenCT5K Task322_AbdomenCT8K
# # for DATASET in Task040_KiTS
# do
cd /mnt/cache/wanghaoyu/data/PREPROCESSED/${DATASET}/nnUNetData_plans_General_stage1/
filenames=`ls *.npy`
for eachfile in $filenames
do
#    echo $eachfile
check_result=`aws --profile=default --endpoint-url=http://10.140.14.254:80 s3 ls s3://nnUNet_preprocessed_${DATASET}_nnUNetData_plans_General_stage1/$eachfile`
if [ $? -ne 0 ]; then
    echo "failed $eachfile"
else
    :
    # echo "success"
fi
done
# cd -
# echo "${DATASET} done"
# done
# echo "all done"
# cd /mnt/lustre/share_data/hejunjun/nnUNet/nnUNet_raw_data_base
# for DATASET in Task003_Liver Task007_Pancreas Task008_HepaticVessel Task009_Spleen Task010_Colon Task022_FLARE2022 Task037_CHAOS_Task_3_5_Variant1 Task154_RibFrac
# for DATASET in Task003_Liver Task009_Spleen Task010_Colon
# do
# cp -r nnUNet_raw_data/${DATASET} /mnt/lustre/share_data/gmai/nnUNet_raw_data_base/nnUNet_raw_data/${DATASET}
# cp -r nnUNet_cropped_data/${DATASET} /mnt/lustre/share_data/gmai/nnUNet_raw_data_base/nnUNet_cropped_data/${DATASET}
# done
# cd -
# Ab1K  
# BTCV 
# MSD03_Liver
# MSD07_Pancreas
# MSD08_HepaticVessel
# MSD09_Spleen
# MSD10_Colon
# CHAOS
# FLARE22_labeled
# KiTS21
# LiTS [x]
# RibFrac2020
# KiTS19