cd /mnt/lustre/share_data/gmai/nnUNet_preprocessed

target_dataset=Task322_AbdomenCT8K

aws --endpoint-url=http://10.140.14.2:80 --profile default s3 mb s3://nnUNet_preprocessed_${target_dataset}_nnUNetData_plans_General_stage0
aws --endpoint-url=http://10.140.14.2:80 --profile default s3 mb s3://nnUNet_preprocessed_${target_dataset}_nnUNetData_plans_General_stage1
# for DATASET in Task010_Colon Task009_Spleen
for DATASET in Task300_AbdomenSubset0 Task301_AbdomenSubset1 Task302_AbdomenSubset2 Task303_AbdomenSubset3 Task304_AbdomenSubset4 Task305_AbdomenSubset5 Task306_AbdomenSubset6 Task307_AbdomenSubset7 Task308_AbdomenSubset8 Task309_AbdomenSubset9 Task310_AbdomenSubset10 Task154_RibFrac Task029_LITS Task022_FLARE2022 Task021_KiTS2021 Task011_BTCV Task010_Colon Task009_Spleen Task008_HepaticVessel Task007_Pancreas Task003_Liver
do
if [ -d ./${DATASET}/nnUNetData_plans_General_stage0 ]; then
echo  uploading ${DATASET} \(stage0\)
aws --endpoint-url=http://10.140.14.2:80 --profile default s3 cp s3://nnUNet_preprocessed_${DATASET}_nnUNetData_plans_General_stage0 s3://nnUNet_preprocessed_${target_dataset}_nnUNetData_plans_General_stage0 --recursive
fi

if [ -d ./${DATASET}/nnUNetData_plans_General_stage1 ]; then
echo  uploading ${DATASET} \(stage1\)
aws --endpoint-url=http://10.140.14.2:80 --profile default s3 cp s3://nnUNet_preprocessed_${DATASET}_nnUNetData_plans_General_stage1 s3://nnUNet_preprocessed_${target_dataset}_nnUNetData_plans_General_stage1 --recursive
fi
done

cd -
