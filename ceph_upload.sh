cd /mnt/lustre/share_data/hejunjun/nnUNet/nnUNet_preprocessed

# for DATASET in Task006_Lung Task009_Spleen Task010_Colon Task011_BTCV Task037_CHAOS_Task_3_5_Variant1
# for DATASET in Task008_HepaticVessel Task007_Pancreas
for DATASET in Task154_RibFrac
do
if [ -d ./${DATASET}/nnUNetData_plans_v2.1_stage0 ]; then
echo  uploading ${DATASET} \(stage0\)
aws s3 mb s3://nnUNet_preprocessed_${DATASET}_nnUNetData_plans_v2.1_stage0
aws s3 cp ./${DATASET}/nnUNetData_plans_v2.1_stage0 s3://nnUNet_preprocessed_${DATASET}_nnUNetData_plans_v2.1_stage0 --recursive
fi

if [ -d ./${DATASET}/nnUNetData_plans_v2.1_stage1 ]; then
echo  uploading ${DATASET} \(stage1\)
aws s3 mb s3://nnUNet_preprocessed_${DATASET}_nnUNetData_plans_v2.1_stage1
aws s3 cp ./${DATASET}/nnUNetData_plans_v2.1_stage1 s3://nnUNet_preprocessed_${DATASET}_nnUNetData_plans_v2.1_stage1 --recursive
fi
done

cd -
