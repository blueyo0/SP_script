cd /mnt/lustre/share_data/gmai/nnUNet_preprocessed

# for DATASET in Task006_Lung Task009_Spleen Task010_Colon Task011_BTCV Task037_CHAOS_Task_3_5_Variant1
# for DATASET in Task008_HepaticVessel Task007_Pancreas
# for DATASET in Task029_LITS
# for DATASET in Task095_Ab_tiny
# for DATASET in Task154_RibFrac
for DATASET in Task101_AMOS22_Task1 Task102_AMOS22_Task2
do
if [ -d ./${DATASET}/nnUNetData_plans_v2.1_stage0 ]; then
# if [ -d ./${DATASET}/nnUNetData_plans_General_stage0 ]; then
echo  uploading ${DATASET} \(stage0\)
aws --endpoint-url=http://10.140.14.2:80 --profile default s3 mb s3://nnUNet_preprocessed_${DATASET}_nnUNetData_plans_v2.1_stage0
aws --endpoint-url=http://10.140.14.2:80 --profile default s3 cp ./${DATASET}/nnUNetData_plans_v2.1_stage0 s3://nnUNet_preprocessed_${DATASET}_nnUNetData_plans_v2.1_stage0 --recursive
# aws --endpoint-url=http://10.140.14.2:80 --profile default s3 mb s3://nnUNet_preprocessed_${DATASET}_nnUNetData_plans_General_stage0
# aws --endpoint-url=http://10.140.14.2:80 --profile default s3 cp ./${DATASET}/nnUNetData_plans_General_stage0 s3://nnUNet_preprocessed_${DATASET}_nnUNetData_plans_General_stage0 --recursive
fi

if [ -d ./${DATASET}/nnUNetData_plans_v2.1_stage1 ]; then
# if [ -d ./${DATASET}/nnUNetData_plans_General_stage1 ]; then
echo  uploading ${DATASET} \(stage1\)
aws --endpoint-url=http://10.140.14.2:80 --profile default s3 mb s3://nnUNet_preprocessed_${DATASET}_nnUNetData_plans_v2.1_stage1
aws --endpoint-url=http://10.140.14.2:80 --profile default s3 cp ./${DATASET}/nnUNetData_plans_v2.1_stage1 s3://nnUNet_preprocessed_${DATASET}_nnUNetData_plans_v2.1_stage1 --recursive
# aws --endpoint-url=http://10.140.14.2:80 --profile default s3 mb s3://nnUNet_preprocessed_${DATASET}_nnUNetData_plans_General_stage1
# aws --endpoint-url=http://10.140.14.2:80 --profile default s3 cp ./${DATASET}/nnUNetData_plans_General_stage1 s3://nnUNet_preprocessed_${DATASET}_nnUNetData_plans_General_stage1 --recursive
fi
done

cd -
