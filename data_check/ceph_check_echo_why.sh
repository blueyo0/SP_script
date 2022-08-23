
cd /mnt/lustre/share_data/gmai/nnUNet_raw_data_base/nnUNet_raw_data

# plan_type=nnUNetData_plans_v2.1_stage1
plan_type=nnUNetData_plans_General_stage1

dataset_list=`ls -l|awk '/^d/ {print $NF}'`
for RAW_DATASET in $dataset_list 
do
DATASET=${RAW_DATASET:8:100}
# echo $DATASET
check_result=`aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 ls s3://dataset_raw_unlabeled/$DATASET`
if [ $? -ne 0 ]; then
    echo "not found on ceph"
else
    echo s3://dataset_raw_unlabeled/$DATASET 
fi
done

cd -