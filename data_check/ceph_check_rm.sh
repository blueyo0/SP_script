
# for DATASET in TCGA-COAD CT_COLONOGRAPHY C4KC-KiTS TCGA-LIHC TCGA-KIRC
# PROCESS_TYPE=raw
# PROCESS_TYPE=preprocessed
# LABEL_TYPE=labeled
LABEL_TYPE=unlabeled

DATASET_BASE=/mnt/lustre/share_data/gmai/nnUNet_raw_data_base/nnUNet_raw_data

# for PROCESS_TYPE in raw 
# for PROCESS_TYPE in processed
# do
# for LABEL_TYPE in labeled unlabeled
# do
# cd ${DATASET_BASE}/${PROCESS_TYPE}/
cd ${DATASET_BASE}
dataset_list=`ls -l|awk '/^d/ {print $NF}'`
for DATASET in $dataset_list 
do
# aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 ls s3://dataset_${PROCESS_TYPE}_${LABEL_TYPE}/${DATASET}
aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 ls s3://nnUNet_raw_data/${DATASET}
if [ $? -ne 0 ]; then
    echo "failed" ${DATASET_BASE}/${DATASET}
else
    echo "succeed" ${DATASET_BASE}/${DATASET}
    rm -rf ${DATASET_BASE}/${DATASET}
fi
# done
done
cd -
# done