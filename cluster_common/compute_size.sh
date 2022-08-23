curr_dir=${1}
cd ${curr_dir}

dataset_list=`ls -l|awk '/^d/ {print $NF}'`
for DATASET in $dataset_list 
do

if [ -d ./${DATASET} ]; then
echo compute ${DATASET}
# ~/sensesync --listers 50 --threads=50 cp ./${DATASET}/ s3://JI5RY6PR1JXE32I5MMU1:ytPBTLAcla3SG7Y9oldWaE8Apnt4aTiCOFI2XWB0@nnUNet_raw_data.10.140.14.254:80/${DATASET}/
fi
done

cd -
