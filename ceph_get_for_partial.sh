for dataset in "Task020_AbdomenCT1K" "Task027_ACDC" "Task083_VerSe2020"
do
mkdir -p /mnt/lustre/share_data/gmai/nnUNet_preprocessed/$dataset/
aws --profile=default --endpoint-url=http://10.140.14.2:80 s3 cp s3://nnUNet_preprocessed/${dataset}/ /mnt/lustre/share_data/gmai/nnUNet_preprocessed/$dataset/ --recursive --exclude="*.gz"
done
