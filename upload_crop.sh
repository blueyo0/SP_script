cd /mnt/lustre/share_data/gmai/nnUNet_raw_data_base/nnUNet_cropped_data

for DATASET in Task099_AbdomenCT5K Task300_AbdomenSubset0 Task301_AbdomenSubset1 Task302_AbdomenSubset2 Task303_AbdomenSubset3 Task304_AbdomenSubset4 Task305_AbdomenSubset5 Task306_AbdomenSubset6 Task307_AbdomenSubset7 Task308_AbdomenSubset8 Task309_AbdomenSubset9 Task310_AbdomenSubset10
do
~/sensesync --listers 50 --threads=50  \
cp ./${DATASET}/ s3://JI5RY6PR1JXE32I5MMU1:ytPBTLAcla3SG7Y9oldWaE8Apnt4aTiCOFI2XWB0@nnUNet_cropped_dataset.10.140.2.254:80/${DATASET}/ 
done

cd -