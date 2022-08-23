cd /mnt/lustre/wanghaoyu/runs/nnUNet/RESULTS_FOLDER/nnUNet/3d_fullres
for DATASET in Task003_Liver  Task007_Pancreas Task009_Spleen  Task011_BTCV  Task022_FLARE2022  Task037_CHAOS_Task_3_5_Variant1  Task154_RibFrac Task008_HepaticVessel  Task010_Colon   Task020_AbdomenCT1K  Task029_LITS
do
    echo upload ${DATASET}
    aws --endpoint-url=http://10.140.14.2:80 --profile default s3 cp ./${DATASET} s3://models_wanghaoyu/${DATASET} --recursive
done
cd -

