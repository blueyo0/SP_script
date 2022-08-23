# bash script/compute_size.sh /mnt/lustre/share_data/gmai/nnUNet_preprocessed > data/quota/nnUNet_preprocessed.log
# bash script/compute_size.sh /mnt/lustre/share_data/gmai/nnUNet_raw_data_base/nnUNet_raw_data/ > data/quota/nnUNet_raw_data.log
# bash script/compute_size.sh /mnt/lustre/share_data/gmai/nnUNet_raw_data_base/nnUNet_raw_data/ > data/quota/nnUNet_raw_data.log

# srun -p gmai --cpus-per-task=32 python /mnt/cache/wanghaoyu/preprocess/script/nnUNet_ckpt_upload.py>ckpt.log

srun -p gmai --cpus-per-task=16 ~/sensesync_v2 --listers 50 --threads=50 sync -o\
 ~/test_text/\
 s3://JI5RY6PR1JXE32I5MMU1:ytPBTLAcla3SG7Y9oldWaE8Apnt4aTiCOFI2XWB0@test_bucket_why.10.140.14.253:80/test_text/

srun -p gmai --cpus-per-task=32 ~/sensesync_v2 --listers 50 --threads=50 sync -o\
 /mnt/lustre/wanghaoyu/runs/nnUNet/RESULTS_FOLDER/\
 s3://73R35CPPQQR9153SHSTD:QFJPSptY38e5T2Y90vwrvE6Iel3xWWmgPLVVzle1@nnUNet_trained_models.10.140.14.254:80/

srun -p gmai --cpus-per-task=32 ~/sensesync_v2 --listers 50 --threads=50 sync -o\
 s3://73R35CPPQQR9153SHSTD:QFJPSptY38e5T2Y90vwrvE6Iel3xWWmgPLVVzle1@models_wanghaoyu.10.140.14.254:80/\
 s3://73R35CPPQQR9153SHSTD:QFJPSptY38e5T2Y90vwrvE6Iel3xWWmgPLVVzle1@nnUNet_trained_models.10.140.14.254:80/


srun -p gmai --cpus-per-task=32 ~/sensesync --listers 50 --threads=50 --dryrun --include="model_final_checkpoint.model" cp \
 s3://73R35CPPQQR9153SHSTD:QFJPSptY38e5T2Y90vwrvE6Iel3xWWmgPLVVzle1@nnUNet_trained_models.10.140.14.254:80/nnUNet/ \
 s3://JI5RY6PR1JXE32I5MMU1:ytPBTLAcla3SG7Y9oldWaE8Apnt4aTiCOFI2XWB0@test_bucket_why.10.140.14.253:80/nnUNet/ > /mnt/cache/wanghaoyu/preprocess/data/ckpt_data/ceph_ckpt.log