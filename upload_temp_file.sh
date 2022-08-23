srun -p gmai --cpus-per-task=32 ~/sensesync --listers 50 --threads=50 cp /mnt/lustre/wanghaoyu/runs/predict/FFbru4_0.6/\
 s3://JI5RY6PR1JXE32I5MMU1:ytPBTLAcla3SG7Y9oldWaE8Apnt4aTiCOFI2XWB0@test_bucket_why.10.140.14.253:80/predict/FFbru4_0.6/ &
srun -p gmai --cpus-per-task=32 ~/sensesync --listers 50 --threads=50 cp /mnt/lustre/wanghaoyu/runs/predict/FFbru4_0.8/\
 s3://JI5RY6PR1JXE32I5MMU1:ytPBTLAcla3SG7Y9oldWaE8Apnt4aTiCOFI2XWB0@test_bucket_why.10.140.14.253:80/predict/FFbru4_0.8/ &
srun -p gmai --cpus-per-task=32 ~/sensesync --listers 50 --threads=50 cp /mnt/lustre/wanghaoyu/runs/predict/FFbru4_1.4/\
 s3://JI5RY6PR1JXE32I5MMU1:ytPBTLAcla3SG7Y9oldWaE8Apnt4aTiCOFI2XWB0@test_bucket_why.10.140.14.253:80/predict/FFbru4_1.4/ &
