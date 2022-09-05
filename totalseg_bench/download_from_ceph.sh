CMD="/mnt/lustre/wanghaoyu/sensesync --listers 50 --threads=50 cp \
s3://JI5RY6PR1JXE32I5MMU1:ytPBTLAcla3SG7Y9oldWaE8Apnt4aTiCOFI2XWB0@nnUNet_raw_data.10.140.14.254:80/${1}/ \
/mnt/petrelfs/wanghaoyu/gmai/totalseg_tmp_data/raw_data/${1}/"

echo "CMD: ${CMD}"

srun -p gmai --cpus-per-task=32 \
-J dd${1:4} \
--quotatype=auto \
--async \
${CMD}