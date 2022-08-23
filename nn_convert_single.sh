ID=${1}
echo ${ID}
# --async --quotatype=spot \
srun -p gmai --cpus-per-task=16 \
--job-name=c${ID} \
--async --quotatype=reserved \
nnUNet_convert_decathlon_task -i /mnt/lustre/share_data/gmai/dataset/raw/labeled/Task30_CT_ORG -p 16
# python /mnt/cache/wanghaoyu/nnUNet/nnunet/dataset_conversion/Task040_KiTS.py
# nnUNet_convert_decathlon_task -i /mnt/lustre/share_data/gmai/dataset/preprocessed/labeled/${ID} -p 16