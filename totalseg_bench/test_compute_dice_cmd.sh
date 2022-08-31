#!/bin/bash
# CMD="TotalSegmentator -i /mnt/petrelfs/wanghaoyu/test/imagesTr/liver_0_0000.nii.gz -o /mnt/petrelfs/wanghaoyu/test/liver_0_0000_fast --fast"

# TASK=Task011_BTCV
# MODE=normal
# SPLIT=all
TASK=${1}
MODE=${2}
SPLIT=${3}
TEST=${4}
echo "info: ${TASK}_${MODE}_${SPLIT}"

JOB=""
if [ ${MODE} == "normal" ];
then JOB="n"
else JOB="f"
fi
if [ ${SPLIT} == "all" ];
then JOB+="_a"
else JOB+="_${SPLIT}"
fi
JOB+=_${TASK:4}
# echo $JOB


srun -p gmai --cpus-per-task=32 \
-J $JOB \
--quotatype=auto \
-o /mnt/cache/wanghaoyu/SP_script/data/totalseg/${TASK}_${MODE}_${SPLIT}${TEST}.log \
--async \
python /mnt/cache/wanghaoyu/SP_script/totalseg/compute_metrics.py ${TASK} ${MODE} ${SPLIT} ${TEST}