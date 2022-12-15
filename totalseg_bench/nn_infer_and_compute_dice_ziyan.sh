#!/bin/bash
# CMD="TotalSegmentator -i /mnt/petrelfs/wanghaoyu/test/imagesTr/liver_0_0000.nii.gz -o /mnt/petrelfs/wanghaoyu/test/liver_0_0000_fast --fast"
PREFIX="/mnt/cache/wanghaoyu/SP_script/totalseg_bench"
# TASK=Task011_BTCV
# MODE=normal
# SPLIT=all
TRAINER=${1}
TASK=${2}
SPLIT=${3}
TEST=${4}
echo "info: ${TASK}_${TRAINER}_${SPLIT}"

JOB="nn"

if [ ${SPLIT} == "all" ];
then JOB+="_a"
else JOB+="_${SPLIT}"
fi
JOB+=_${TASK:4}
echo $JOB


srun -p gmai --cpus-per-task=16 \
-J $JOB \
--gres gpu:1 --ntasks 1 --ntasks-per-node 1 \
--quotatype=auto \
--async \
-o /mnt/cache/wanghaoyu/SP_script/data/ziyan_models/${TASK}_${TRAINER}_${SPLIT}${TEST}.log \
python ${PREFIX}/nnUNet_infer_and_compute_dice.py ${TRAINER} ${TASK} ${SPLIT} ${TEST}