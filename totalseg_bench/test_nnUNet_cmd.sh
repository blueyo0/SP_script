# CMD="TotalSegmentator -i /mnt/petrelfs/wanghaoyu/test/imagesTr/liver_0_0000.nii.gz -o /mnt/petrelfs/wanghaoyu/test/liver_0_0000_fast --fast"
CMD="python /mnt/cache/wanghaoyu/SP_script/totalseg_bench/test_nnUNet_infer.py ${1} ${2}"

echo "CMD: ${CMD}"

JOB=${1:0-3:3}_${2:4}
echo $JOB
srun -p gmai --cpus-per-task=24 \
-J $JOB \
--gres gpu:1 --ntasks 1 --ntasks-per-node 1 \
--async --quotatype=auto \
${CMD}