# CMD="TotalSegmentator -i /mnt/petrelfs/wanghaoyu/test/imagesTr/liver_0_0000.nii.gz -o /mnt/petrelfs/wanghaoyu/test/liver_0_0000_fast --fast"
CMD="python /mnt/cache/wanghaoyu/SP_script/totalseg/test_totalseg_infer.py ${1}"

echo "CMD: ${CMD}"
srun -p gmai --cpus-per-task=24 \
--gres gpu:1 --ntasks 1 --ntasks-per-node 1 \
--async --quotatype=auto \
${CMD}