CMD=${1}

echo "CMD: ${CMD}"
srun -p gmai --cpus-per-task=32 \
--gres gpu:1 --ntasks 1 --ntasks-per-node 1\
--job-name=pre${ID} \
--quotatype=auto \
${CMD}