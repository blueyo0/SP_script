CMD=${1}

echo "python ${CMD}"
srun -p gmai --cpus-per-task=16 \
--job-name=pre${ID} \
--async --quotatype=reserved \
python ${CMD}