ID=${1}

# --async --quotatype=spot \
srun -p gmai --cpus-per-task=16 \
--job-name=pre${ID} \
--async --quotatype=reserved \
nnUNet_plan_and_preprocess \
-t ${ID} -pl2d None