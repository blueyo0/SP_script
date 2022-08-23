ID=${1}

# --async --quotatype=spot \
srun -p gmai --cpus-per-task=16 \
--job-name=pre${ID} \
--async --quotatype=reserved \
nnUNet_plan_and_preprocess \
-t ${ID} -pl3d ExperimentPlanner3D_General -pl2d None