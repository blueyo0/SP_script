# bash nn_pre_single.sh 354
# bash nn_pre_single.sh 300
# bash nn_pre_single.sh 301
# bash nn_pre_single.sh 302
# bash nn_pre_single.sh 303
# bash nn_pre_single.sh 304
# bash nn_pre_single.sh 305
# bash nn_pre_single.sh 306
# bash nn_pre_single.sh 307
# bash nn_pre_single.sh 308
# bash nn_pre_single.sh 309
# bash nn_pre_single.sh 310
# bash /mnt/cache/wanghaoyu/preprocess/script/nn_convert_single.sh Task11_BTCV & 
# bash /mnt/cache/wanghaoyu/preprocess/script/nn_convert_single.sh Task20_AbdomenCT1K
# bash nn_pre_single.sh 20
# bash nn_pre_single.sh 21
# bash nn_pre_single.sh 29
# bash nn_pre_single.sh 37
# bash nn_pre_single.sh 154
# bash nn_pre_single.sh 11
# bash nn_pre_single.sh 22
# bash nn_pre_single.sh 40
# bash nn_pre_single_default.sh 40
# bash nn_pre_single.sh 3
# bash nn_pre_single.sh 7
# bash nn_pre_single.sh 8
# bash nn_pre_single.sh 9
# bash nn_pre_single.sh 10
# bash nn_pre_single_default.sh 3
# bash nn_pre_single_default.sh 9
# bash nn_pre_single_default.sh 10
# bash nn_pre_single_default.sh 30
# bash nn_pre_single_default.sh 40
# bash nn_pre_single.sh 21
# bash nn_pre_single.sh 20
# bash nn_pre_single.sh 11
srun -p gmai --cpus-per-task=16 \
--job-name=raw_data \
--async --quotatype=reserved \
bash /mnt/cache/wanghaoyu/preprocess/script/ceph_upload_why_sensesync.sh