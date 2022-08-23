# srun -p gmai --cpus-per-task=32 ~/sensesync_v2 --listers 50 --threads=50 sync -o\
#  s3://QZ4FUCEX2J7M9P4TFS5L:heFoPPVg6n43VqKDjIC7crJwj3OotnjdfJPPpQZI@save_29.10.140.2.204:80/nnUNet/RESULTS_FOLDER/\
#  s3://73R35CPPQQR9153SHSTD:QFJPSptY38e5T2Y90vwrvE6Iel3xWWmgPLVVzle1@nnUNet_trained_models.10.140.14.254:80/
# srun -p gmai --cpus-per-task=32 ~/sensesync_v2 --listers 50 --threads=50 sync -o\
#  s3://QZ4FUCEX2J7M9P4TFS5L:heFoPPVg6n43VqKDjIC7crJwj3OotnjdfJPPpQZI@save_33.10.140.2.204:80/nnUNet/RESULTS_FOLDER/\
#  s3://73R35CPPQQR9153SHSTD:QFJPSptY38e5T2Y90vwrvE6Iel3xWWmgPLVVzle1@nnUNet_trained_models.10.140.14.254:80/
# srun -p gmai --cpus-per-task=32 ~/sensesync_v2 --listers 50 --threads=50 sync -o\
#  s3://QZ4FUCEX2J7M9P4TFS5L:heFoPPVg6n43VqKDjIC7crJwj3OotnjdfJPPpQZI@save_33.10.140.2.204:80/nvme/wanghaoyu/nnUNet_root/RESULTS_FOLDER/\
#  s3://73R35CPPQQR9153SHSTD:QFJPSptY38e5T2Y90vwrvE6Iel3xWWmgPLVVzle1@nnUNet_trained_models.10.140.14.254:80/
# srun -p gmai --cpus-per-task=32 ~/sensesync_v2 --listers 50 --threads=50 sync -o\
#  s3://QZ4FUCEX2J7M9P4TFS5L:heFoPPVg6n43VqKDjIC7crJwj3OotnjdfJPPpQZI@save_33.10.140.2.204:80/home/hejunjun/nnUNet/RESULTS_FOLDER/\
#  s3://73R35CPPQQR9153SHSTD:QFJPSptY38e5T2Y90vwrvE6Iel3xWWmgPLVVzle1@nnUNet_trained_models.10.140.14.254:80/
# srun -p gmai --cpus-per-task=32 ~/sensesync_v2 --listers 50 --threads=50 sync -o\
#  s3://QZ4FUCEX2J7M9P4TFS5L:heFoPPVg6n43VqKDjIC7crJwj3OotnjdfJPPpQZI@save_27.10.140.2.204:80/home/hejunjun/nnUNet/RESULTS_FOLDER/\
#  s3://73R35CPPQQR9153SHSTD:QFJPSptY38e5T2Y90vwrvE6Iel3xWWmgPLVVzle1@nnUNet_trained_models.10.140.14.254:80/
for d in Task001_BrainTumour  Task002_Heart  Task003_Liver  Task004_Hippocampus  Task005_Prostate  Task006_Lung  Task007_Pancreas  Task008_HepaticVessel  Task009_Spleen  Task010_Colon
do
srun -p gmai --cpus-per-task=32 ~/sensesync --listers 50 --threads=50 cp\
 /mnt/lustre/wanghaoyu/runs/nnUNet/bench_ckpts/$d/\
 s3://73R35CPPQQR9153SHSTD:QFJPSptY38e5T2Y90vwrvE6Iel3xWWmgPLVVzle1@nnUNet_trained_models.10.140.14.254:80/nnUNet/3d_fullres/$d/
done

srun -p gmai --cpus-per-task=32 ~/sensesync --listers 50 --threads=50 cp\
 /mnt/lustre/wanghaoyu/runs/nnUNet/bench_ckpts/\
 s3://73R35CPPQQR9153SHSTD:QFJPSptY38e5T2Y90vwrvE6Iel3xWWmgPLVVzle1@nnUNet_trained_models.10.140.14.254:80/nnUNet/3d_fullres/

