RESULT_DIR="/mnt/lustre/wanghaoyu/runs/nnUNet/RESULTS_FOLDER/nnUNet/3d_fullres"
for dataset in "Task001_BrainTumour" "Task002_Heart" "Task003_Liver" "Task004_Hippocampus" "Task005_Prostate" "Task006_Lung" "Task007_Pancreas" "Task008_HepaticVessel" "Task009_Spleen" "Task010_Colon"
do
echo $dataset
# ~/sensesync --listers 50 --threads=50 --include="pkl$" cp \
# s3://73R35CPPQQR9153SHSTD:QFJPSptY38e5T2Y90vwrvE6Iel3xWWmgPLVVzle1@nnUNet_trained_models.10.140.14.254:80/nnUNet/3d_fullres/$dataset/BigResUNetTrainerV4_222222_64__nnUNetPlansv2.1/ \
# $RESULT_DIR/$dataset/BigResUNetTrainerV4_222222_64__nnUNetPlansv2.1/ 
# ~/sensesync --listers 50 --threads=50 --include="model$" cp \
# s3://73R35CPPQQR9153SHSTD:QFJPSptY38e5T2Y90vwrvE6Iel3xWWmgPLVVzle1@nnUNet_trained_models.10.140.14.254:80/nnUNet/3d_fullres/$dataset/BigResUNetTrainerV4_222222_64__nnUNetPlansv2.1/ \
# $RESULT_DIR/$dataset/BigResUNetTrainerV4_222222_64__nnUNetPlansv2.1/ 
# aws --profile=default --endpoint-url=http://10.140.14.254:80 s3 ls nnUNet_trained_models/nnUNet/3d_fullres/$dataset/BigResUNetTrainerV4_222222_64__nnUNetPlansv2.1/fold_0/
# aws --profile=default --endpoint-url=http://10.140.14.254:80 s3 ls nnUNet_trained_models/nnUNet/3d_fullres/$dataset/BigResUNetTrainerV4_222222_64__nnUNetPlansv2.1/fold_1/
# aws --profile=default --endpoint-url=http://10.140.14.254:80 s3 ls nnUNet_trained_models/nnUNet/3d_fullres/$dataset/BigResUNetTrainerV4_222222_64__nnUNetPlansv2.1/fold_2/
# aws --profile=default --endpoint-url=http://10.140.14.254:80 s3 ls nnUNet_trained_models/nnUNet/3d_fullres/$dataset/BigResUNetTrainerV4_222222_64__nnUNetPlansv2.1/fold_3/
# aws --profile=default --endpoint-url=http://10.140.14.254:80 s3 ls nnUNet_trained_models/nnUNet/3d_fullres/$dataset/BigResUNetTrainerV4_222222_64__nnUNetPlansv2.1/fold_4/
ls $RESULT_DIR/$dataset/
done
