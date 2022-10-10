DATA_DIR="/mnt/petrelfs/wanghaoyu/why/temp/MRBrainS13DataNii"
cd $DATA_DIR
mkdir -p $DATA_DIR/imagesTr
mkdir -p $DATA_DIR/labelsTr
mkdir -p $DATA_DIR/imagesTs
for i in 1 2 3 4 5
do   
cp $DATA_DIR/TrainingData/$i/LabelsForTraining.nii    $DATA_DIR/labelsTr/MRBrain13_$i.nii
cp $DATA_DIR/TrainingData/$i/T1_1mm.nii               $DATA_DIR/imagesTr/MRBrain13_${i}_0000.nii
cp $DATA_DIR/TrainingData/$i/T1_IR.nii                $DATA_DIR/imagesTr/MRBrain13_${i}_0001.nii
cp $DATA_DIR/TrainingData/$i/T1.nii                   $DATA_DIR/imagesTr/MRBrain13_${i}_0002.nii
cp $DATA_DIR/TrainingData/$i/T2_FLAIR.nii             $DATA_DIR/imagesTr/MRBrain13_${i}_0003.nii
gzip -c $DATA_DIR/labelsTr/MRBrain13_$i.nii > $DATA_DIR/labelsTr/MRBrain13_$i.nii.gz
gzip -c $DATA_DIR/imagesTr/MRBrain13_${i}_0000.nii > $DATA_DIR/imagesTr/MRBrain13_${i}_0000.nii.gz
gzip -c $DATA_DIR/imagesTr/MRBrain13_${i}_0001.nii > $DATA_DIR/imagesTr/MRBrain13_${i}_0001.nii.gz
gzip -c $DATA_DIR/imagesTr/MRBrain13_${i}_0002.nii > $DATA_DIR/imagesTr/MRBrain13_${i}_0002.nii.gz
gzip -c $DATA_DIR/imagesTr/MRBrain13_${i}_0003.nii > $DATA_DIR/imagesTr/MRBrain13_${i}_0003.nii.gz
done

for i in 1  10  11  12  13  14  15  2  3  4  5  6  7  8  9
do   
cp $DATA_DIR/TestData/$i/T1_1mm.nii               $DATA_DIR/imagesTs/MRBrain13Ts_${i}_0000.nii
cp $DATA_DIR/TestData/$i/T1_IR.nii                $DATA_DIR/imagesTs/MRBrain13Ts_${i}_0001.nii
cp $DATA_DIR/TestData/$i/T1.nii                   $DATA_DIR/imagesTs/MRBrain13Ts_${i}_0002.nii
cp $DATA_DIR/TestData/$i/T2_FLAIR.nii             $DATA_DIR/imagesTs/MRBrain13Ts_${i}_0003.nii
gzip -c $DATA_DIR/imagesTs/MRBrain13Ts_${i}_0000.nii > $DATA_DIR/imagesTs/MRBrain13Ts_${i}_0000.nii.gz
gzip -c $DATA_DIR/imagesTs/MRBrain13Ts_${i}_0001.nii > $DATA_DIR/imagesTs/MRBrain13Ts_${i}_0001.nii.gz
gzip -c $DATA_DIR/imagesTs/MRBrain13Ts_${i}_0002.nii > $DATA_DIR/imagesTs/MRBrain13Ts_${i}_0002.nii.gz
gzip -c $DATA_DIR/imagesTs/MRBrain13Ts_${i}_0003.nii > $DATA_DIR/imagesTs/MRBrain13Ts_${i}_0003.nii.gz
done

mkdir -p $DATA_DIR/Task161_MRBrainS13DataNii
mv imagesTr $DATA_DIR/Task161_MRBrainS13DataNii
mv labelsTr $DATA_DIR/Task161_MRBrainS13DataNii
mv imagesTs $DATA_DIR/Task161_MRBrainS13DataNii
python /mnt/cache/wanghaoyu/SP_script/dataset_label_check/generate_json.py