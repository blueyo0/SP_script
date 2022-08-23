for i in {1..50}
do
# echo $i
aws --profile=why --endpoint-url=http://10.140.14.254:80 s3 rm s3://nnUNet_cropped_dataset/ --recursive
done
