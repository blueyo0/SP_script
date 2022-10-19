# totalseg benchmark
## 一般使用流程
### 数据准备
根据code里的路径准备好数据即可

```
input_folder = f'/mnt/petrelfs/xxx/gmai/totalseg_tmp_data/raw_data/{dataset}/imagesTr'
output_folder = f'/mnt/petrelfs/xxx/gmai/totalseg_result/{dataset}'
parameter_folder = f'/mnt/lustre/xxx/runs/nnUNet/RESULTS_FOLDER/nnUNet/3d_fullres/Task558_Totalsegmentator_dataset'
```
### nnUNet推理
1. 确认Trainer存在
2. 确认环境中可以连接aws或者本地数据
3. 确认使用正确安装nnUNet及其他依赖（deepspeed,mmcv）的环境
3. 使用exec_nn_infer.sh脚本进行推理即可

### 计算指标
使用exec_dice.sh脚本计算指标

