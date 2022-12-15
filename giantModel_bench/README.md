# GiantModel Totalseg Benchmark
## 项目简介

本项目用处是使用不同epoch的模型进行infer，并计算dice指标

## 操作步骤

0. 配置测试数据集和pkl文件
    - 测试数据集放到input_folder对应路径
    - splits.pkl文件放到general_split_root对应路径
1. 训练模型，确保model在ceph上，pkl在ceph或本地RESULT_FOLDERS对应目录中
2. 运行run_deepspeed

