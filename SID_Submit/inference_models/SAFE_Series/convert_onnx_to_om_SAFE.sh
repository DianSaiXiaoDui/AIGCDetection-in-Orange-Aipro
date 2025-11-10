#!/bin/bash

# 1. 设置环境变量（关键步骤！）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 创建输出目录
mkdir -p ./SAFE_om_models

# 3. 转换SAFE特征提取器
echo "Converting safe_body.onnx..."
atc --model=./SAFE_onnx_models/safe_body.onnx \
    --framework=5 \
    --output=./SAFE_om_models/safe_body.om \
    --input_format=NCHW \
    --input_shape="input:1,3,256,256" \
    --log=debug \
    --soc_version=Ascend310B1


echo "All conversions completed!"