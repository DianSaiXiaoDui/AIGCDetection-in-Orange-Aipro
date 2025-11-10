#!/bin/bash

# 1. 设置环境变量（关键步骤！）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 创建输出目录
mkdir -p ./SR_om_models

# 3. 转换SAFE特征提取器
echo "Converting safe_feature.onnx..."
atc --model=./SR_onnx_models/safe_feature.onnx \
    --framework=5 \
    --output=./om_models/safe_feature \
    --input_format=NCHW \
    --input_shape="input:1,3,256,256" \
    --log=debug \
    --soc_version=Ascend310B1

# 4. 转换CLIP特征提取器
echo "Converting clip_feature.onnx..."
atc --model=./SR_onnx_models/clip_feature.onnx \
    --framework=5 \
    --output=./om_models/clip_feature \
    --input_format=NCHW \
    --input_shape="input:1,3,256,256" \
    --log=debug \
    --soc_version=Ascend310B1

# 5. 转换分类头
echo "Converting classifier.onnx..."
atc --model=./SR_onnx_models/classifier.onnx \
    --framework=5 \
    --output=./om_models/classifier \
    --input_format=ND \
    --input_shape="safe_feature:1,512;clip_feature:1,1024" \
    --log=debug \
    --soc_version=Ascend310B1

echo "All conversions completed!"