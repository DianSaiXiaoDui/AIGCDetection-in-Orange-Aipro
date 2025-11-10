#!/bin/bash

# 1. 设置环境变量（关键步骤！）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 创建输出目录
mkdir -p ./om_models

# 3. ATC转换模型(.onnx->.om) 
echo "Converting fused_expert.onnx..."
atc --model=./onnx_models/fused_expert.onnx \
    --framework=5 \
    --output=./om_models/fused_expert.om \
    --input_format=NCHW \
    --input_shape="input:1,3,224,224" \
    --log=debug \
    --soc_version=Ascend310B1

echo "All conversions completed!"