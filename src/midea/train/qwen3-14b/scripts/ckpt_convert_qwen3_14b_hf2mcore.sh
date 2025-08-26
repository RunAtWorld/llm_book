#!/bin/bash
set -e
#---导入环境变量---
source /root/miniconda3/etc/profile.d/conda.sh && conda activate py310
source ~/.bashrc
#---进入代码目录------
cd /src/train25.1.0/MindSpeed-LLM-2.1.0
#---------------------

export CUDA_DEVICE_MAX_CONNECTIONS=1

OUTPUT_BASE_DIR=/data/private/qwen3
log_path="${OUTPUT_BASE_DIR}/14b/logs/ckpt_convert_qwen3_14b_hf2mcore.log"
mkdir -p ${OUTPUT_BASE_DIR}/14b/mg_weights/qwen3_14b_mcore_tp4pp2/
mkdir -p ${OUTPUT_BASE_DIR}/14b/logs

# 设置需要的权重转换参数
python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 2 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir /data/share/platform_test/models/Qwen3-14B/ \
    --save-dir ${OUTPUT_BASE_DIR}/14b/mg_weights/qwen3_14b_mcore_tp4pp2/ \
    --tokenizer-model /data/share/platform_test/models/Qwen3-14B/tokenizer.json \
    --params-dtype bf16 \
    --model-type-hf qwen3 \
        | tee $log_path
