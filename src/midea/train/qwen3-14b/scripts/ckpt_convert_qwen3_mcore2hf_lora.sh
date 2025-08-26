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
log_path="${OUTPUT_BASE_DIR}/14b/logs/ckpt_convert_qwen3_mcore2hf_lora.log"
mkdir -p ${OUTPUT_BASE_DIR}/14b/logs

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir  ${OUTPUT_BASE_DIR}/14b/save_weights_lora/qwen3_14b_mcore_tp4pp2_intent10k_qwen3_think_merge/ \
    --save-dir /data/share/platform_test/models/Qwen3-14B_lora/ \
    --model-type-hf qwen3 \
        | tee $log_path

cp -rf /data/share/platform_test/models/Qwen3-14B_lora/mg2hf ${OUTPUT_BASE_DIR}/14b/save_weights_lora/