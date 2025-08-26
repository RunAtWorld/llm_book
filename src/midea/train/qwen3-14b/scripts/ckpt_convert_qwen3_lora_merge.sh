#!/bin/bash
set -e
#---导入环境变量---
source /root/miniconda3/etc/profile.d/conda.sh && conda activate py310
source ~/.bashrc
#---进入代码目录------
cd /src/train25.1.0/MindSpeed-LLM-2.1.0
#---------------------

OUTPUT_BASE_DIR=/data/private/qwen3
log_path="${OUTPUT_BASE_DIR}/14b/logs/ckpt_convert_qwen3_lora_merge.log"
mkdir -p ${OUTPUT_BASE_DIR}/14b/logs

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type mg \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 2 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --load-dir ${OUTPUT_BASE_DIR}/14b/mg_weights/qwen3_14b_mcore_tp4pp2/ \
    --lora-load ${OUTPUT_BASE_DIR}/14b/save_weights_lora/qwen3_14b_mcore_tp4pp2_intent10k_qwen3_think/ \
    --save-dir ${OUTPUT_BASE_DIR}/14b/save_weights_lora/qwen3_14b_mcore_tp4pp2_intent10k_qwen3_think_merge/ \
    --model-type-hf qwen3 \
        | tee $log_path
