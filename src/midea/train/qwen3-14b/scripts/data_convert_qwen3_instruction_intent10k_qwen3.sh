#!/bin/bash
set -e
#---导入环境变量---
source /root/miniconda3/etc/profile.d/conda.sh && conda activate py310
source ~/.bashrc
#---进入代码目录------
cd /src/train25.1.0/MindSpeed-LLM-master
#---------------------

OUTPUT_BASE_DIR=/data/private/qwen3
mkdir -p ${OUTPUT_BASE_DIR}/finetune_dataset_intent10k_qwen3_think
log_path="${OUTPUT_BASE_DIR}/14b/logs/data_convert_qwen3_instruction_intent10k_qwen3.log"
mkdir -p ${OUTPUT_BASE_DIR}/14b/logs

python ./preprocess_data.py \
    --input  /data/share/platform_test/datasets/intent_train_10k.jsonl \
    --tokenizer-name-or-path /data/share/platform_test/models/Qwen3-14B/ \
    --output-prefix ${OUTPUT_BASE_DIR}/finetune_dataset_intent10k_qwen3_think/intent \
    --handler-name SharegptStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --enable-thinking true \
    --workers 4 \
    --log-interval 1000 \
    --map-keys '{"messages":"messages", "tags":{"role_tag": "role","content_tag": "content","user_tag": "user","assistant_tag": "assistant","system_tag": "system"}}' \
    --prompt-type qwen3 \
        | tee $log_path
