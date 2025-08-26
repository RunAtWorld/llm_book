#!/bin/bash
set -e
#---导入环境变量---
source /root/miniconda3/etc/profile.d/conda.sh && conda activate py310
source ~/.bashrc
#---进入代码目录------
cd /src/train25.1.0/MindSpeed-LLM-2.1.0
#---------------------

OUTPUT_BASE_DIR=/data/private/qwen3
mkdir -p ${OUTPUT_BASE_DIR}/pretrain_datasets_enwiki
log_path="${OUTPUT_BASE_DIR}/14b/logs/data_convert_qwen3_pretrain_enwiki.log"

python ./preprocess_data.py \
    --input /data/share/platform_test/datasets/enwiki/train-00000-of-00042-d964455e17e96d5a.parquet \
    --tokenizer-name-or-path /data/share/platform_test/models/Qwen3-14B/ \
    --tokenizer-type PretrainedFromHF \
    --handler-name GeneralPretrainHandler \
    --output-prefix ${OUTPUT_BASE_DIR}/pretrain_datasets_enwiki/enwiki \
    --json-keys text \
    --workers 4 \
    --log-interval 1000 \
        | tee $log_path
