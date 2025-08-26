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
CHECKPOINT="${OUTPUT_BASE_DIR}/14b/save_weights_full/qwen3_14b_mcore_tp4pp2_intent10k_qwen3_think/" # 指向微调后权重的保存路径
TOKENIZER_PATH="/data/share/platform_test/models/Qwen3-14B/" # 指向模型tokenizer的路径
eval_log_path="${OUTPUT_BASE_DIR}/14b/logs/eval_qwen3_14b_full_intent10k.log"
EVAL_DATA_PATH="/data/share/platform_test/datasets/intent_test_1k.jsonl"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1 # 集群里的节点数，以实际情况填写,
NODE_RANK=0  # 当前节点的RANK，多个节点不能重复，主节点为0, 其他节点可以是1,2..
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=4
PP=2
SEQ_LENGTH=4096

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS inference_4_midea_test.py \
         --use-mcore-models \
         --tensor-model-parallel-size ${TP} \
         --pipeline-model-parallel-size ${PP} \
         --load ${CHECKPOINT} \
         --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
         --kv-channels 128 \
         --qk-layernorm \
         --num-layers 40 \
         --hidden-size 5120 \
         --use-rotary-position-embeddings \
         --untie-embeddings-and-output-weights \
         --num-attention-heads 40 \
         --ffn-hidden-size 17408 \
         --max-position-embeddings 40960 \
         --seq-length ${SEQ_LENGTH} \
         --make-vocab-size-divisible-by 1 \
         --padded-vocab-size 151936 \
         --rotary-base 1000000 \
         --micro-batch-size 1 \
         --disable-bias-linear \
         --swiglu \
         --use-rotary-position-embeddings \
         --tokenizer-type PretrainedFromHF \
         --tokenizer-name-or-path ${TOKENIZER_PATH} \
         --normalization RMSNorm \
         --position-embedding-type rope \
         --norm-epsilon 1e-6 \
         --hidden-dropout 0 \
         --attention-dropout 0 \
         --max-new-tokens 256 \
         --no-gradient-accumulation-fusion \
         --attention-softmax-in-fp32 \
         --exit-on-missing-checkpoint \
         --no-masked-softmax-fusion \
         --group-query-attention \
         --num-query-groups 8 \
         --seed 42 \
         --bf16 \
      	 --eval-data-path ${EVAL_DATA_PATH} \
       	 --eval-data-size 1000 \
	     --temperature 0.00001 \
       		| tee $eval_log_path
