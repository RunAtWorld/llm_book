#!/bin/bash
set -e
export CUDA_DEVICE_MAX_CONNECTIONS=1  #定义了任务流能够利用或映射到的硬件队列的数量。

# 基础配置
NPUS_PER_NODE=8  #使用单节点的8卡NPU
MASTER_ADDR=localhost #以本节点ip地址为master_ip
MASTER_PORT=6015 #本节点端口号为6014
NNODES=1  #单机，即一台节点，多机即多节点
NODE_RANK=0  #单机RANK为0，多机为(0,NNODES-1)，不同节点不可重复
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES)) #最终使用的NPU数

# 根据实际情况配置权重保存、权重加载、词表、数据集路径
CKPT_LOAD_DIR="/apps/hw/outputs/midea/qwen3/14b/mg_weights/qwen3_14b_mcore_tp4pp2/"
CKPT_SAVE_DIR="/apps/hw/outputs/midea/qwen3/14b/save_weights_lora/qwen3_14b_mcore_tp4pp2_intent10k_qwen3_think_14/"
DATA_PATH="/apps/hw/outputs/midea/qwen3/finetune_dataset_intent10k_qwen3_think/intent"
TOKENIZER_PATH="/apps/hw/weights/Qwen3-14B/"
train_log_path="logs/tune_qwen3_14b_lora_intent10k_ptd_14.log"
tensorboard_dir="/apps/hw/outputs/qwen3/14b/tb/lora/midea/intent10k_14"
# Evaluation params
CHECKPOINT_LORA=$CKPT_SAVE_DIR
EVAL_DATA_PATH="/apps/hw/datasets/midea_dataset/intent/intent_test_1k.jsonl"
eval_log_path="logs/ganerate_qwen3_14b_lora_intent10k_ptd_14.log"

TP=4
PP=2
SEQ_LENGTH=2048
TRAIN_ITERS=790

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer
"
TRAIN_ARGS="
    --micro-batch-size 4 \
    --global-batch-size 64 \
    --lr 1.0e-5 \
    --lr-warmup-fraction 0.1 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
    --no-shared-storage
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP}
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 40 \
    --hidden-size 5120 \
    --ffn-hidden-size 17408 \
    --num-attention-heads 40 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 8
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng
"

TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --tokenizer-not-use-fast \
    --prompt-type qwen \
    --variable-seq-lengths \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-fusion \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
"
echo "==========Start train=========="
torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $TUNE_ARGS \
    $MODEL_PARALLEL_ARGS \
    --tensorboard-dir $tensorboard_dir \
    --distributed-backend nccl \
    | tee  $train_log_path
	
echo "==========Start evaluation=========="
LORA_ARGS="
    --lora-load ${CHECKPOINT_LORA} \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-fusion \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
"

torchrun $DISTRIBUTED_ARGS inference_4_midea_test.py \
         --use-mcore-models \
         --tensor-model-parallel-size ${TP} \
         --pipeline-model-parallel-size ${PP} \
         --load ${CKPT_LOAD_DIR} \
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
         --eval-data-path $EVAL_DATA_PATH \
         --eval-data-size 1000 \
         ${LORA_ARGS} \
         --temperature 0.00001 \
         | tee $eval_log_path
