#!/bin/bash
set -e
export CUDA_DEVICE_MAX_CONNECTIONS=1 #定义了任务流能够利用或映射到的硬件队列的数量。
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True #内存碎片优化开关，默认False

# 基础配置
NPUS_PER_NODE=16  #使用单节点的8卡NPU
MASTER_ADDR=localhost #以本节点ip地址为master_ip
MASTER_PORT=6016 #本节点端口号为6015
NNODES=1  #单机，即一台节点，多机即多节点
NODE_RANK=0  #单机RANK为0，多机为(0,NNODES-1)，不同节点不可重复
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES)) #最终使用的NPU数

# 根据实际情况配置权重保存、权重加载、词表、数据集路径
CKPT_LOAD_DIR="/apps/hw/outputs/midea/qwen3/14b/mg_weights/qwen3_14b_mcore_tp4pp2/"  #权重加载路径，填入权重转换时保存的权重路径
CKPT_SAVE_DIR="/apps/hw/outputs/midea/qwen3/14b/save_weights_full/qwen3_14b_mcore_tp4pp2_intent10k_qwen3_think_13/"  #训练完成后的权重保存路径
DATA_PATH="/apps/hw/outputs/midea/qwen3/finetune_dataset_intent10k_qwen3_think/intent"  #数据集路径，填入数据预处理时保存的数据路径，注意需要添加后缀
TOKENIZER_PATH="/apps/hw/weights/Qwen3-14B/" #词表路径，填入下载的开源权重词表路径

tensorboard_dir="/apps/hw/outputs/qwen3/14b/tb/full/midea/intent10k_13"
train_log_path="logs/tune_qwen3_14b_full_intent10k_13.log"
# Evaluation params
CHECKPOINT=$CKPT_SAVE_DIR
EVAL_DATA_PATH="/apps/hw/datasets/midea_dataset/intent/intent_test_1k.jsonl"
eval_log_path="logs/generate_mcore_qwen3_14b_eval_intent10k_13.log"

TP=4
PP=2
MBS=16
GBS=128
SEQ_LENGTH=800
TRAIN_ITERS=390 #训练步数
# TRAIN_ITERS=10 #训练步数

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
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
    --train-iters ${TRAIN_ITERS} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --use-flash-attn \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --disable-bias-linear \
    --swiglu \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --norm-epsilon 1e-6 \
    --hidden-dropout 0 \
    --attention-dropout 0 \
    --no-gradient-accumulation-fusion \
    --attention-softmax-in-fp32 \
    --exit-on-missing-checkpoint \
    --no-masked-softmax-fusion \
    --group-query-attention \
    --num-query-groups 8 \
    --seed 42 \
    --bf16 \
    --lr-warmup-fraction 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-load-optim \
    --no-load-rng \
    --lr 1.0e-5 \
    --sequence-parallel
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --log-throughput
"

TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --prompt-type qwen \
    --variable-seq-lengths
"
echo "==========Start train=========="
torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $TUNE_ARGS \
    --tensorboard-dir $tensorboard_dir \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee $train_log_path

echo "==========Start evaluation=========="
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
         --eval-data-path $EVAL_DATA_PATH \
         --temperature 0.00001 \
         --eval-data-size 1000 \
                        | tee $eval_log_path
