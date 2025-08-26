#!/bin/bash
set -e
#---导入环境变量---
source /root/miniconda3/etc/profile.d/conda.sh && conda activate py310
source ~/.bashrc
#---进入代码目录------
cd /src/train25.1.0/MindSpeed-LLM-2.1.0
#---------------------

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# 基础配置
NPUS_PER_NODE=16  #A3单节点16卡NPU
MASTER_ADDR=localhost #以本节点ip地址为master_ip
MASTER_PORT=6115 #本节点端口号为6015
NNODES=1  #单机，即一台节点，多机即多节点
NODE_RANK=0  #单机RANK为0，多机为(0,NNODES-1)，不同节点不可重复
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES)) #最终使用的NPU数

OUTPUT_BASE_DIR=/data/private/qwen3

# 根据实际情况配置权重保存、权重加载、词表、数据集路径
CKPT_LOAD_DIR="${OUTPUT_BASE_DIR}/14b/mg_weights/qwen3_14b_mcore_tp4pp2/"  # 权重加载路径，填入权重转换时保存的权重路径
CKPT_SAVE_DIR="${OUTPUT_BASE_DIR}/14b/save_weights_pretrain/qwen3_14b_mcore_tp4pp2/"                # 训练完成后的权重保存路径
DATA_PATH="${OUTPUT_BASE_DIR}/pretrain_datasets_enwiki/enwiki_text_document"      # 数据集路径，填入数据预处理时保存的数据路径，注意需要添加后缀
TOKENIZER_PATH="/data/share/platform_test/models/Qwen3-14B/" # 词表路径，填入下载的开源权重词表路径
log_path="${OUTPUT_BASE_DIR}/14b/logs/pretrain_qwen3_14b_enwiki_4K_ptd.log"

TP=4  
PP=2 
CP=1  #序列并行
MBS=4 #设置micro-batch-size为4
GBS=64 #设置global-batch-size为128
SEQ_LENGTH=4096  #设置seq_length为4096 
TRAIN_ITERS=1000000 
CP_TYPE='ulysses_cp_algo' #序列并行算法

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
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 1.25e-6 \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_TYPE} \
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --num-layers 40 \
    --hidden-size 5120 \
    --untie-embeddings-and-output-weights \
    --num-attention-heads 40 \
    --ffn-hidden-size 17408 \
    --max-position-embeddings 40960 \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --disable-bias-linear \
    --swiglu \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --norm-epsilon 1e-6 \
    --no-gradient-accumulation-fusion \
    --attention-softmax-in-fp32 \
    --exit-on-missing-checkpoint \
    --group-query-attention \
    --num-query-groups 8 \
    --no-load-optim \
    --no-load-rng \
    --tensorboard-dir /data/share/platform_test/src/llm-qwen3/MindSpeed-LLM/tb \
    --sequence-parallel
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 20000 \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --log-throughput
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    --distributed-backend nccl \
    --tensorboard-dir ${OUTPUT_BASE_DIR}/14b/tb/pretrain/enwiki \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee $log_path
