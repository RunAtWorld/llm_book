# Qwen3-14B基于业务数据集容器内验证

> GPU基线：
>
> qwen3_14B_intent_10k_5e, 8 * 80G A800 Accuracy: 0.942
> qwen3_14B_lora_intent_10k_5e, 4 * 48G L40 Accuracy: 0.887
>
> SFT模型权重（含 loss ）、训练参数、评估脚本跟 Qwen2.5-7B 训练好的模型在同一路径下/apps/likuan/llm_experiment/sft/

## 一、版本配套

相关组件配套表如下：

| 软件          | 版本              | 部署方式 |
| ------------- | ----------------- | -------- |
| Driver        | AscendHDK 25.2.0  | 宿主机   |
| Firmware      | AscendHDK 25.2.0  | 宿主机   |
| Python        | 3.10              | 容器内   |
| CANN          | CANN 8.2.0        | 容器内   |
| Torch         | 2.6.0             | 容器内   |
| Torch_npu     | release v7.1.0    | 容器内   |
| MindSpeed     | 2.1.0_core_r0.8.0 | 容器内   |
| MindSpeed-LLM | 2.1.0             | 容器内   |
| Megatron-LM   | core_r0.8.0       | 容器内   |
| Docker镜像OS  | Ubuntu 20.04.6    | /        |

## 二、创建容器

```bash
docker run -it -u root --privileged --ipc=host \
    --name 'qwen3_sft' \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    --device=/dev/davinci8 \
    --device=/dev/davinci9 \
    --device=/dev/davinci10 \
    --device=/dev/davinci11 \
    --device=/dev/davinci12 \
    --device=/dev/davinci13 \
    --device=/dev/davinci14 \
    --device=/dev/davinci15 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -p 50000:3000 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/sbin/:/usr/local/sbin/ \
    -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
    -v /apps/shared_storage/hw_test/hw:/apps/hw \
    image.midea.com/caas/a3_llm_release:v2.1.2  bash
```

> * -p 50000:3000 暴露tensorboard端口，50000为宿主机端口号，3000为容器端口号
> * 宿主机挂载的代码目录为：/apps/shared_storage/hw_test/hw/src/hw-qwen3/MindSpeed-LLM，容器内目录为 /apps/hw/src/hw-qwen3/MindSpeed-LLM
> * 进入容器时会生效 .bashrc 中的环境变量，所以需要等待几秒。环境变量相关代码如下
>
> ```
> PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe:/home/CI_daily/Train_PT/MindSpeed-MM/decord/python:/home/CI_daily/Train_PT/MindSpeed-MM/decord/python:/home/caiyy/decord/python
> export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH}
> conda activate py310
> source /usr/local/Ascend/ascend-toolkit/set_env.sh
> # 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
> export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
> ```

## 三、HF权重转为MG格式

昇腾MindSpeed-LLM要求模型权重采用Megatron-LM格式，在这里我们将原始HuggingFace权重格式转换为Megatron-Mcore格式。

使用转换脚本，获取对应切分的mg权重。

```bash
cd /apps/hw/src/hw-qwen3/MindSpeed-LLM
mkdir midea_qwen3
cp examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh midea_qwen3/ckpt_convert_qwen3_14b_hf2mcore.sh
vim midea_qwen3/ckpt_convert_qwen3_14b_hf2mcore.sh
```

这里采用**TP2PP4**的切分策略，按照如下配置修改权重转换脚本

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1 #定义了任务流能够利用或映射到的硬件队列的数量。

# 设置需要的权重转换参数
python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 4 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir /apps/hw/weights/Qwen3-14B/ \
    --save-dir /apps/hw/outputs/midea/qwen3/14b/mg_weights/qwen3_14b_mcore_tp2pp4/ \
    --tokenizer-model /apps/hw/weights/Qwen3-14B/tokenizer.json \
    --params-dtype bf16 \
    --model-type-hf qwen3
```
启动转换

```bash
bash midea_qwen3/ckpt_convert_qwen3_14b_hf2mcore.sh
```

转换过程部分日志如下
![image-20250802232634678](qwen3-14B业务数据集容器内验证.assets/image-20250802232634678.png)

参数解析

| 参数                                  | 说明                                                         | 必填 |
|-------------------------------------|------------------------------------------------------------|---|
| `--model-type GPT`                  | 指定模型类型为GPT系列                                               | ✅ |
| `--use-mcore-models`                | 转换为Megatron-Mcore格式                        | ✅ |
| `--target-tensor-parallel-size`   | 张量并行度设置                                             | ✅ |
| `--target-pipeline-parallel-size` | 流水线并行度设置                                            | ✅ |
| `--tokenizer-model`                 | 指定分词器路径                                                    | ✅ |
| `--load-model-type`                 | 加载权重的类别（可以是hf、mg）                                          | ✅ |
| `--save-model-type`                 | 存储权重的类别（可以是hf、mg）                                          | ✅ |
| `--load-dir`                 | 权重文件加载路径                                          | ✅ |
| `--save-dir`                 | 权重文件保存路径                                          | ✅ |
| `--model-type-hf`                 | huggingface模型类别，默认为llama2                                  |   |
| `--params-dtype`                 | 指定权重转换后的权重精度模式，默认为fp16，如果源文件格式为bf16，则需要设置为bf16 | ✅  |

## 四、基于intent_10k数据集全参微调

### 4.1 数据集转换

对数据集处理脚本的配置进行修改：

```
cp examples/mcore/qwen3/data_convert_qwen3_instruction.sh midea_qwen3/data_convert_qwen3_instruction_intent10k.sh
mkdir -p /apps/hw/outputs/midea/qwen3/finetune_dataset_intent10k
vim midea_qwen3/data_convert_qwen3_instruction_intent10k.sh
```

```shell
python ./preprocess_data.py \
    --input  /apps/hw/datasets/midea_dataset/intent/intent_train_10k.jsonl \
    --tokenizer-name-or-path /apps/hw/weights/Qwen3-14B/ \
    --output-prefix /apps/hw/outputs/midea/qwen3/finetune_dataset_intent10k/intent \
    --handler-name SharegptStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 4 \
    --log-interval 1000 \
    --prompt-type qwen \
    --map-keys '{"messages":"messages", "tags":{"role_tag": "role","content_tag": "content","user_tag": "user","assistant_tag": "assistant","system_tag": "system"}}'
```

修改后，执行脚本

```bash
bash midea_qwen3/data_convert_qwen3_instruction_intent10k.sh
```

转换日志如下：

![image-20250802231835592](qwen3-14B业务数据集容器内验证.assets/image-20250802231835592.png)


指令微调数据集处理结果如下：

```shell
intent_packed_attention_mask_document.bin
intent_packed_attention_mask_document.idx
intent_packed_input_ids_document.bin
intent_packed_input_ids_document.idx
intent_packed_labels_document.bin
intent_packed_labels_document.idx
```

### 4.2 全参微调训练

修改单机全参微调脚本

```
cp examples/mcore/qwen3/tune_qwen3_14b_4K_full_ptd.sh midea_qwen3/tune_qwen3_14b_4K_full_intent10k_ptd.sh
vim midea_qwen3/tune_qwen3_14b_4K_full_intent10k_ptd.sh 
```

脚本内容修改如下：


```shell
export CUDA_DEVICE_MAX_CONNECTIONS=1 #定义了任务流能够利用或映射到的硬件队列的数量。
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True #内存碎片优化开关，默认False

# 基础配置
NPUS_PER_NODE=8  #使用单节点的8卡NPU
MASTER_ADDR=localhost #以本节点ip地址为master_ip
MASTER_PORT=6015 #本节点端口号为6015
NNODES=1  #单机，即一台节点，多机即多节点
NODE_RANK=0  #单机RANK为0，多机为(0,NNODES-1)，不同节点不可重复
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES)) #最终使用的NPU数

# 根据实际情况配置权重保存、权重加载、词表、数据集路径
CKPT_LOAD_DIR="/apps/hw/outputs/midea/qwen3/14b/mg_weights/qwen3_14b_mcore_tp2pp4/"  #权重加载路径，填入权重转换时保存的权重路径
CKPT_SAVE_DIR="/apps/hw/outputs/midea/qwen3/14b/save_weights_full/qwen3_14b_mcore_tp2pp4_intent10k/"  #训练完成后的权重保存路径
DATA_PATH="/apps/hw/outputs/midea/qwen3/finetune_dataset_intent10k/intent"  #数据集路径，填入数据预处理时保存的数据路径，注意需要添加后缀
TOKENIZER_PATH="/apps/hw/weights/Qwen3-14B/" #词表路径，填入下载的开源权重词表路径

TP=2
PP=4
MBS=16
GBS=128
SEQ_LENGTH=4096
TRAIN_ITERS=1000 #训练1000步

...

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
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

...

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
 	...
    --tensorboard-dir /apps/hw/outputs/qwen3/14b/tb/full/midea/intent10k_0802 \
	...
    | tee ./logs/tune_qwen3_14b_full_intent10k_0802.log

```

微调脚本相关参数说明：

- `DATA_PATH`：数据集路径。请注意实际数据预处理生成文件末尾会增加`_input_ids_document`等后缀，该参数填写到数据集的前缀即可。例如实际的数据集相对路径是`./finetune_dataset/alpaca/alpaca_packed_input_ids_document.bin`等，那么只需要填`./finetune_dataset/alpaca/alpaca`即可。
- `is-instruction-dataset`：用于指定微调过程中采用指令微调数据集，以确保模型依据特定指令数据进行微调。
- `variable-seq-lengths`：在不同的mini-batch间支持以动态的序列长度进行微调，默认padding到`8`的整数倍，可以通过`pad-to-multiple-of`参数来修改padding的倍数。详细可参考[variable length flash attention](https://gitee.com/ascend/MindSpeed-LLM/blob/2.1.0/docs/pytorch/features/variable_length_flash_attention.md)。假设微调时指定`--seq-length`序列长度为1024，开启`--variable-seq-lengths`后，序列长度会padding到真实数据长度的8整数倍。如下图所示：  ![variable_seq_lengths](qwen3-14B.assets/variable_seq_lengths-17529189902421.png)

* `--save-interval` 多少步存一次ckpt

如果是单机运行场景，只需要在一台机器上启动微调脚本：

```bash
bash midea_qwen3/tune_qwen3_14b_4K_full_intent10k_ptd.sh 
```

训练过程部分日志如下：

![image-20250804163739250](qwen3-14B业务数据集容器内验证.assets/image-20250804163739250.png)

tensorboard监控微调loss下降如下：

![image-20250804173025387](qwen3-14B业务数据集容器内验证.assets/image-20250804173025387.png)

### 4.3 全参推理验证

完成微调后，需要进一步验证模型是否具备了预期的输出能力。我们提供了简单的模型生成脚本，只需要加载微调后的模型权重，便可观察模型在不同生成参数配置下的回复。

使用推理脚本

```bash
cp examples/mcore/qwen3/generate_qwen3_14b_ptd.sh midea_qwen3/generate_qwen3_14b_full_intent10k_ptd.sh
vim midea_qwen3/generate_qwen3_14b_full_intent10k_ptd.sh
```

修改以下参数：

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1 #定义了任务流能够利用或映射到的硬件队列的数量。
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True #内存碎片优化开关，默认False

CHECKPOINT="/apps/hw/outputs/midea/qwen3/14b/save_weights_full/qwen3_14b_mcore_tp2pp4_intent10k/" # 指向微调后权重的保存路径
TOKENIZER_PATH="/apps/hw/weights/Qwen3-14B/" # 指向模型tokenizer的路径

MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1 # 集群里的节点数，以实际情况填写,
NODE_RANK=0  # 当前节点的RANK，多个节点不能重复，主节点为0, 其他节点可以是1,2..
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=2
PP=4
SEQ_LENGTH=4096
```

运行推理脚本：

```bash
bash midea_qwen3/generate_qwen3_14b_full_intent10k_ptd.sh
```

过程部分日志如下：

![image-20250804221552307](qwen3-14B业务数据集容器内验证.assets/image-20250804221552307.png)

### 4.4 intent_10k数据集评测

#### 4.4.1 NPU评测

使用评测脚本

```bash
vim midea_qwen3/qwen3_14b_full_ptd_eval_intent10k.sh
```

修改以下参数：

```bash
CHECKPOINT="/apps/hw/outputs/midea/qwen3/14b/save_weights_full/qwen3_14b_mcore_tp2pp4_intent10k/" # 指向微调后权重的保存路径
TOKENIZER_PATH="/apps/hw/weights/Qwen3-14B/" # 指向模型tokenizer的路径

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1 # 集群里的节点数，以实际情况填写,
NODE_RANK=0  # 当前节点的RANK，多个节点不能重复，主节点为0, 其他节点可以是1,2..
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=2
PP=4
SEQ_LENGTH=4096

#重点设置eval-data-path和tee的输出日志路径
torchrun $DISTRIBUTED_ARGS inference_4_midea_test.py \
...
         --eval-data-path "/apps/hw/datasets/midea_dataset/intent/intent_test_1k.jsonl" \
         --eval-data-size 1000 \
                | tee logs/generate_mcore_qwen3_14b_eval_intent10k_dataset.log

```

然后运行评测脚本：

```bash
bash examples/mcore/qwen3/qwen3_14b_full_ptd_eval_midea_dataset.sh
```

部分评测日志如下：

![image-20250804230428051](qwen3-14B业务数据集容器内验证.assets/image-20250804230428051.png)

评测结果如下，准确率

```
Average accuracy: 0.9410000000000004
```

![image-20250804230139709](qwen3-14B业务数据集容器内验证.assets/image-20250804230139709.png)

#### 4.4.2 NPU转GPU评测

##### 4.4.2.1  权重MG转为HF

训练完成后，如果需要将Megatron-Mcore（MG）权重转为HuggingFace（HF）权重，修改转换脚本

```bash
cp examples/mcore/qwen3/ckpt_convert_qwen3_mcore2hf.sh midea_qwen3/ckpt_convert_qwen3_14b_intent10k_mcore2hf.sh
vim midea_qwen3/ckpt_convert_qwen3_14b_intent10k_mcore2hf.sh
```

将 Lora权重 合并转换为HF权重的配置参数：

```shell
export CUDA_DEVICE_MAX_CONNECTIONS=1 #定义了任务流能够利用或映射到的硬件队列的数量。
python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir /apps/hw/outputs/midea/qwen3/14b/save_weights_sft/qwen3_14b_mcore_tp2pp4/ \
    --save-dir /apps/hw/weights/Qwen3-14B/ \
    --params-dtype bf16 \
    --model-type-hf qwen3
```

>  关键参数： 
>
>  * --save-model-type参数设置为 hf 
>
>  * save-dir 参数设置为原始hf权重的路径，脚本执行后，会在该目录下生成mg2hf的目录存放转换后的权重。
>  * --target-tensor-parallel-size 1， --target-pipeline-parallel-size 1 ：因为转为HF后，无切分策略，因此tp和pp都需要设置为1

运行脚本

```shell
bash midea_qwen3/ckpt_convert_qwen3_14b_intent10k_mcore2hf.sh
```

转换过程部分日志如下



##### 4.4.2.2  GPU推理评测



## 五、Lora微调

### 5.1 数据集转换

Lora 微调的数据集复用全参微调数据集即可。

### 5.2 Lora微调训练

修改单机lora微调脚本

```bash
cp examples/mcore/qwen3/tune_qwen3_14b_4K_lora_ptd.sh midea_qwen3/tune_qwen3_14b_4K_lora_intent10k_ptd.sh
vim midea_qwen3/tune_qwen3_14b_4K_lora_intent10k_ptd.sh
```

脚本内容如下：

```shell
export CUDA_DEVICE_MAX_CONNECTIONS=1  #定义了任务流能够利用或映射到的硬件队列的数量。

# 基础配置
NPUS_PER_NODE=8  #使用单节点的8卡NPU
MASTER_ADDR=localhost #以本节点ip地址为master_ip
MASTER_PORT=6015 #本节点端口号为6015
NNODES=1  #单机，即一台节点，多机即多节点
NODE_RANK=0  #单机RANK为0，多机为(0,NNODES-1)，不同节点不可重复
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES)) #最终使用的NPU数

# 根据实际情况配置权重保存、权重加载、词表、数据集路径
CKPT_LOAD_DIR="/apps/hw/outputs/midea/qwen3/14b/mg_weights/qwen3_14b_mcore_tp2pp4/"  #权重加载路径，填入权重转换时保存的权重路径
CKPT_SAVE_DIR="/apps/hw/outputs/midea/qwen3/14b/save_weights_lora/qwen3_14b_mcore_tp2pp4_intent10k/"  #训练完成后的权重保存路径
DATA_PATH="/apps/hw/outputs/midea/qwen3/finetune_dataset_intent10k/intent"  #数据集路径，填入数据预处理时保存的数据路径，注意需要添加后缀
TOKENIZER_PATH="/apps/hw/weights/Qwen3-14B/" #词表路径，填入下载的开源权重词表路径

TP=2
PP=4
SEQ_LENGTH=4096
TRAIN_ITERS=2000

...

# 重点设置lr=1.0e-5，lr-warmup-fraction=0.1
TRAIN_ARGS="
    --micro-batch-size 4 \
    --global-batch-size 64 \
    --lr 1.0e-5 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.1 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
    --no-shared-storage
"


torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
	...
    --tensorboard-dir /apps/hw/outputs/qwen3/14b/tb/lora/midea/intent10k_0804 \
    ...
    | tee /apps/hw/exps/qwen3/midea/lora0804/tune_qwen3_14b_lora_intent10k_ptd.log
```

运行脚本

```
bash midea_qwen3/tune_qwen3_14b_4K_lora_intent10k_ptd.sh
```

训练部分日志如下：

![image-20250804230714019](qwen3-14B业务数据集容器内验证.assets/image-20250804230714019.png)

tensorboard监控微调loss下降结果如下：

![image-20250805105357659](qwen3-14B业务数据集容器内验证.assets/image-20250805105357659.png)

### 5.3 Lora权重与Base权重的合并

基于lora微调训练完后，得到Lora权重与Base权重存在区别，无法直接用于续训，需要转换后才可使用，添加以下参数即可将训练好的 Lora 权重与 Base 权重融合,合并后转换为 Mcore 权重。

修改合并脚本

```
cp examples/mcore/qwen3/ckpt_convert_qwen3_mcore2hf.sh midea_qwen3/ckpt_convert_qwen3_mcore2mcore_lora_intent10k.sh
vim midea_qwen3/ckpt_convert_qwen3_mcore2mcore_lora_intent10k.sh
```

将 Lora权重 合并转换为 Mcore 权重的示例参数：

```shell
python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type mg \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 4 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --load-dir /apps/hw/outputs/midea/qwen3/14b/mg_weights/qwen3_14b_mcore_tp2pp4/ \
    --lora-load /apps/hw/outputs/midea/qwen3/14b/save_weights_lora/qwen3_14b_mcore_tp2pp4_intent10k/ \
    --save-dir /apps/hw/outputs/midea/qwen3/14b/save_weights_lora/qwen3_14b_mcore_tp2pp4_intent10k_merge/ \
    --model-type-hf qwen3
```

> 关键参数： --save-model-type mg

启动转换脚本：

```shell
bash midea_qwen3/ckpt_convert_qwen3_mcore2mcore_lora_intent10k.sh
```

转换过程部分日志如下：

![image-20250805111925230](qwen3-14B业务数据集容器内验证.assets/image-20250805111925230.png)

权重合并后的目录下：

```
rwxr-xr-x. 10 root root 4096 Aug  5 11:18 iter_0000001/
-rw-r--r--.  1 root root    1 Aug  5 11:18 latest_checkpointed_iteration.txt
```

### 5.4 LoRA推理验证

> 推理方法与全参微调推理方法相同

编辑推理脚本

```
cp examples/mcore/qwen3/generate_qwen3_14b_ptd.sh midea_qwen3/generate_qwen3_14b_lora_intent10k_ptd.sh
vim midea_qwen3/generate_qwen3_14b_lora_intent10k_ptd.sh
```

修改以下参数：

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1 #定义了任务流能够利用或映射到的硬件队列的数量。
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True #内存碎片优化开关，默认False

CHECKPOINT="/apps/hw/outputs/midea/qwen3/14b/save_weights_lora/qwen3_14b_mcore_tp2pp4_intent10k_merge/" # 指向微调后权重的保存路径

...

TP=2
PP=4
SEQ_LENGTH=4096

torchrun $DISTRIBUTED_ARGS inference.py \
	...
         | tee /apps/hw/exps/qwen3/midea/lora0804/generate_qwen3_14b_lora_intent10k.log
```

运行推理脚本：

```bash
bash midea_qwen3/generate_qwen3_14b_lora_intent10k_ptd.sh
```

过程部分日志如下：

![image-20250805112558208](qwen3-14B业务数据集容器内验证.assets/image-20250805112558208.png)

### 5.5 intent_10k数据集评测

#### 5.5.1 NPU评测

使用评测脚本

```bash
vim midea_qwen3/qwen3_14b_lora_ptd_eval_intent10k.sh
```

修改以下参数：

```bash
CHECKPOINT="/apps/hw/outputs/midea/qwen3/14b/save_weights_lora/qwen3_14b_mcore_tp2pp4_intent10k_merge/" # 指向微调后权重的保存路径
TOKENIZER_PATH="/apps/hw/weights/Qwen3-14B/" # 指向模型tokenizer的路径

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1 # 集群里的节点数，以实际情况填写,
NODE_RANK=0  # 当前节点的RANK，多个节点不能重复，主节点为0, 其他节点可以是1,2..
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=2
PP=4
SEQ_LENGTH=4096

#重点设置eval-data-path和tee的输出日志路径
torchrun $DISTRIBUTED_ARGS inference_4_midea_test.py \
...
         --eval-data-path "/apps/hw/datasets/midea_dataset/intent/intent_test_1k.jsonl" \
         --eval-data-size 1000 \
               | tee /apps/hw/exps/qwen3/midea/lora0804/generate_mcore_qwen3_14b_lora_eval_intent10k.log

```

然后运行评测脚本：

```bash
bash examples/mcore/qwen3/qwen3_14b_lora_ptd_eval_intent10k.sh 
```

部分评测日志如下：

![image-20250805114923117](qwen3-14B业务数据集容器内验证.assets/image-20250805114923117.png)

评测结果如下，准确率

```
Average accuracy: 0.795
```

![image-20250805120327386](qwen3-14B业务数据集容器内验证.assets/image-20250805120327386.png)

#### 5.5.2 NPU转GPU评测

##### 5.5.2.1 lora权重合并后转换为HF权重

若希望将lora权重合并转换为Hugging Face格式，修改合并脚本

```bash
cp examples/mcore/qwen3/ckpt_convert_qwen3_mcore2hf_lora.sh midea_qwen3/ckpt_convert_qwen3_mcore2hf_lora_intent10k.sh
vim midea_qwen3/ckpt_convert_qwen3_mcore2hf_lora_intent10k.sh
```

将 Lora权重 合并转换为HF权重的示例参数：

```shell
python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --load-dir /apps/hw/outputs/midea/qwen3/14b/mg_weights/qwen3_14b_mcore_tp2pp4/ \
    --lora-load /apps/hw/outputs/midea/qwen3/14b/save_weights_lora/qwen3_14b_mcore_tp2pp4_intent10k/ \
    --save-dir /apps/hw/weights/Qwen3-14B/ \
    --model-type-hf qwen3
```

>  * --save-model-type参数设置为 hf
>
>  * save-dir 参数设置为原始hf权重的路径，脚本执行后，会在该目录下生成mg2hf的目录存放转换后的权重。
>  * tp和pp都设置为1：  --target-tensor-parallel-size 1 ， --target-pipeline-parallel-size 1 

**注意：** `lora` 参数的值应与微调时的参数设置保持一致，以确保转换后的模型具有相同的性能表现和兼容性。

启动转换脚本：

```shell
bash midea_qwen3/ckpt_convert_qwen3_mcore2hf_lora_intent10k.sh | tee /apps/hw/exps/qwen3/midea/lora0804/ckpt_convert_qwen3_mcore2hf_lora_intent10k.log
```

转换过程部分日志如下：

![image-20250805120240184](qwen3-14B业务数据集容器内验证.assets/image-20250805120240184.png)

##### 5.5.2.2  GPU推理评测





## 六、预训练

### 6.1 预训练数据集处理

通过对各种格式的数据做提前预处理，避免原始数据的反复处理加载，将所有的数据都统一存储到为.bin和.idx两个文件中，详见[预训练数据处理](https://gitee.com/ascend/MindSpeed-LLM/docs/pytorch/solutions/pretrain/pretrain_dataset.md)。常用的预训练数据集包括alpaca、enwiki、c4等。

**预训练数据集下载**

如下以alpaca数据集为例，进行预训练数据集示例。

```shell
# 根据链接提供地址，通过wget获取数据集数据
wget https://huggingface.co/datasets/tatsu-lab/alpaca/blob/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
```

修改脚本 data_convert_qwen3_pretrain.sh 

```bash
cp examples/mcore/qwen3/data_convert_qwen3_pretrain.sh midea_qwen3/data_convert_qwen3_pretrain.sh
vim midea_qwen3/data_convert_qwen3_pretrain.sh
mkdir -p /apps/hw/outputs/midea/qwen3/pretrain_datasets_enwiki/
```

配置做如下修改：

```shell
python ./preprocess_data.py \
    --input /apps/hw/datasets/enwiki/train-00000-of-00042-d964455e17e96d5a.parquet \
    --tokenizer-name-or-path /apps/hw/weights/Qwen3-14B/ \
    --tokenizer-type PretrainedFromHF \
    --handler-name GeneralPretrainHandler \
    --output-prefix /apps/hw/outputs/midea/qwen3/pretrain_datasets_enwiki/enwiki \
    --json-keys text \
    --workers 4 \
    --log-interval 1000
```

启动转换

```bash
bash midea_qwen3/data_convert_qwen3_pretrain.sh
```

转换过程如下：



参数解析如下：

| 参数                       | 说明                                                         | 必填 |
| -------------------------- | ------------------------------------------------------------ | ---- |
| `--input`                  | 支持输入数据集目录或文件，目录则处理全部文件, 支持.parquet、.csv、.json、.jsonl、.txt、.arrow格式，同一目录要求数据格式保持一致 | ✅    |
| `--tokenizer-type`         | 说明使用tokenizer类别，参数值为PretrainedFromHF时，词表路径填写模型目录即可 | ✅    |
| `--tokenizer-name-or-path` | 配合tokenizer-type，目标模型的tokenizer原数据文件夹，用于数据集的转换 |      |
| `--tokenizer-model`        | 配合指定分词器模型的路径，路径具体到tokenizer.model文件      |      |
| `--output-prefix`          | 转换后输出的数据集文件的文件名前缀                           | ✅    |
| `--workers`                | 多进程数据集处理                                             | ✅    |

### 6.2 预训练任务启动

打开脚本

```
cp examples/mcore/qwen3/pretrain_qwen3_14b_4K_ptd.sh midea_qwen3/
vim midea_qwen3/pretrain_qwen3_14b_4K_ptd.sh
```

脚本内容修改如下：

 ```shell
export CUDA_DEVICE_MAX_CONNECTIONS=1 #定义了任务流能够利用或映射到的硬件队列的数量。
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True #内存碎片优化开关，默认False
export HCCL_IF_BASE_PORT=30000 #设置HCCL端口

# 基础配置
NPUS_PER_NODE=8  #使用单节点的8卡NPU
MASTER_ADDR=localhost #以本节点ip地址为master_ip
MASTER_PORT=6015 #本节点端口号为6015
NNODES=1  #单机，即一台节点，多机即多节点
NODE_RANK=0  #单机RANK为0，多机为(0,NNODES-1)，不同节点不可重复
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES)) #最终使用的NPU数

# 根据实际情况配置权重保存、权重加载、词表、数据集路径
CKPT_LOAD_DIR="/apps/hw/outputs/midea/qwen3/14b/mg_weights/qwen3_14b_mcore_tp2pp4/"  #权重加载路径，填入权重转换时保存的权重路径
CKPT_SAVE_DIR="/apps/hw/outputs/midea/qwen3/14b/save_weights_pretrain/qwen3_14b_mcore_tp2pp4/"  #训练完成后的权重保存路径
DATA_PATH="/apps/hw/outputs/midea/qwen3/pretrain_datasets_enwiki/enwiki_text_document"  #数据集路径，填入数据预处理时保存的数据路径，注意需要添加后缀
TOKENIZER_PATH="/apps/hw/weights/Qwen3-14B/" #词表路径，填入下载的开源权重词表路径

TP=2  #权重转换设置--target-tensor-parallel-size 2，修改为2
PP=4  #权重转换设置--target-pipeline-parallel-size 4，修改为4
CP=1  #序列并行
MBS=1 #设置micro-batch-size为1
GBS=128 #设置global-batch-size为64
SEQ_LENGTH=4096  #设置seq_length为4096 
TRAIN_ITERS=2000 #训练多少步
CP_TYPE='ulysses_cp_algo' #序列并行算法
 ```

启动脚本
```shell
bash midea_qwen3/pretrain_qwen3_14b_4K_ptd.sh
```

训练日志如下：



tensorboard监控微调loss下降结果如下：



脚本中特性包含部分参数解释

| 参数名                                  | 说明                                                         |
| --------------------------------------- | ------------------------------------------------------------ |
| `--use-mcore-models`                    | 使用mcore分支运行模型                                        |
| `--disable-bias-linear`                 | 去掉linear的偏移值，与qwen原模型一致                         |
| `--add-qkv-bias`                        | 增加Q、K、V的偏移值，是权重的组成部分                        |
| `--group-query-attention`               | 开启GQA注意力处理机制                                        |
| `--num-query-groups 4`                  | 配合GQA使用，设置groups为4                                   |
| `--position-embedding-type rope`        | 位置编码采用rope方案                                         |
| `--untie-embeddings-and-output-weights` | 根据原模型要求将output层和embedding层的权重解耦              |
| `--bf16`                                | 昇腾芯片对BF16精度支持良好，可显著提升训练速度               |
| `--tensorboard-dir`                     | 如需要tensorboard显示loss曲线，增加此参数，用于事件存储目录，如`/apps/hw/outputs/midea/qwen3/14b/tb/pretrain `。查看训练loss曲线使用方法为 `tensorboard --logdir=/apps/hw/outputs/midea/qwen3/14b/tb --port=3000 --bind_all` |

如果需要后台启动任务

```bash
nohup bash midea_qwen3/pretrain_qwen3_14b_4K_ptd.sh >pretrain_qwen3_14b_4K_ptd.log 2>&1 &
```

终止训练进程

```bash
pkill -9 python
```

如果是多机运行，则需要在单机的脚本上修改以下参数：

```shell
# 多机配置 
# 根据分布式集群实际情况配置分布式参数
GPUS_PER_NODE=8  # 每个节点的卡数
MASTER_ADDR="your master node IP"  # 都需要修改为主节点的IP地址（不能为localhost）
MASTER_PORT=6000
NNODES=2  # 集群里的节点数，以实际情况填写,
NODE_RANK="current node id"  # 当前节点的RANK，多个节点不能重复，主节点为0, 其他节点可以是1,2..
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
```

最后确保每台机器上的模型路径和数据集路径等无误后，在多个终端上同时启动预训练脚本即可开始训练。

**注意**：

- 多机训练需在多个终端同时启动预训练脚本(每个终端的预训练脚本只有NODE_RANK参数不同，其他参数均相同)
- 如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`--no-shared-storage`参数，设置此参数之后将会根据布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据

## 七、评测汇总



## 八、附加说明

### 7.1 模型训练常见环境变量说明

模型训练常见环境变量说明：

| 环境变量名称                | 环境变量描述                                                 |
| --------------------------- | ------------------------------------------------------------ |
| ASCEND_LAUNCH_BLOCKING      | 将Host日志输出到串口,0-关闭/1-开启。                         |
| ASCEND_SLOG_PRINT_TO_STDOUT | 设置默认日志级别,0-debug/1-info/2-warning/3-error。          |
| HCCL_WHITELIST_DISABLE      | HCCL白名单开关,1-关闭/0-开启。                               |
| HCCL_CONNECT_TIMEOUT        | 设置HCCL超时时间，默认值为120。                              |
| HCCL_NPU_SOCKET_PORT_RANGE  | 配置HCCL在NPU侧使用的通信端口，支持配置为具体的端口、端口范围或者字符串“auto”。[详细参考](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/apiref/envvar/envref_07_0144.html) |
| CUDA_DEVICE_MAX_CONNECTIONS | 定义了任务流能够利用或映射到的硬件队列的数量。               |
| TASK_QUEUE_ENABLE           | 用于控制开启task_queue算子下发队列优化的等级，0：关闭，1：开启Level 1优化，2：开启Level 2优化。 |
| COMBINED_ENABLE             | 设置combined标志。设置为0表示关闭此功能；设置为1表示开启，用于优化非连续两个算子组合类场景。 |
| PYTORCH_NPU_ALLOC_CONF      | 内存碎片优化开关，默认是expandable_segments:False，使能时expandable_segments:True。 |
| ASCEND_RT_VISIBLE_DEVICES   | 指定哪些Device对当前进程可见，支持一次指定一个或多个Device ID。通过该环境变量，可实现不修改应用程序即可调整所用Device的功能。 |
| NPUS_PER_NODE               | 配置一个计算节点上使用的NPU数量。                            |
| HCCL_SOCKET_IFNAME          | 指定hccl socket通讯走的网卡配置。                            |
| GLOO_SOCKET_IFNAME          | 指定gloo socket通讯走的网卡配置。                            |
| HCCL_LOGIC_SUPERPOD_ID      | 指定当前设备的逻辑超节点ID，如果走ROCE，不同多机超节点ID不同，0-N。 |

### 7.2 训练参数说明

训练脚本常见参数解释

| 参数                              | 说明                                                         | 必填 |
| --------------------------------- | ------------------------------------------------------------ | ---- |
| `--model-type GPT`                | 指定模型类型为GPT系列                                        | ✅    |
| `--use-mcore-models`              | 转换为Megatron-Mcore格式                                     | ✅    |
| `--target-tensor-parallel-size`   | 张量并行度设置                                               | ✅    |
| `--target-pipeline-parallel-size` | 流水线并行度设置                                             | ✅    |
| `--tokenizer-model`               | 指定分词器路径                                               | ✅    |
| `--load-model-type`               | 加载权重的类别（可以是hf、mg）                               | ✅    |
| `--save-model-type`               | 存储权重的类别（可以是hf、mg）                               | ✅    |
| `--load-dir`                      | 权重文件加载路径                                             | ✅    |
| `--save-dir`                      | 权重文件保存路径                                             | ✅    |
| `--model-type-hf`                 | huggingface模型类别，默认为llama2                            |      |
| `--params-dtype`                  | 指定权重转换后的权重精度模式，默认为fp16，如果源文件格式为bf16，则需要设置为bf16 | ✅    |
| `--tensorboard-dir`               | 开启tensorboard数据收集并指定tensorboard的事件存储目录，如` ./tb `。查看训练loss曲线使用方法为 `tensorboard --logdir=./tb --port=3000` |      |
