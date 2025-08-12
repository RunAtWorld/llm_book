# Qwen3-14B适配

## 一、依赖安装&制作镜像

### 1.1 HDK与CANN安装

HDK安装：参考版本配套表，根据系统和硬件产品型号，选择对应版本的`driver`和`firmware`下载[驱动固件](https://www.hiascend.com/hardware/firmware-drivers/community)，并参考完成[NPU驱动固件安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/softwareinst/instg/instg_0005.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)，安装后重启os系统生效。容器环境一般安装在宿主机上。

CANN安装：参考版本配套表，根据系统选择`aarch64`或`x86_64`对应版本的`cann-toolkit`、`cann-kernel`和`cann-nnal`，下载[CANN软件包](https://www.hiascend.com/developer/download/community/result?module=cann)，参考[CANN安装教程](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/softwareinst/instg/instg_0008.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)完成cann包安装。容器环境一般安装在容器内。

### 1.2 创建python环境

```shell
conda create --name py310 python=3.10
conda activate py310
```

### 1.3 PTA安装


准备[torch_npu](https://www.hiascend.com/developer/download/community/result?module=pt)和[apex](https://gitee.com/ascend/apex)，参考[Ascend Extension for PyTorch 配置与安装](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/configandinstg/instg/insg_0001.html)或执行以下命令安装：

```shell
# 安装torch和torch_npu 
#torch参考 https://download.pytorch.org/whl/cpu/
#torch_npu构建参考 https://gitee.com/ascend/pytorch/releases
pip install https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp310-cp310-manylinux_2_28_aarch64.whl
pip install https://gitee.com/ascend/pytorch/releases/download/v7.0.RC1.alpha002-pytorch2.6.0/torch_npu-2.6.0rc1-cp310-cp310-manylinux_2_28_aarch64.whl

# apex for Ascend 需要自行参考 https://gitee.com/ascend/apex 完成whl包的构建，再通过如下指令完成安装
pip install apex-0.1.dev20241015+ascend-cp310-cp310-linux_aarch64.whl
```

执行如下命令，若返回True则说明安装成功。

```
python3 -c "import torch;import torch_npu;print(torch_npu.npu.is_available())"
```

**apex构建参考命令**

> 注意：后续过程经常需要CANN环境变量，为了方便可以提前讲相关生效环境变量的脚本写入 .bashrc 中
```
# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```
```
git clone -b master https://gitee.com/ascend/apex.git
cd apex/
#编译apex的二进制包
bash scripts/build.sh --python=3.10
#安装
cd apex/dist/
pip3 uninstall apex
pip3 install --upgrade torch_npu-2.6.0rc1-cp310-cp310-manylinux_2_28_aarch64.whl # version中包含python版本和cpu架构
```
### 1.4 MindSpeed-LLM及相关依赖安装

```shell
# 使能环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 安装MindSpeed加速库
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout -b core_r0.8.0
git checkout 2c085cc9  # checkout commit from MindSpeed core_r0.8.0 in 2025.04.01
pip install -r requirements.txt 
pip3 install -e .
cd ..

# 准备MindSpeed-LLM及Megatron-LM源码
git clone https://gitee.com/ascend/MindSpeed-LLM.git 
git clone https://github.com/NVIDIA/Megatron-LM.git  # megatron从github下载，请确保网络能访问
cd Megatron-LM
git checkout core_r0.8.0
cp -r megatron ../MindSpeed-LLM/
cd ../MindSpeed-LLM
git checkout 2.1.0

pip install -r requirements.txt  # 安装其余依赖库
```

由于qwen3依赖 transformers==4.51.0，单独安装
```shell
pip install transformers==4.51.0
```

所有依赖都安装后，可以将环境保存为镜像以便下次使用

```
docker commit -a "lipengfei" -m "add nnal"  qwen3_img a3_llm_poc:v2.1.3
```

如果不需要历史信息，将容器存储为镜像

```
docker export qwen3_14b_pretrain | docker import - a3_llm_poc:v2.1.0
```

以上安装后，相关组件配套表如下：

| 软件          | 版本               | 部署方式 |
| ------------- | ------------------ | -------- |
| Driver        | AscendHDK 25.0.RC1 | 宿主机   |
| Firmware      | AscendHDK 25.0.RC1 | 宿主机   |
| Python        | 3.8.10             | 容器内   |
| CANN          | CANN 8.1.RC1       | 容器内   |
| Torch         | 2.1.0              | 容器内   |
| Torch_npu     | release v7.0.0     | 容器内   |
| MindSpeed     | 2.0.0_core_r0.8.0  | 容器内   |
| MindSpeed-LLM | 2.1.0              | 容器内   |
| Megatron-LM   | core_r0.8.0        | 容器内   |
| Docker镜像OS  | Ubuntu 20.04.6     | /        |

## 二、创建容器

```bash
docker run -dit -u root --privileged --ipc=host \
    --name 'qwen3_14b_pretrain' \
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
    -p 50002:3000 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/sbin/:/usr/local/sbin/ \
    -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
    -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
    -v /apps:/apps \
    -v /home:/home \
    a3_llm:v2.1.0 bash
```

> 增加 --network host 可以映射使用主机网络

## 三、预训练

### 3.1 权重转换

昇腾MindSpeed-LLM要求模型权重采用Megatron-LM格式，在这里我们将原始HuggingFace权重格式转换为Megatron-Mcore格式。

使用官方提供的转换脚本，获取对应切分的mg权重。

```bash
bash examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
# 请先根据如下指导完成脚本修改配置
```

如下 ckpt_convert_qwen3_hf2mcore.sh 为调整后的hf2mcore权重转换示例脚本

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
# 生效 ascend-toolkit 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

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
    --save-dir /apps/hw/outputs/qwen3/14b/mg_weights/qwen3_14b_mcore_tp2pp4/ \
    --tokenizer-model /apps/hw/weights/Qwen3-14B/tokenizer.json \
    --params-dtype bf16 \
    --model-type-hf qwen3
```

![image-20250729164531346](qwen3-14B.assets/image-20250729164531346.png)

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

### 3.2 预训练数据集处理

通过对各种格式的数据做提前预处理，避免原始数据的反复处理加载，将所有的数据都统一存储到为.bin和.idx两个文件中，详见[预训练数据处理](https://gitee.com/ascend/MindSpeed-LLM/docs/pytorch/solutions/pretrain/pretrain_dataset.md)

常用的预训练数据集包括alpaca、enwiki、c4等，链接中提供了数据集下载地址。

**预训练数据集下载**

如下以alpaca数据集为例，进行预训练数据集示例。

```shell
# 根据链接提供地址，通过wget获取数据集数据
mkdir dataset
cd dataset/
wget https://huggingface.co/datasets/tatsu-lab/alpaca/blob/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
cd ..

# 使用仓库提供的数据处理脚本，转换预训练数据集为mg格式
bash examples/mcore/qwen3/data_convert_qwen3_pretrain.sh
# 请根据如下指导完成脚本修改配置
```
data_convert_qwen3_pretrain.sh 中的配置需做如下修改：
```shell
python ./preprocess_data.py \
    --input /apps/hw/datasets/enwiki/train-00000-of-00042-d964455e17e96d5a.parquet \
    --tokenizer-name-or-path /apps/hw/weights/Qwen3-14B/ \
    --tokenizer-type PretrainedFromHF \
    --handler-name GeneralPretrainHandler \
    --output-prefix /apps/hw/outputs/qwen3/pretrain_datasets_enwiki/enwiki \
    --json-keys text \
    --workers 4 \
    --log-interval 1000# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

mkdir ./dataset

python ./preprocess_data.py \
    --input /apps/hw/datasets/train-00000-of-00042-d964455e17e96d5a.parquet \
    --tokenizer-name-or-path /apps/hw/weights/Qwen3-14B/ \
    --tokenizer-type PretrainedFromHF \
    --handler-name GeneralPretrainHandler \
    --output-prefix /apps/hw/outputs/qwen3/pretrain_datasets/enwiki \
    --json-keys text \
    --workers 4 \
    --log-interval 1000
```

参数解析

| 参数                       | 说明                                                               | 必填 |
|---------------------------|------------------------------------------------------------------|--|
| `--input`                 | 支持输入数据集目录或文件，目录则处理全部文件, 支持.parquet、.csv、.json、.jsonl、.txt、.arrow格式，同一目录要求数据格式保持一致 | ✅ |
| `--tokenizer-type` | 说明使用tokenizer类别，参数值为PretrainedFromHF时，词表路径填写模型目录即可               | ✅ |
| `--tokenizer-name-or-path`| 配合tokenizer-type，目标模型的tokenizer原数据文件夹，用于数据集的转换                   |  |
| `--tokenizer-model`       | 配合指定分词器模型的路径，路径具体到tokenizer.model文件                              |  |
| `--output-prefix`  | 转换后输出的数据集文件的文件名前缀                                                | ✅ |
| `--workers`               | 多进程数据集处理                                                         | ✅ |

### 3.3 预训练任务启动

完成了数据集处理和权重转换之后，可以开始拉起预训练任务。

#### 3.3.1 启动单机预训练

打开示例脚本

```
vim examples/mcore/qwen3/pretrain_qwen3_14b_4K_ptd.sh
```

示例脚本内容如下：

 ```shell
export CUDA_DEVICE_MAX_CONNECTIONS=1 #定义了任务流能够利用或映射到的硬件队列的数量。
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True #内存碎片优化开关，默认False
export HCCL_IF_BASE_PORT=30000 #设置HCCL端口
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

# 基础配置
NPUS_PER_NODE=8  #使用单节点的8卡NPU
MASTER_ADDR=localhost #以本节点ip地址为master_ip
MASTER_PORT=6015 #本节点端口号为6015
NNODES=1  #单机，即一台节点，多机即多节点
NODE_RANK=0  #单机RANK为0，多机为(0,NNODES-1)，不同节点不可重复
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES)) #最终使用的NPU数

# 根据实际情况配置权重保存、权重加载、词表、数据集路径
CKPT_LOAD_DIR="/apps/hw/outputs/qwen3/14b/mg_weights/qwen3_14b_mcore_tp2pp4/"  #权重加载路径，填入权重转换时保存的权重路径
CKPT_SAVE_DIR="/apps/hw/outputs/qwen3/14b/save_weights_pretrain/qwen3_14b_mcore_tp2pp4/"  #训练完成后的权重保存路径
DATA_PATH="/apps/hw/outputs/qwen3/pretrain_datasets/enwiki_text_document"  #数据集路径，填入数据预处理时保存的数据路径，注意需要添加后缀
TOKENIZER_PATH="/apps/hw/weights/Qwen3-14B/" #词表路径，填入下载的开源权重词表路径

TP=2  #权重转换设置--target-tensor-parallel-size 2，修改为2
PP=4  #权重转换设置--target-pipeline-parallel-size 4，修改为4
CP=1  #序列并行
MBS=1 #设置micro-batch-size为1
GBS=128 #设置global-batch-size为64
SEQ_LENGTH=4096  #设置seq_length为4096 
TRAIN_ITERS=2000 #训练多少步
CP_TYPE='ulysses_cp_algo' #序列并行算法
# 完成如上修改后保存关闭脚本
 ```

```shell
# 初始化环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 启动预训练脚本
bash examples/mcore/qwen3/pretrain_qwen3_14b_4K_ptd.sh
```

![image-20250718113913675](qwen3-14B.assets/image-20250718113913675.png)

脚本中特性包含训练参数也包含优化特性，如下部分参数解释

| 参数名                                     | 说明                              |
|-----------------------------------------|---------------------------------|
| `--use-mcore-models`                    | 使用mcore分支运行模型     |
| `--disable-bias-linear`                 | 去掉linear的偏移值，与qwen原模型一致         |
| `--add-qkv-bias`                        | 增加Q、K、V的偏移值，是权重的组成部分            |
| `--group-query-attention`               | 开启GQA注意力处理机制                    |
| `--num-query-groups 4`                  | 配合GQA使用，设置groups为4              |
| `--position-embedding-type rope`        | 位置编码采用rope方案                    |
| `--untie-embeddings-and-output-weights` | 根据原模型要求将output层和embedding层的权重解耦 |
| `--bf16`                                | 昇腾芯片对BF16精度支持良好，可显著提升训练速度       |
| `--tensorboard-dir`               | 如需要tensorboard显示loss曲线，增加此参数，用于事件存储目录，如`/apps/hw/outputs/qwen3/14b/tb/pretrain `。查看训练loss曲线使用方法为 `tensorboard --logdir=/apps/hw/outputs/qwen3/14b/tb --port=3000 --bind_all` |

训练任务启动后，日志存储在 logs 下。

如果需要后台启动任务

```
nohup bash examples/mcore/qwen3/pretrain_qwen3_14b_4K_ptd.sh >pretrain_qwen3_14b_4K_ptd.log 2>&1 &
```

终止训练进程

```
pkill -9 python
```

#### 3.3.2 启动多机预训练任务(待验证)

启动多机预训练任务，对在单机预训练脚本的基础上

```
vim examples/mcore/qwen3/pretrain_qwen3_14b_4K_ptd.sh
```

做如下修改

 ```shell
# 多机配置如下
NPUS_PER_NODE=8           # 同单机
MASTER_ADDR=${master_ip}  # 参与多机训练的节点都配置为master_ip
MASTER_PORT=6015          # 本节点端口号为6000
NNODES=1                  # 根据参与节点数量配置
NODE_RANK=0               # 多机为(0,NNODES-1)，不同节点不可重复，master_node rank为0，其ip为master_ip
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))


# 根据实际情况配置权重保存、权重加载、词表、数据集路径
CKPT_LOAD_DIR="/apps/hw/outputs/qwen3/14b/mg_weights/qwen3_14b_mcore_tp2pp4/"  #权重加载路径，填入权重转换时保存的权重路径
CKPT_SAVE_DIR="/apps/hw/outputs/qwen3/14b/save_weights_pretrain/qwen3_14b_mcore_tp2pp4/"  #训练完成后的权重保存路径
DATA_PATH="/apps/hw/outputs/qwen3/pretrain_datasets/enwiki_text_document"  #数据集路径，填入数据预处理时保存的数据路径，注意需要添加后缀
TOKENIZER_PATH="/apps/hw/weights/Qwen3-14B/" #词表路径，填入下载的开源权重词表路径
 ```

**注意**：

- 多机训练需在多个终端同时启动预训练脚本(每个终端的预训练脚本只有NODE_RANK参数不同，其他参数均相同)
- 如果使用多机训练，且没有设置数据共享，需要在训练启动脚本中增加`--no-shared-storage`参数，设置此参数之后将会根据布式参数判断非主节点是否需要load数据，并检查相应缓存和生成数据

### 3.5 权重MG转为HF

修改转换脚本 ckpt_convert_qwen3_mcore2hf.sh

```
vim examples/mcore/qwen3/ckpt_convert_qwen3_mcore2hf_pretrain.sh
```

将 Lora权重 合并转换为HF权重的示例参数：

```shell
# 生效 ascend-toolkit 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir /apps/hw/outputs/qwen3/14b/save_weights_pretrain/qwen3_14b_mcore_tp2pp4/ \
    --save-dir /apps/hw/weights/Qwen3-14B/ \
    --params-dtype bf16 \
    --model-type-hf qwen3
```

>  关键参数： 
>
>  * --save-model-type参数设置为 hf 
>
>  * save-dir 参数设置为原始hf权重的路径，脚本执行后，会在该目录下生成mg2hf的目录存放转换后的权重。
>  * tp和pp都设置为1：  --target-tensor-parallel-size 1 ， --target-pipeline-parallel-size 1 

```shell
bash examples/mcore/qwen3/ckpt_convert_qwen3_mcore2hf_pretrain.sh
```

运行日志如下

![image-20250729170847637](qwen3-14B.assets/image-20250729170847637.png)

## 四、SFT微调

### 4.1 指令微调场景介绍
#### 4.1.1 单样本微调

每条数据为一个独立的任务样本，包括指令和目标回复。适合用于问答、翻译、单轮任务等。

- 数据格式示例：

    ```json
    {
        "instruction": "请将下面的句子翻译成英文：我爱自然语言处理。",
        "response": "I love natural language processing."
    }
    ```

#### 4.1.2 多样本pack微调

为了提高训练效率，将多个样本拼接打包成一个长序列，减少填充，提高显存利用率。

- 原始样本：

    ```json
    [
        {
            "instruction": "请将以下句子翻译成英文：我们正在开发一款新的人工智能助手。",
            "response": "We are developing a new AI assistant."
        },
        {
            "instruction": "请列出三个可再生能源的例子。",
            "response": "太阳能、风能和水能是三种常见的可再生能源。"
        }
    ]
    ```

- 拼接后输入示例：

    ```json
    <bos> Instruction: 请将以下句子翻译成英文：我们正在开发一款新的人工智能助手。
    Response: We are developing a new AI assistant. <eos>
    Instruction: 请列出三个可再生能源的例子。
    Response: 太阳能、风能和水能是三种常见的可再生能源。 <eos>
    ```

#### 4.1.3 多轮对话微调

用于训练模型进行连续对话，保留上下文信息。每条样本包含一轮以上的用户和模型对话。

- 数据示例：

    ```json
    {
        "conversations": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，有什么我可以帮你的吗？"},
            {"role": "user", "content": "什么是强化学习？"},
            {"role": "assistant", "content": "强化学习是一种让智能体通过试错方式学习策略的方法。"}
        ]
    }
    ```

- 构造的输入示例：

    ```json
    <user>: 你好
    <assistant>: 你好，有什么我可以帮你的吗？
    <user>: 什么是强化学习？
    <assistant>: 强化学习是一种让智能体通过试错方式学习策略的方法。
    ```

### 4.2 使用Alpaca数据集

#### 4.2.1 Alpaca数据集

Alpaca风格示例：

```
[
{
    "instruction": "人类指令（必填）",
    "input": "人类输入（选填）",
    "output": "模型回答（必填）",
    "system": "系统提示词（选填）",
    "history": [
    ["第一轮指令（选填）", "第一轮回答（选填）"],
    ["第二轮指令（选填）", "第二轮回答（选填）"]
    ]
}
]
```

常用的对话指令微调数据集有：
- [单轮对话：Alpaca英文数据集](https://huggingface.co/datasets/tatsu-lab/alpaca)
- [单轮对话：Alpaca中文数据集](https://huggingface.co/datasets/llm-wizard/alpaca-gpt4-data-zh/tree/main)
- [多轮对话：AlpacaHistroy数据集](https://huggingface.co/datasets/kimnt93/oaast-selected)
- [链式思维 (CoT)：Alpaca数据集](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT/tree/main/Auto-CoT)
- [BELLE：指令微调数据集](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN)

Alpaca风格微调数据集下载可以基于网页直接下载，也可以基于命令行下载，比如：

```bash
wget https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet
```
#### 4.2.2 Alpaca风格数据集处理

##### 4.2.2.1 Alpaca单轮单样本数据集处理

在指令监督微调时，`instruction` 列对应的内容会与 `input` 列对应的内容拼接后作为人类指令，即人类指令为 `instruction\ninput`其中 `\n`为用于连接的换行符。而 `output` 列对应的内容为模型回答。如果指定了history，则会将历史对话内容也加入进来。如果指定system 列，则对应的内容将被作为系统提示词。

对数据集处理脚本 data_convert_qwen3_instruction.sh 中的配置进行修改：

```
vim examples/mcore/qwen3/data_convert_qwen3_instruction.sh
```

```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

mkdir ./finetune_dataset

python ./preprocess_data.py \
    --input  /apps/hw/datasets/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
    --tokenizer-name-or-path /apps/hw/weights/Qwen3-14B/ \
    --output-prefix /apps/hw/outputs/qwen3/finetune_dataset/alpaca \
    --handler-name AlpacaStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 4 \
    --log-interval 1000 \
    --prompt-type qwen
    # --map-keys '{"prompt":"instruction","query":"input","response":"output"}' # 默认值，可不传
```

修改后，执行脚本

```
mkdir -p /apps/hw/outputs/qwen3/finetune_dataset
bash examples/mcore/qwen3/data_convert_qwen3_instruction.sh
```

转换日志如下：

![image-20250729172451260](qwen3-14B.assets/image-20250729172451260.png)

参数解析如下：

* `--input`: 可以直接输入到数据集目录或具体文件，如果是目录，则处理全部文件, 支持 .parquet \ .csv \ .json \ .jsonl \ .txt \ .arrow 格式， 同一个文件夹下的数据格式需要保持一致 
* `--prompt-type`: 用于指定模型模板，能够让base模型微调后能具备更好的对话能力。`prompt-type`的可选项可以在[templates](https://gitee.com/ascend/MindSpeed-LLM/blob/2.1.0/mindspeed_llm/tasks/preprocess/templates.py)文件内查看。
* `--map-keys`: 参数用于配置字段映射来使用数据集。参数完整应为`'{"prompt":"instruction","query":"input","response":"output","system":"system","history":"history"}'`。其中参数的key值`"prompt"、"query"、"response"、"system"、"history"`代表数据集列映射后的属性，在代码中是固定的，不应改变，value值`"instruction"、"input"、"output"、"system"、"history"`对应数据集的列名。考虑到alpaca数据集大部分都是`["instruction", "input", "output"]`型格式，因此我们为key值`["prompt", "query", "response"]`设置了默认值。因此上面格式`--map-keys`参数可简略为`'{"system": "system","history": "history"}'`。若数据集中无`system`与`history`列，则`--map-keys`可省略。

**示例1：**

```
--map-keys '{"prompt":"notice","query":"question","response":"answer","system":"system_test","history":"histories"}'
```

则会提取数据集里的`"notice"、"question"、"answer"、"system_test"、"histories"`列。

**示例2：**

```
--map-keys '{"history":"histories"}'
```

则会提取数据集里的`"instruction"、"input"、"output"、"histories"`列，其中`"instruction"、"input"、"output"`列作为默认值隐式存在。 

转换后，finetune_dataset 目录下文件如下：

```
alpaca_packed_attention_mask_document.bin
alpaca_packed_attention_mask_document.idx
alpaca_packed_input_ids_document.bin
alpaca_packed_input_ids_document.idx
alpaca_packed_labels_document.bin
alpaca_packed_labels_document.idx
```

##### 4.2.2.2 Alpaca单轮多样本数据pack处理

##### 4.2.2.3 Alpaca多轮对话数据处理

### 4.3  使用ShareGPT数据集

#### 4.3.1 对话指令微调数据集

常用的对话指令微调数据集有：

- [Roleplay多轮对话：ShareGPT数据集](https://huggingface.co/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data)
- [Chain-of-though: ShareGPT数据集](https://huggingface.co/datasets/isaiahbjork/chain-of-thought-sharegpt)
- [Capybara: ShareGPT数据集](https://huggingface.co/datasets/Undi95/Capybara-ShareGPT)

`ShareGPT` 风格微调数据集下载可以基于网页直接下载，也可以基于命令行下载，比如：

```shell
wget https://huggingface.co/datasets/shibing624/roleplay-zh-sharegpt-gpt4-data/resolve/main/sharegpt_formatted_data-evol-gpt4.jsonl
```

#### 4.3.2 ShareGPT风格数据集处理

`ShareGPT` 格式支持更多的角色种类，例如 `human、gpt、observation、function`等等。它们构成一个对象列表呈现在`conversations`列中。

`ShareGPT`风格示例：

```
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "人类指令"
      },
      {
        "from": "function_call",
        "value": "工具参数"
      },
      {
        "from": "observation",
        "value": "工具结果"
      },
      {
        "from": "gpt",
        "value": "模型回答"
      }
    ],
    "system": "系统提示词（选填）",
    "tools": "工具描述（选填）"
  }
]
```

`ShareGPT`格式数据预处理示例脚本：

```shell
# 请按照真实环境 source set_env.sh 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir ./finetune_dataset

python ./preprocess_data.py \
    --input ./dataset/sharegpt_formatted_data-evol-gpt4.jsonl \
    --tokenizer-name-or-path ./model_from_hf/llama-2-7b-hf/ \
    --output-prefix ./finetune_dataset/sharegpt \
    --workers 4 \
    --log-interval 1000 \
    --tokenizer-type PretrainedFromHF \
    --handler-name SharegptStyleInstructionHandler \
    --prompt-type llama2  #需要填入模型模板
    # --map-keys '{"messages":"conversations", "tags":{"role_tag": "from","content_tag": "value","user_tag": "human","assistant_tag": "gpt","system_tag": "system", "observation_tag":"observation", "function_tag":"function_call"}}' # 默认值，可不传
```

* `--prompt-type`: 用于指定模型模板，能够让base模型微调后能具备更好的对话能力。`prompt-type`的可选项可以在[templates](../../../../../configs/finetune/templates.json)文件内查看。
* `--map-keys`:参数用于配置字段映射来使用数据集。默认值为

    ```
    '{"messages":"conversations", "tags":{"role_tag": "from","content_tag": "value","user_tag": "human","assistant_tag": "gpt","system_tag": "system", "observation_tag":"observation", "function_tag":"function_call"}}'
    ```

    其中key值`"messages"、"tags"`代表数据集列映射后的属性，在代码中是固定的，不应改变。value值中`"conversations"`对应数据集的列名、`"from"`对应角色标志、`"human"、"gpt"、"system"、"observation"、"function_call"`对应角色种类、`"value"`对应具体内容标志。

    以OpenAI格式为例，OpenAI 格式是 sharegpt 格式的一种特殊情况，其中第一条消息可能是系统提示词。OpenAI格式示例：
    ```
    [
    {
        "messages": [
        {
            "role": "system",
            "content": "系统提示词（选填）"
        },
        {
            "role": "user",
            "content": "人类指令"
        },
        {
            "role": "assistant",
            "content": "模型回答"
        }
        ]
    }
    ]
    ```

    OpenAI格式数据预处理示例脚本：

    ```shell
    # 请按照您的真实环境 source set_env.sh 环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    mkdir ./finetune_dataset

    python ./preprocess_data.py \
        --input ./dataset/xxx.json \
        --tokenizer-name-or-path ./model_from_hf/llama-2-7b-hf/ \
        --output-prefix ./finetune_dataset/openai \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF \
        --handler-name SharegptStyleInstructionHandler \
        --prompt-type llama2 \
        --map-keys '{"messages":"messages", "tags":{"role_tag": "role","content_tag": "content","user_tag": "user","assistant_tag": "assistant","system_tag": "system"}}'
    ```

* `--handler-name`: 微调数据预处理`ShareGPT`风格数据集时，应指定为`SharegptStyleInstructionHandler`，并根据`--map-keys`参数提取对应数据的列。

    **示例1：**

    ```
    --map-keys '{"messages":"chat"}'
    ```

    则会提取数据集里的`"chat"`列，其中`"tags"`属性包含角色格式和内容格式，做为默认值隐式存在，角色格式可以为：`"from": "human"、"from": "gpt"、"from": "observation"、"from": "function_call"`，内容格式为`"value": "具体内容"`

    **示例2：**

    ```
    --map-keys '{"messages":"messages", "tags":{"role_tag": "role","content_tag": "content","user_tag": "user","assistant_tag": "assistant"}}'
    ```

    则会提取数据集里的`"messages"`列，其中角色格式可以为：`"role": "user"、"role": "assistant"`，内容格式为`"content": "具体内容"`

#### 4.3.3 转换启动脚本

对数据集处理脚本 data_convert_qwen3_instruction.sh 中的配置进行修改：

```
cp examples/mcore/qwen3/data_convert_qwen3_instruction.sh examples/mcore/qwen3/data_convert_qwen3_instruction_sharegpt.sh
vim examples/mcore/qwen3/data_convert_qwen3_instruction_sharegpt.sh
```

```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

mkdir -p /apps/hw/outputs/qwen3/finetune_dataset_sharegpt
python ./preprocess_data.py \
    --input  /apps/hw/datasets/roleplay-zh-sharegpt-gpt4-data/sharegpt_formatted_data-evol-gpt4.jsonl \
    --tokenizer-name-or-path /apps/hw/weights/Qwen3-14B/ \
    --output-prefix /apps/hw/outputs/qwen3/finetune_dataset_sharegpt/sharegpt \
    --handler-name SharegptStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 4 \
    --log-interval 1000 \
    --prompt-type qwen
    # --map-keys '{"messages":"conversations", "tags":{"role_tag": "from","content_tag": "value","user_tag": "human","assistant_tag": "gpt","system_tag": "system", "observation_tag":"observation", "function_tag":"function_call"}}' # 默认值，可不传
```

修改后，执行脚本

```
bash examples/mcore/qwen3/data_convert_qwen3_instruction_sharegpt.sh
```

指令微调数据集处理结果如下：

```shell
sharegpt_packed_attention_mask_document.bin
sharegpt_packed_attention_mask_document.idx
sharegpt_packed_input_ids_document.bin
sharegpt_packed_input_ids_document.idx
sharegpt_packed_labels_document.bin
sharegpt_packed_labels_document.idx
```

### 4.4 SFT微调训练

修改单机微调脚本 tune_qwen3_14b_4K_full_ptd.sh 

```
vim examples/mcore/qwen3/tune_qwen3_14b_4K_full_ptd.sh 
```

示例脚本内容如下：


```shell
export CUDA_DEVICE_MAX_CONNECTIONS=1 #定义了任务流能够利用或映射到的硬件队列的数量。
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True #内存碎片优化开关，默认False
export HCCL_IF_BASE_PORT=30000 #设置HCCL端口
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

# 基础配置
NPUS_PER_NODE=8  #使用单节点的8卡NPU
MASTER_ADDR=localhost #以本节点ip地址为master_ip
MASTER_PORT=6015 #本节点端口号为6015
NNODES=1  #单机，即一台节点，多机即多节点
NODE_RANK=0  #单机RANK为0，多机为(0,NNODES-1)，不同节点不可重复
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES)) #最终使用的NPU数

# 根据实际情况配置权重保存、权重加载、词表、数据集路径
CKPT_LOAD_DIR="/apps/hw/outputs/qwen3/14b/mg_weights/qwen3_14b_mcore_tp2pp4/"  #权重加载路径，填入权重转换时保存的权重路径
CKPT_SAVE_DIR="/apps/hw/outputs/qwen3/14b/save_weights_sft/qwen3_14b_mcore_tp2pp4/"  #训练完成后的权重保存路径
DATA_PATH="/apps/hw/outputs/qwen3/finetune_dataset/alpaca"  #数据集路径，填入数据预处理时保存的数据路径，注意需要添加后缀
TOKENIZER_PATH="/apps/hw/weights/Qwen3-14B/" #词表路径，填入下载的开源权重词表路径

TP=2
PP=4
MBS=1
GBS=16
SEQ_LENGTH=4096
TRAIN_ITERS=1000 #训练1000步

...

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
```

微调脚本相关参数说明
- `DATA_PATH`：数据集路径。请注意实际数据预处理生成文件末尾会增加`_input_ids_document`等后缀，该参数填写到数据集的前缀即可。例如实际的数据集相对路径是`./finetune_dataset/alpaca/alpaca_packed_input_ids_document.bin`等，那么只需要填`./finetune_dataset/alpaca/alpaca`即可。
- `is-instruction-dataset`：用于指定微调过程中采用指令微调数据集，以确保模型依据特定指令数据进行微调。
- `variable-seq-lengths`：在不同的mini-batch间支持以动态的序列长度进行微调，默认padding到`8`的整数倍，可以通过`pad-to-multiple-of`参数来修改padding的倍数。假设微调时指定`--seq-length`序列长度为1024，开启`--variable-seq-lengths`后，序列长度会padding到真实数据长度的8整数倍。如下图所示：  ![variable_seq_lengths](qwen3-14B.assets/variable_seq_lengths-17529189902421.png)

* `--save-interval` 多少步存一次ckpt

如果是单机运行场景，只需要在一台机器上启动微调脚本：

```bash
bash examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh
```

训练日志如下：

![image-20250729174919503](qwen3-14B.assets/image-20250729174919503.png)

tensorboard监控微调loss下降结果如下：

![image-20250728181840558](qwen3-14B.assets/image-20250728181840558.png)

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

### 4.5 推理验证

完成微调后，需要进一步验证模型是否具备了预期的输出能力。我们提供了简单的模型生成脚本，只需要加载微调后的模型权重，便可观察模型在不同生成参数配置下的回复。

使用推理脚本  generate_qwen3_14b_ptd.sh

```
vim examples/mcore/qwen3/generate_qwen3_14b_ptd.sh
```

修改以下参数：

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1 #定义了任务流能够利用或映射到的硬件队列的数量。
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True #内存碎片优化开关，默认False
export HCCL_IF_BASE_PORT=30000 #设置HCCL端口
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

# please fill these path configurations
CHECKPOINT="/apps/hw/outputs/qwen3/14b/save_weights_sft" # 指向微调后权重的保存路径
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
```

运行推理脚本：

```bash
bash examples/mcore/qwen3/generate_qwen3_14b_ptd.sh
```

过程部分日志如下：

![image-20250730180314383](qwen3-14B.assets/image-20250730180314383.png)

### 4.6 评测集验证

#### 4.6.1 mmlu推理评估

使用评估脚本  generate_qwen3_14b_ptd.sh

```
vim examples/mcore/qwen3/evaluate_qwen3_14b_ptd.sh
```

修改以下参数：

```shell
# 初始化环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
# The number of parameters is not aligned
export CUDA_DEVICE_MAX_CONNECTIONS=1 #定义了任务流能够利用或映射到的硬件队列的数量。
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

# please fill these path configurations
TOKENIZER_PATH="/apps/hw/weights/Qwen3-14B/" # 指向模型tokenizer的路径
CHECKPOINT="/apps/hw/outputs/qwen3/14b/mg_weights/qwen3_14b_mcore_tp2pp4/"
DATA_PATH="/apps/hw/datasets/mmlu/data"
TASK="mmlu"

# 基础配置
NPUS_PER_NODE=8  #使用单节点的8卡NPU
MASTER_ADDR=localhost #以本节点ip地址为master_ip
MASTER_PORT=6015 #本节点端口号为6015
NNODES=1  #单机，即一台节点，多机即多节点
NODE_RANK=0  #单机RANK为0，多机为(0,NNODES-1)，不同节点不可重复
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES)) #最终使用的NPU数

TP=2
PP=4
SEQ_LENGTH=4096
```

评估参数说明

- **`--prompt-type`**：模型对话模板，确保评估输入结构匹配模型需求。
- **`--hf-chat-template`**：若模型具备内置 `chat_template` 属性，可添加该参数，使用内置模板。
- **`--eval-language`**：指定数据集语言，默认 `en`（英文），中文数据集应设置为 `zh`。

注意事项

- **对话模板选择**：根据模型和数据需求选择合适模板，确保微调和评估一致性。
- **语言匹配**：设定数据集语言以优化评估效果。
- **内置模板使用**：启用 `--hf-chat-template` 时简化输入格式，确保评估的准确性。

## 五、Lora微调

### 5.1 Alpaca数据集处理

Alpaca风格数据集Lora 微调处理和SFT微调数据集处理方式相同。

### 5.2 Lora微调训练
修改单机lora微调脚本 tune_qwen3_14b_4K_lora_ptd.sh

```
vim examples/mcore/qwen3/tune_qwen3_14b_4K_lora_ptd.sh
```

示例脚本内容如下：

```shell
# 初始化环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export HCCL_IF_BASE_PORT=30000 #设置HCCL端口
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

export HCCL_CONNECT_TIMEOUT=1800 #定义HCCL建立链接的超时时间。
export CUDA_DEVICE_MAX_CONNECTIONS=1  #定义了任务流能够利用或映射到的硬件队列的数量。

# 基础配置
NPUS_PER_NODE=8  #使用单节点的8卡NPU
MASTER_ADDR=localhost #以本节点ip地址为master_ip
MASTER_PORT=6015 #本节点端口号为6015
NNODES=1  #单机，即一台节点，多机即多节点
NODE_RANK=0  #单机RANK为0，多机为(0,NNODES-1)，不同节点不可重复
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES)) #最终使用的NPU数

# 根据实际情况配置权重保存、权重加载、词表、数据集路径
CKPT_LOAD_DIR="/apps/hw/outputs/qwen3/14b/mg_weights/qwen3_14b_mcore_tp2pp4/"  #权重加载路径，填入权重转换时保存的权重路径
CKPT_SAVE_DIR="/apps/hw/outputs/qwen3/14b/save_weights_lora/qwen3_14b_mcore_tp2pp4/"  #训练完成后的权重保存路径
DATA_PATH="/apps/hw/outputs/qwen3/finetune_dataset/alpaca"  #数据集路径，填入数据预处理时保存的数据路径，注意需要添加后缀
TOKENIZER_PATH="/apps/hw/weights/Qwen3-14B/" #词表路径，填入下载的开源权重词表路径4
TP=2
PP=4
SEQ_LENGTH=4096
TRAIN_ITERS=2000

...

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
```
训练日志如下：

![image-20250729181647243](qwen3-14B.assets/image-20250729181647243.png)

tensorboard监控微调loss下降结果如下：

![image-20250728182657139](qwen3-14B.assets/image-20250728182657139.png)

参数说明：

- **`--load`**:  若不指定该参数加载权重，模型会随机初始化权重。
- **`--prompt-type`**:  指定模型模板，使 base 模型在微调后具备更好的对话能力。
- **`--variable-seq-lengths`**:  支持动态序列长度微调，默认按 8 的倍数进行 padding，可以通过 `--pad-to-multiple-of` 参数修改 padding 的倍数。
- **`--lora-r`**:  LoRA rank，表示低秩矩阵的维度。较低的 rank 值模型在训练时会使用更少的参数更新，从而减少计算量和内存消耗。然而，过低的 rank 可能限制模型的表达能力。
- **`--lora-alpha`**:  控制 LoRA 权重对原始权重的影响比例, 数值越高则影响越大。一般保持 `α/r` 为 2。
- **`--lora-fusion`**:  是否启用<a href="https://gitee.com/ascend/MindSpeed-LLM/blob/2.1.0/docs/features/cc_lora.md">CCLoRA</a>算法，该算法通过计算通信掩盖提高性能。
- **`--lora-target-modules`**:  选择需要添加 LoRA 的模块。  *mcore 模型可选模块：* `linear_qkv`, `linear_proj`, `linear_fc1`, `linear_fc2`
- **`--lora-load`**:  加载 LoRA 权重断点继续训练或用于推理。在推理时需与 `--load` 参数配合使用，加载 `CKPT_SAVE_DIR` 路径下的 LoRA 权重。

### 5.3 Lora权重与Base权重的合并

基于lora微调训练完后，得到Lora权重与Base权重存在区别，无法直接用于续训，需要转换后才可使用，添加以下参数即可将训练好的 Lora 权重与 Base 权重融合,合并后转换为 Mcore 权重：

```bash
    --lora-load ${CHECKPOINT_LORA}  \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
```

修改合并脚本 ckpt_convert_qwen3_mcore2mcore_lora.sh

```
vim examples/mcore/qwen3/ckpt_convert_qwen3_mcore2mcore_lora.sh
```

将 Lora权重 合并转换为 Mcore 权重的示例参数：

```shell
# 生效 ascend-toolkit 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

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
    --load-dir /apps/hw/outputs/qwen3/14b/mg_weights/qwen3_14b_mcore_tp2pp4/ \
    --lora-load /apps/hw/outputs/qwen3/14b/save_weights_lora/qwen3_14b_mcore_tp2pp4/ \
    --save-dir /apps/hw/outputs/qwen3/14b/save_weights_lora/qwen3_14b_mcore_tp2pp4_merge/ \
    --model-type-hf qwen3
```

> 关键参数： --save-model-type mg

启动转换脚本：

```shell
bash examples/mcore/qwen3/ckpt_convert_qwen3_mcore2mcore_lora.sh
```

转换过程部分日志如下：

![image-20250730184219084](qwen3-14B.assets/image-20250730184219084.png)

### 5.4 lora权重合并后转换为HF权重

若希望将lora权重合并转换为 Hugging Face（HF）格式，修改合并脚本 ckpt_convert_qwen3_mcore2hf_lora.sh

```
vim examples/mcore/qwen3/ckpt_convert_qwen3_mcore2hf_lora.sh
```

将 Lora权重 合并转换为HF权重的示例参数：

```shell
# 生效 ascend-toolkit 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

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
    --load-dir /apps/hw/outputs/qwen3/14b/mg_weights/qwen3_14b_mcore_tp2pp4/ \
    --lora-load /apps/hw/outputs/qwen3/14b/save_weights_lora/qwen3_14b_mcore_tp2pp4/ \
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
bash examples/mcore/qwen3/ckpt_convert_qwen3_mcore2hf_lora.sh
```

转换过程部分日志如下：

![image-20250730184716011](qwen3-14B.assets/image-20250730184716011.png)

### 5.5 LoRA 推理验证

> 推理方法与SFT的推理方法相同

使用推理脚本  generate_qwen3_14b_ptd.sh

```
vim examples/mcore/qwen3/generate_qwen3_14b_ptd.sh
```

修改  `CHECKPOINT` 的权重路径为

修改以下参数：

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1 #定义了任务流能够利用或映射到的硬件队列的数量。
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True #内存碎片优化开关，默认False
export HCCL_IF_BASE_PORT=30000 #设置HCCL端口
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

# please fill these path configurations
CHECKPOINT="/apps/hw/outputs/qwen3/14b/save_weights_lora/qwen3_14b_mcore_tp2pp4_merge/" # 指向微调后权重的保存路径

...

TP=2
PP=4
SEQ_LENGTH=4096
```

然后运行推理脚本：

```bash
bash examples/mcore/qwen3/generate_qwen3_14b_ptd.sh
```

运行过程日志如下：



### 5.6 LoRA权重推理评估

> LoRA权重推理评估方法和SFT中mmlu推理评估方法相同，修改权重路径即可。

使用评估脚本  generate_qwen3_14b_ptd.sh

```
cp examples/mcore/qwen3/evaluate_qwen3_14b_ptd.sh midea_train/evaluate_qwen3_14b_lora_mmlu_ptd.sh
vim midea_train/evaluate_qwen3_14b_lora_mmlu_ptd.sh
```

修改以下参数：

```shell
export CUDA_DEVICE_MAX_CONNECTIONS=1 #定义了任务流能够利用或映射到的硬件队列的数量。

TOKENIZER_PATH="/apps/hw/weights/Qwen3-14B/" # 指向模型tokenizer的路径
CHECKPOINT="/apps/hw/outputs/qwen3/14b/save_weights_lora/qwen3_14b_mcore_tp2pp4_merge/"
DATA_PATH="/apps/hw/datasets/mmlu/data/test"
TASK="mmlu"

# 基础配置
NPUS_PER_NODE=8  #使用单节点的8卡NPU
MASTER_ADDR=localhost #以本节点ip地址为master_ip
MASTER_PORT=6015 #本节点端口号为6015
NNODES=1  #单机，即一台节点，多机即多节点
NODE_RANK=0  #单机RANK为0，多机为(0,NNODES-1)，不同节点不可重复
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES)) #最终使用的NPU数

TP=2
PP=4
SEQ_LENGTH=4096
```

运行脚本

```
bash midea_train/evaluate_qwen3_14b_lora_mmlu_ptd.sh
```

运行过程日志如下：

![image-20250730191330384](qwen3-14B.assets/image-20250730191330384.png)

评测结果如下：




## 六、附加说明

### 6.1.1 模型脚本环境变量说明

模型列表中脚本的常见环境变量说明具体如下：

| 环境变量名称                      | 环境变量描述                                                                    |
|-----------------------------|---------------------------------------------------------------------------|
| ASCEND_LAUNCH_BLOCKING      | 将Host日志输出到串口,0-关闭/1-开启。                                                   |
| ASCEND_SLOG_PRINT_TO_STDOUT | 设置默认日志级别,0-debug/1-info/2-warning/3-error。                                |
| HCCL_WHITELIST_DISABLE      | HCCL白名单开关,1-关闭/0-开启。                                                      |
| HCCL_CONNECT_TIMEOUT        | 设置HCCL超时时间，默认值为120。                                                       |
| CUDA_DEVICE_MAX_CONNECTIONS | 定义了任务流能够利用或映射到的硬件队列的数量。                                                   |
| TASK_QUEUE_ENABLE           | 用于控制开启task_queue算子下发队列优化的等级，0：关闭，1：开启Level 1优化，2：开启Level 2优化。             |
| COMBINED_ENABLE             | 设置combined标志。设置为0表示关闭此功能；设置为1表示开启，用于优化非连续两个算子组合类场景。                       |
| PYTORCH_NPU_ALLOC_CONF      | 内存碎片优化开关，默认是expandable_segments:False，使能时expandable_segments:True。        |
| ASCEND_RT_VISIBLE_DEVICES   | 指定哪些Device对当前进程可见，支持一次指定一个或多个Device ID。通过该环境变量，可实现不修改应用程序即可调整所用Device的功能。 |
| NPUS_PER_NODE               | 配置一个计算节点上使用的NPU数量。                                                        |
| HCCL_SOCKET_IFNAME          | 指定hccl socket通讯走的网卡配置。                                                    |
| GLOO_SOCKET_IFNAME          | 指定gloo socket通讯走的网卡配置。                                                    |
| HCCL_LOGIC_SUPERPOD_ID      | 指定当前设备的逻辑超节点ID，如果走ROCE，不同多机超节点ID不同，0-N。                                   |

### 6.1.2 训练参数说明

训练参数解析

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
| `--tensorboard-dir`               | 指定tensorboard的事件存储目录，如` ./tb `。查看训练loss曲线使用方法为 `tensorboard --logdir=./tb --port=6007` |      |

## 七、业务数据集调测

### 7.1 微调训练

#### 7.1.1 转换微调数据集

原始数据集目录： /apps/hw/datasets/midea_dataset/midea_datasets_sft

对`ShareGPT`格式数据数据集处理脚本 data_convert_qwen3_instruction.sh 中的配置进行修改：

```
cp examples/mcore/qwen3/data_convert_qwen3_instruction.sh examples/mcore/qwen3/data_convert_qwen3_instruction_midea_sharegpt10k.sh
vim examples/mcore/qwen3/data_convert_qwen3_instruction_midea_sharegpt10k.sh
```

```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

mkdir -p /apps/hw/outputs/qwen3/finetune_dataset_midea_sharegpt10k
python ./preprocess_data.py \
    --input  /apps/hw/datasets/midea_dataset/midea_datasets_sft/intent_train_10k.jsonl \
    --tokenizer-name-or-path /apps/hw/weights/Qwen3-14B/ \
    --output-prefix /apps/hw/outputs/qwen3/finetune_dataset_midea_sharegpt10k/openai \
    --handler-name SharegptStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 4 \
    --log-interval 1000 \
    --prompt-type qwen \
    --map-keys '{"messages":"messages", "tags":{"role_tag": "role","content_tag": "content","user_tag": "user","assistant_tag": "assistant","system_tag": "system"}}'
```

修改后，执行脚本

```
bash examples/mcore/qwen3/data_convert_qwen3_instruction_midea_sharegpt10k.sh
```

指令微调数据集处理结果如下：

```shell
openai_packed_attention_mask_document.bin
openai_packed_attention_mask_document.idx
openai_packed_input_ids_document.bin
openai_packed_input_ids_document.idx
openai_packed_labels_document.bin
openai_packed_labels_document.idx
```

#### 7.1.2 全参微调

##### 7.1.2.1 全参微调训练

修改单机微调脚本 tune_qwen3_14b_4K_full_ptd.sh 

```
cp examples/mcore/qwen3/tune_qwen3_14b_4K_full_ptd.sh  examples/mcore/qwen3/tune_qwen3_14b_4K_full_ptd_midea.sh 
vim examples/mcore/qwen3/tune_qwen3_14b_4K_full_ptd_midea.sh 
```

示例脚本内容如下：


```shell
export CUDA_DEVICE_MAX_CONNECTIONS=1 #定义了任务流能够利用或映射到的硬件队列的数量。
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True #内存碎片优化开关，默认False
export HCCL_NPU_SOCKET_PORT_RANGE="auto" 
export HCCL_PORT_RANGE="7015-7200"
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

# 基础配置
NPUS_PER_NODE=8  #使用单节点的8卡NPU
MASTER_ADDR=localhost #以本节点ip地址为master_ip
MASTER_PORT=6015 #本节点端口号为6015
NNODES=1  #单机，即一台节点，多机即多节点
NODE_RANK=0  #单机RANK为0，多机为(0,NNODES-1)，不同节点不可重复
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES)) #最终使用的NPU数

# 根据实际情况配置权重保存、权重加载、词表、数据集路径
CKPT_LOAD_DIR="/apps/hw/outputs/qwen3/14b/mg_weights/qwen3_14b_mcore_tp2pp4/"  #权重加载路径，填入权重转换时保存的权重路径
CKPT_SAVE_DIR="/apps/hw/outputs/qwen3/14b/save_weights_sft/midea_qwen3_14b_mcore_tp2pp4/"  #训练完成后的权重保存路径
DATA_PATH="/apps/hw/outputs/qwen3/finetune_dataset_midea_sharegpt10k/openai"  #数据集路径，填入数据预处理时保存的数据路径，注意需要添加后缀
TOKENIZER_PATH="/apps/hw/weights/Qwen3-14B/" #词表路径，填入下载的开源权重词表路径

TP=2
PP=4
MBS=1
GBS=16
SEQ_LENGTH=4096
TRAIN_ITERS=1000 #训练1000步

...

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
```

微调脚本相关参数说明

- `DATA_PATH`：数据集路径。请注意实际数据预处理生成文件末尾会增加`_input_ids_document`等后缀，该参数填写到数据集的前缀即可。例如实际的数据集相对路径是`./finetune_dataset/alpaca/alpaca_packed_input_ids_document.bin`等，那么只需要填`./finetune_dataset/alpaca/alpaca`即可。
- `is-instruction-dataset`：用于指定微调过程中采用指令微调数据集，以确保模型依据特定指令数据进行微调。
- `variable-seq-lengths`：在不同的mini-batch间支持以动态的序列长度进行微调，默认padding到`8`的整数倍，可以通过`pad-to-multiple-of`参数来修改padding的倍数。假设微调时指定`--seq-length`序列长度为1024，开启`--variable-seq-lengths`后，序列长度会padding到真实数据长度的8整数倍。如下图所示：  ![variable_seq_lengths](qwen3-14B.assets/variable_seq_lengths-17529189902421.png)

* `--save-interval` 多少步存一次ckpt

如果是单机运行场景，只需要在一台机器上启动微调脚本：

```bash
bash examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh
```

tensorboard监控微调loss下降结果如下：

![image-20250728181723712](qwen3-14B.assets/image-20250728181723712.png)

##### 7.1.2.2 全参微调推理验证

完成微调后，需要进一步验证模型是否具备了预期的输出能力。只需要加载微调后的模型权重，便可观察模型在不同生成参数配置下的回复。

修改推理脚本  generate_qwen3_14b_ptd.sh

```
cp examples/mcore/qwen3/generate_qwen3_14b_ptd.sh examples/mcore/qwen3/midea_generate_qwen3_14b_ptd.sh
vim examples/mcore/qwen3/midea_generate_qwen3_14b_ptd.sh
```

修改以下参数：

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1 #定义了任务流能够利用或映射到的硬件队列的数量。
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True #内存碎片优化开关，默认False
export HCCL_IF_BASE_PORT=30000 #设置HCCL端口
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

# please fill these path configurations
CHECKPOINT="/apps/hw/outputs/qwen3/14b/save_weights_sft/midea_qwen3_14b_mcore_tp2pp4/" # 指向微调后权重的保存路径
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
```

然后运行推理脚本：

```bash
bash examples/mcore/qwen3/midea_generate_qwen3_14b_ptd.sh
```

![image-20250725170604384](qwen3-14B.assets/image-20250725170604384.png)

##### 7.1.2.3 全参微调评测

修改qwen3_14b_full_ptd_eval_midea_dataset.sh脚本：

```
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

CHECKPOINT="/apps/hw/outputs/qwen3/14b/save_weights_sft/midea_qwen3_14b_mcore_tp2pp4/" # 指向微调后权重的保存路径
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
```

然后运行评测脚本：

```bash
bash examples/mcore/qwen3/qwen3_14b_full_ptd_eval_midea_dataset.sh
```

评测结果如下，准确率0.941

![image-20250728175337223](qwen3-14B.assets/image-20250728175337223.png)

#### 7.1.5 lora微调训练

##### 7.1.5.1 启动lora微调

修改单机lora微调脚本 tune_qwen3_14b_4K_lora_ptd.sh

```
cp examples/mcore/qwen3/tune_qwen3_14b_4K_lora_ptd.sh  examples/mcore/qwen3/tune_qwen3_14b_4K_lora_ptd_midea.sh 
vim examples/mcore/qwen3/tune_qwen3_14b_4K_lora_ptd_midea.sh 
```

示例脚本内容如下：

```shell
# 初始化环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export HCCL_IF_BASE_PORT=30000 #设置HCCL端口
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

export HCCL_CONNECT_TIMEOUT=1800 #定义HCCL建立链接的超时时间。
export CUDA_DEVICE_MAX_CONNECTIONS=1  #定义了任务流能够利用或映射到的硬件队列的数量。

# 基础配置
NPUS_PER_NODE=8  #使用单节点的8卡NPU
MASTER_ADDR=localhost #以本节点ip地址为master_ip
MASTER_PORT=6015 #本节点端口号为6015
NNODES=1  #单机，即一台节点，多机即多节点
NODE_RANK=0  #单机RANK为0，多机为(0,NNODES-1)，不同节点不可重复
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES)) #最终使用的NPU数

# 根据实际情况配置权重保存、权重加载、词表、数据集路径
CKPT_LOAD_DIR="/apps/hw/outputs/qwen3/14b/mg_weights/qwen3_14b_mcore_tp2pp4/"  #权重加载路径，填入权重转换时保存的权重路径
CKPT_SAVE_DIR="/apps/hw/outputs/qwen3/14b/save_weights_lora/midea_qwen3_14b_mcore_tp2pp4/"  #训练完成后的权重保存路径
DATA_PATH="/apps/hw/outputs/qwen3/finetune_dataset_midea_sharegpt10k/openai"  #数据集路径，填入数据预处理时保存的数据路径，注意需要添加后缀
TOKENIZER_PATH="/apps/hw/weights/Qwen3-14B/" #词表路径，填入下载的开源权重词表路径4
TP=2
PP=4
SEQ_LENGTH=4096
TRAIN_ITERS=2000

...

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
#    --variable-seq-lengths \ #不设置动态长度，便于对精度
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-fusion \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
"
```

参数说明：

- **`--load`**:  若不指定该参数加载权重，模型会随机初始化权重。
- **`--prompt-type`**:  指定模型模板，使 base 模型在微调后具备更好的对话能力。
- **`--variable-seq-lengths`**:  支持动态序列长度微调，默认按 8 的倍数进行 padding，可以通过 `--pad-to-multiple-of` 参数修改 padding 的倍数。
- **`--lora-r`**:  LoRA rank，表示低秩矩阵的维度。较低的 rank 值模型在训练时会使用更少的参数更新，从而减少计算量和内存消耗。然而，过低的 rank 可能限制模型的表达能力。
- **`--lora-alpha`**:  控制 LoRA 权重对原始权重的影响比例, 数值越高则影响越大。一般保持 `α/r` 为 2。
- **`--lora-fusion`**:  是否启用<a href="https://gitee.com/ascend/MindSpeed-LLM/blob/2.1.0/docs/features/cc_lora.md">CCLoRA</a>算法，该算法通过计算通信掩盖提高性能。
- **`--lora-target-modules`**:  选择需要添加 LoRA 的模块。  *mcore 模型可选模块：* `linear_qkv`, `linear_proj`, `linear_fc1`, `linear_fc2`
- **`--lora-load`**:  加载 LoRA 权重断点继续训练或用于推理。在推理时需与 `--load` 参数配合使用，加载 `CKPT_SAVE_DIR` 路径下的 LoRA 权重。

tensorboard监控微调loss下降结果如下：

![image-20250728181657240](qwen3-14B.assets/image-20250728181657240.png)

##### 7.1.5.2 lora权重合并

修改合并脚本 ckpt_convert_qwen3_mcore2mcore_lora.sh

```
cp examples/mcore/qwen3/ckpt_convert_qwen3_mcore2mcore_lora.sh examples/mcore/qwen3/midea_ckpt_convert_qwen3_mcore2mcore_lora.sh
vim examples/mcore/qwen3/midea_ckpt_convert_qwen3_mcore2mcore_lora.sh
```

将 Lora权重 合并转换为 Mcore 权重的示例参数：

```shell
# 生效 ascend-toolkit 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

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
    --load-dir /apps/hw/outputs/qwen3/14b/mg_weights/qwen3_14b_mcore_tp2pp4/ \
    --lora-load /apps/hw/outputs/qwen3/14b/save_weights_lora/midea_qwen3_14b_mcore_tp2pp4/ \
    --save-dir /apps/hw/outputs/qwen3/14b/save_weights_lora/midea_qwen3_14b_mcore_tp2pp4_merge/ \
    --model-type-hf qwen3
```

启动转换脚本：

```shell
bash examples/mcore/qwen3/midea_ckpt_convert_qwen3_mcore2mcore_lora.sh
```

##### 7.1.5.3 lora微调推理验证

修改推理脚本  generate_qwen3_14b_ptd.sh

```
cp examples/mcore/qwen3/generate_qwen3_14b_ptd.sh examples/mcore/qwen3/midea_generate_qwen3_14b_lora_ptd.sh
vim examples/mcore/qwen3/midea_generate_qwen3_14b_lora_ptd.sh
```

修改以下参数：

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1 #定义了任务流能够利用或映射到的硬件队列的数量。
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True #内存碎片优化开关，默认False
export HCCL_IF_BASE_PORT=30000 #设置HCCL端口
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

# please fill these path configurations
CHECKPOINT="/apps/hw/outputs/qwen3/14b/save_weights_lora/midea_qwen3_14b_mcore_tp2pp4_merge/" # 指向微调后权重的保存路径
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
```

然后运行推理脚本：

```bash
bash examples/mcore/qwen3/midea_generate_qwen3_14b_lora_ptd.sh
```

![image-20250725181626268](qwen3-14B.assets/image-20250725181626268.png)

##### 7.1.5.4 lora微调评测

修改qwen3_14b_lora_ptd_eval_midea_dataset.sh脚本：

```
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

CHECKPOINT="/apps/hw/outputs/qwen3/14b/save_weights_sft/midea_qwen3_14b_mcore_tp2pp4/" # 指向微调后权重的保存路径
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
```

> tips: 评测脚本和全参微调评测脚本一致，只需要修改权重路径

然后运行评测脚本：

```bash
bash examples/mcore/qwen3/qwen3_14b_lora_ptd_eval_midea_dataset.sh
```

评测结果如下，准确率0.93

![image-20250728175212530](qwen3-14B.assets/image-20250728175212530.png)

### 7.2 权重转换

昇腾MindSpeed-LLM要求模型权重采用Megatron-LM格式，在这里我们将原始HuggingFace权重格式转换为Megatron-Mcore格式。
详见[hf2mg权重转换](./pytorch/solutions/checkpoint_convert.md#21-huggingface权重转换到megatron-lm格式)

使用官方提供的转换脚本，获取对应切分的mg权重。

```bash
cp examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh examples/mcore/qwen3/midea_ckpt_convert_qwen3_hf2mcore.sh
vim examples/mcore/qwen3/midea_ckpt_convert_qwen3_hf2mcore.sh
```

如下 为调整后的hf2mcore权重转换示例脚本

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
# 生效 ascend-toolkit 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

# 设置需要的权重转换参数
python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 2 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --load-dir /apps/hw/weights/Qwen3-14B/ \
    --save-dir /apps/hw/outputs/qwen3/14b/mg_weights/qwen3_14b_mcore_tp4pp2/ \
    --tokenizer-model /apps/hw/weights/Qwen3-14B/tokenizer.json \
    --params-dtype bf16 \
    --model-type-hf qwen3
```

修改后，执行转换脚本

```
bash examples/mcore/qwen3/midea_ckpt_convert_qwen3_hf2mcore.sh
```

参数解析

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

### 7.3 微调训练（对齐精度）

#### 7.3.1 转换微调数据集

原始数据集目录： /apps/hw/datasets/midea_dataset/midea_datasets_sft

对`ShareGPT`格式数据数据集处理脚本 data_convert_qwen3_instruction.sh 中的配置进行修改：

```
cp examples/mcore/qwen3/data_convert_qwen3_instruction.sh examples/mcore/qwen3/data_convert_qwen3_instruction_midea_sharegpt10k.sh
vim examples/mcore/qwen3/data_convert_qwen3_instruction_midea_sharegpt10k.sh
```

```shell
# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

mkdir -p /apps/hw/outputs/qwen3/finetune_dataset_midea_sharegpt10k
python ./preprocess_data.py \
    --input  /apps/hw/datasets/midea_dataset/midea_datasets_sft/intent_train_10k.jsonl \
    --tokenizer-name-or-path /apps/hw/weights/Qwen3-14B/ \
    --output-prefix /apps/hw/outputs/qwen3/finetune_dataset_midea_sharegpt10k/openai \
    --handler-name SharegptStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 4 \
    --log-interval 1000 \
    --prompt-type qwen \
    --map-keys '{"messages":"messages", "tags":{"role_tag": "role","content_tag": "content","user_tag": "user","assistant_tag": "assistant","system_tag": "system"}}'
```

修改后，执行脚本

```
bash examples/mcore/qwen3/data_convert_qwen3_instruction_midea_sharegpt10k.sh
```

指令微调数据集处理结果如下：

```shell
openai_packed_attention_mask_document.bin
openai_packed_attention_mask_document.idx
openai_packed_input_ids_document.bin
openai_packed_input_ids_document.idx
openai_packed_labels_document.bin
openai_packed_labels_document.idx
```

#### 7.3.2 全参微调

##### 7.2.2.1 全参微调训练

修改单机微调脚本 tune_qwen3_14b_4K_full_ptd.sh 

```
cp examples/mcore/qwen3/tune_qwen3_14b_4K_full_ptd.sh  examples/mcore/qwen3/tune_qwen3_14b_4K_full_ptd_midea.sh 
vim examples/mcore/qwen3/tune_qwen3_14b_4K_full_ptd_midea.sh 
```

示例脚本内容如下：


```shell
export CUDA_DEVICE_MAX_CONNECTIONS=1 #定义了任务流能够利用或映射到的硬件队列的数量。
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True #内存碎片优化开关，默认False
export HCCL_NPU_SOCKET_PORT_RANGE="auto" 
export HCCL_PORT_RANGE="7015-7200"
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

# 基础配置
NPUS_PER_NODE=8  #使用单节点的8卡NPU
MASTER_ADDR=localhost #以本节点ip地址为master_ip
MASTER_PORT=6015 #本节点端口号为6015
NNODES=1  #单机，即一台节点，多机即多节点
NODE_RANK=0  #单机RANK为0，多机为(0,NNODES-1)，不同节点不可重复
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES)) #最终使用的NPU数

# 根据实际情况配置权重保存、权重加载、词表、数据集路径
CKPT_LOAD_DIR="/apps/hw/outputs/qwen3/14b/mg_weights/qwen3_14b_mcore_tp4pp2/"  #权重加载路径，填入权重转换时保存的权重路径
CKPT_SAVE_DIR="/apps/hw/outputs/qwen3/14b/save_weights_sft/midea_qwen3_14b_mcore_tp4pp2/"  #训练完成后的权重保存路径
DATA_PATH="/apps/hw/outputs/qwen3/finetune_dataset_midea_sharegpt10k/openai"  #数据集路径，填入数据预处理时保存的数据路径，注意需要添加后缀
TOKENIZER_PATH="/apps/hw/weights/Qwen3-14B/" #词表路径，填入下载的开源权重词表路径

TP=4
PP=2
MBS=32
GBS=256
SEQ_LENGTH=4096
TRAIN_ITERS=1000 #训练1000步

...

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
```

> 去掉--variable-seq-lengths，GBS改为256，MBS=32，TP-4，PP=2

微调脚本相关参数说明

- `DATA_PATH`：数据集路径。请注意实际数据预处理生成文件末尾会增加`_input_ids_document`等后缀，该参数填写到数据集的前缀即可。例如实际的数据集相对路径是`./finetune_dataset/alpaca/alpaca_packed_input_ids_document.bin`等，那么只需要填`./finetune_dataset/alpaca/alpaca`即可。
- `is-instruction-dataset`：用于指定微调过程中采用指令微调数据集，以确保模型依据特定指令数据进行微调。
- `variable-seq-lengths`：在不同的mini-batch间支持以动态的序列长度进行微调，默认padding到`8`的整数倍，可以通过`pad-to-multiple-of`参数来修改padding的倍数。假设微调时指定`--seq-length`序列长度为1024，开启`--variable-seq-lengths`后，序列长度会padding到真实数据长度的8整数倍。如下图所示：  ![variable_seq_lengths](qwen3-14B.assets/variable_seq_lengths-17529189902421.png)

* `--save-interval` 多少步存一次ckpt

如果是单机运行场景，只需要在一台机器上启动微调脚本：

```bash
bash examples/mcore/qwen3/tune_qwen3_8b_4K_full_ptd.sh
```

tensorboard监控微调loss下降结果如下：



##### 7.2.2.2 全参微调推理验证

完成微调后，需要进一步验证模型是否具备了预期的输出能力。只需要加载微调后的模型权重，便可观察模型在不同生成参数配置下的回复。

修改推理脚本  generate_qwen3_14b_ptd.sh

```
cp examples/mcore/qwen3/generate_qwen3_14b_ptd.sh examples/mcore/qwen3/midea_generate_qwen3_14b_ptd.sh
vim examples/mcore/qwen3/midea_generate_qwen3_14b_ptd.sh
```

修改以下参数：

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1 #定义了任务流能够利用或映射到的硬件队列的数量。
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True #内存碎片优化开关，默认False
export HCCL_IF_BASE_PORT=30000 #设置HCCL端口
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

# please fill these path configurations
CHECKPOINT="/apps/hw/outputs/qwen3/14b/save_weights_sft/midea_qwen3_14b_mcore_tp2pp4/" # 指向微调后权重的保存路径
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
```

然后运行推理脚本：

```bash
bash examples/mcore/qwen3/midea_generate_qwen3_14b_ptd.sh
```



##### 7.2.2.3 全参微调评测

修改qwen3_14b_full_ptd_eval_midea_dataset.sh脚本：

```
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

CHECKPOINT="/apps/hw/outputs/qwen3/14b/save_weights_sft/midea_qwen3_14b_mcore_tp2pp4/" # 指向微调后权重的保存路径
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
```

然后运行评测脚本：

```bash
bash examples/mcore/qwen3/qwen3_14b_full_ptd_eval_midea_dataset.sh
```

评测结果如下，准确率



#### 7.3.5 lora微调训练

##### 7.2.5.1 启动lora微调

修改单机lora微调脚本 tune_qwen3_14b_4K_lora_ptd.sh

```
cp examples/mcore/qwen3/tune_qwen3_14b_4K_lora_ptd.sh  examples/mcore/qwen3/tune_qwen3_14b_4K_lora_ptd_midea.sh 
vim examples/mcore/qwen3/tune_qwen3_14b_4K_lora_ptd_midea.sh 
```

示例脚本内容如下：

```shell
# 初始化环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export HCCL_IF_BASE_PORT=30000 #设置HCCL端口
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

export HCCL_CONNECT_TIMEOUT=1800 #定义HCCL建立链接的超时时间。
export CUDA_DEVICE_MAX_CONNECTIONS=1  #定义了任务流能够利用或映射到的硬件队列的数量。

# 基础配置
NPUS_PER_NODE=8  #使用单节点的8卡NPU
MASTER_ADDR=localhost #以本节点ip地址为master_ip
MASTER_PORT=6015 #本节点端口号为6015
NNODES=1  #单机，即一台节点，多机即多节点
NODE_RANK=0  #单机RANK为0，多机为(0,NNODES-1)，不同节点不可重复
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES)) #最终使用的NPU数

# 根据实际情况配置权重保存、权重加载、词表、数据集路径
CKPT_LOAD_DIR="/apps/hw/outputs/qwen3/14b/mg_weights/qwen3_14b_mcore_tp2pp4/"  #权重加载路径，填入权重转换时保存的权重路径
CKPT_SAVE_DIR="/apps/hw/outputs/qwen3/14b/save_weights_lora/midea_qwen3_14b_mcore_tp2pp4/"  #训练完成后的权重保存路径
DATA_PATH="/apps/hw/outputs/qwen3/finetune_dataset_midea_sharegpt10k/openai"  #数据集路径，填入数据预处理时保存的数据路径，注意需要添加后缀
TOKENIZER_PATH="/apps/hw/weights/Qwen3-14B/" #词表路径，填入下载的开源权重词表路径4
TP=2
PP=4
SEQ_LENGTH=4096
TRAIN_ITERS=2000

...

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
#    --variable-seq-lengths \ #不设置动态长度，便于对精度
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-fusion \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
"
```

参数说明：

- **`--load`**:  若不指定该参数加载权重，模型会随机初始化权重。
- **`--prompt-type`**:  指定模型模板，使 base 模型在微调后具备更好的对话能力。
- **`--variable-seq-lengths`**:  支持动态序列长度微调，默认按 8 的倍数进行 padding，可以通过 `--pad-to-multiple-of` 参数修改 padding 的倍数。
- **`--lora-r`**:  LoRA rank，表示低秩矩阵的维度。较低的 rank 值模型在训练时会使用更少的参数更新，从而减少计算量和内存消耗。然而，过低的 rank 可能限制模型的表达能力。
- **`--lora-alpha`**:  控制 LoRA 权重对原始权重的影响比例, 数值越高则影响越大。一般保持 `α/r` 为 2。
- **`--lora-fusion`**:  是否启用<a href="https://gitee.com/ascend/MindSpeed-LLM/blob/2.1.0/docs/features/cc_lora.md">CCLoRA</a>算法，该算法通过计算通信掩盖提高性能。
- **`--lora-target-modules`**:  选择需要添加 LoRA 的模块。  *mcore 模型可选模块：* `linear_qkv`, `linear_proj`, `linear_fc1`, `linear_fc2`
- **`--lora-load`**:  加载 LoRA 权重断点继续训练或用于推理。在推理时需与 `--load` 参数配合使用，加载 `CKPT_SAVE_DIR` 路径下的 LoRA 权重。

tensorboard监控微调loss下降结果如下：



##### 7.2.5.2 lora权重合并

修改合并脚本 ckpt_convert_qwen3_mcore2mcore_lora.sh

```
cp examples/mcore/qwen3/ckpt_convert_qwen3_mcore2mcore_lora.sh examples/mcore/qwen3/midea_ckpt_convert_qwen3_mcore2mcore_lora.sh
vim examples/mcore/qwen3/midea_ckpt_convert_qwen3_mcore2mcore_lora.sh
```

将 Lora权重 合并转换为 Mcore 权重的示例参数：

```shell
# 生效 ascend-toolkit 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

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
    --load-dir /apps/hw/outputs/qwen3/14b/mg_weights/qwen3_14b_mcore_tp2pp4/ \
    --lora-load /apps/hw/outputs/qwen3/14b/save_weights_lora/midea_qwen3_14b_mcore_tp2pp4/ \
    --save-dir /apps/hw/outputs/qwen3/14b/save_weights_lora/midea_qwen3_14b_mcore_tp2pp4_merge/ \
    --model-type-hf qwen3
```

启动转换脚本：

```shell
bash examples/mcore/qwen3/midea_ckpt_convert_qwen3_mcore2mcore_lora.sh
```

##### 7.2.5.3 lora微调推理验证

修改推理脚本  generate_qwen3_14b_ptd.sh

```
cp examples/mcore/qwen3/generate_qwen3_14b_ptd.sh examples/mcore/qwen3/midea_generate_qwen3_14b_lora_ptd.sh
vim examples/mcore/qwen3/midea_generate_qwen3_14b_lora_ptd.sh
```

修改以下参数：

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1 #定义了任务流能够利用或映射到的硬件队列的数量。
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True #内存碎片优化开关，默认False
export HCCL_IF_BASE_PORT=30000 #设置HCCL端口
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

# please fill these path configurations
CHECKPOINT="/apps/hw/outputs/qwen3/14b/save_weights_lora/midea_qwen3_14b_mcore_tp2pp4_merge/" # 指向微调后权重的保存路径
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
```

然后运行推理脚本：

```bash
bash examples/mcore/qwen3/midea_generate_qwen3_14b_lora_ptd.sh
```



##### 7.2.5.4 lora微调评测

修改qwen3_14b_lora_ptd_eval_midea_dataset.sh脚本：

```
# 解决libgomp在arm平台上的一个bug:cannot allocate memory in static TLS block
export LD_PRELOAD=$LD_PRELOAD:/root/miniconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

CHECKPOINT="/apps/hw/outputs/qwen3/14b/save_weights_sft/midea_qwen3_14b_mcore_tp2pp4/" # 指向微调后权重的保存路径
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
```

> tips: 评测脚本和全参微调评测脚本一致，只需要修改权重路径

然后运行评测脚本：

```bash
bash examples/mcore/qwen3/qwen3_14b_lora_ptd_eval_midea_dataset.sh
```

评测结果如下，准确率

