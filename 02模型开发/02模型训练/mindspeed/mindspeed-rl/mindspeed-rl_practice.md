# MindSpeed-RL安装

## 获取CANN镜像与拉起容器
```
docker pull --platform=arm64 swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.1.rc1-910b-ubuntu22.04-py3.10
```

创建容器
```bash
docker stop llm_rl && docker rm llm_rl
docker run -dit --ipc=host --network host --name 'llm_rl' --privileged \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/sbin/:/usr/local/sbin/  \
    -v /home/data/pae101:/home/data/pae101  \
    swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.1.rc1-910b-ubuntu22.04-py3.10 \
    bash
```

登录容器
```bash
docker exec -it llm_rl bash                           
npu-smi info
```

## 安装环境

### PTA安装

```shell
# 安装torch和torch_npu
wget https://download.pytorch.org/whl/cpu/torch-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

pip install torch-2.5.1-cp310-cp310-*.whl
pip install torch_npu-2.5.1.*.manylinux2014_aarch64.whl

# apex for Ascend 构建参考 https://gitee.com/ascend/apex
pip install --upgrade apex-0.1+ascend-*.whl
```

[验证 torch_npu 安装是否成功](https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/configandinstg/instg/insg_0001.html)
```
python3 -c "import torch;import torch_npu;print(torch_npu.npu.is_available())"
```

### vllm及相关依赖安装：
> 注：环境中需要安装git，因为vllm的安装过程依赖git
> 安装包在: /usr/local/python3.10.17/lib/python3.10/site-packages
```shell
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout -b v0.7.3
pip install -r requirements-build.txt
VLLM_TARGET_DEVICE=empty pip install .
```

> 如果依赖安装较慢，换源
> pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
> pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
> pip config set global.extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple/
> 
> ~/.config/pip/pip.conf文件内容如下
> [global]
> index-url = https://pypi.tuna.tsinghua.edu.cn/simple
> extra-index-url = https://mirrors.aliyun.com/pypi/simple/
### vllm_ascend安装
```shell
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout -b v0.7.3-dev
git checkout 0713836e95fe993feefe334945b5b273e4add1f1
pip install -e .
```

### 高性能内存库 jemalloc 安装
为了确保 Ray 进程能够正常回收内存，需要安装并使能 jemalloc 库进行内存管理。
#### Ubuntu 操作系统
通过操作系统源安装jemalloc（注意： 要求ubuntu版本>=20.04）：
```shell
sudo apt install libjemalloc2
```
在启动任务前执行如下命令通过环境变量导入jemalloc：
```shell
# arm64架构
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2 $LD_PRELOAD
# x86_64架构
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 $LD_PRELOAD
```

#### OpenEuler 操作系统

执行如下命令重操作系统源安装jemalloc
```shell
yum install jemalloc
```
如果上述方法无法正常安装，可以通过源码编译安装
前往jamalloc官网下载最新稳定版本，官网地址:https://github.com/jemalloc/jemalloc/releases/
```shell
tar -xvf jemalloc-{version}.tar.bz2
cd jemalloc-{version}
./configure --prefix=/usr/local
make
make install
```
在启动任务前执行如下命令通过环境变量导入jemalloc：
```shell
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2 $LD_PRELOAD
```

> 如以上安装过程出现错误，可以通过提出issue获得更多解决建议。

## 准备源码
```shell
git clone https://gitee.com/ascend/MindSpeed-RL.git 

git clone https://gitee.com/ascend/MindSpeed.git 
cd MindSpeed
git checkout 0dfa0035ec54d9a74b2f6ee2867367df897299df  # 参考MindSpeed-LLM依赖版本
pip install -r requirements.txt 
cp -r mindspeed ../MindSpeed-RL/
cd ..

git clone https://github.com/NVIDIA/Megatron-LM.git  # Megatron从github下载，请确保网络能访问
cd Megatron-LM
git checkout -b core_r0.8.0
cp -r megatron ../MindSpeed-RL/
cd ..

git clone https://gitee.com/ascend/MindSpeed-LLM.git
cd MindSpeed-LLM
git checkout 421ef7bcb83fb31844a1efb688cde71705c0526e
cp -r mindspeed_llm ../MindSpeed-RL/
cd ..

cd ./MindSpeed-RL
pip install -r requirements.txt
pip install antlr4-python3-runtime==4.7.2 --no-deps 
```

## 保存Docker镜像

保存镜像

```
docker commit -m "mindspeed-rl env" -a "lpf"  llm_rl mindspeed-rl:0618
```

镜像save为镜像.tar

```
docker save -o mindspeed-rl_0618.tar mindspeed-rl:0618
```

# 后训练方法 Ray GRPO

## 数据预处理
配置好环境后，需要对数据集进行预处理。

以 [**DeepScaler**](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset/tree/main) 为例。

```bash
# 读取deepscaler数据集
mkdir dataset
cd dataset/
wget https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset/resolve/main/deepscaler.json --no-check
cd ..
```

数据预处理的yaml配置文件放置于configs/datasets文件夹下，通过以下命令进行数据集预处理：
[示例yaml配置文件](../../configs/datasets/deepscaler.yaml)
```bash
# 读取configs/datasets/deepscaler.yaml文件 
bash examples/data/preprocess_data.sh
```

数据集处理配置可以根据需求自行配置，以下是数据集处理的yaml文件中基础参数的介绍：
* `input`：数据集的路径，需指定具体文件，例如/datasets/deepscaler.json
* `tokenizer_type`：指定分词器的类型，例如 HuggingFaceTokenizer 使用 Hugging Face 库提供的分词器来对文本进行分词处理;
* `tokenizer_name_or_path`：指定分词器的名称或路径;
* `output_prefix`：输出结果的前缀路径，例如 /datasets/data;
* `workers`：设置处理数据时使用的 worker 数;
* `prompt_type`: 用于指定对话模板，能够让 base 模型微调后能具备更好的对话能力，`prompt-type` 的可选项可以在 `configs/model/templates.json` 文件内查看;
* `log_interval`：设置日志记录的间隔，每处理多少条数据时记录一次日志，用于监控数据处理的进度和状态;
* `handler_name`：指定处理数据的处理器名称；
* `seq_length`：设置数据预处理最大序列长度，超过了会过滤掉;

## 模型权重转换

根据 GRPO 算法要求，Actor 和 Reference 模型应该使用 SFT 微调后的模型进行初始化，Reward 模型应该使用规则奖励。GRPO 算法模型权重均使用 Megatron-mcore 格式，其他格式的权重需要进行模型权重转换。

接下来，以 Qwen25-7B 模型的权重转换脚本为参考，相应的权重转换步骤如下:

### 获取权重文件
权重文件可以从 Huggingface 网站上获取，可以根据模型的使用场景灵活选择，在这里以
[Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)  为参考。
### hf 转 mcore
在训练前，需要将 Hugging Face 权重转换成Mcore格式。

注：这里会调用到 mindspeed_llm 仓，进行权重转换时注意按照安装手册中的环境准备步骤，将 mindspeed_llm 放入 MindSpeed-RL 目录下。

脚本启动命令可以用bash启动，可根据真实情况配置脚本，[示例脚本](../../examples/ckpt/ckpt_convert_qwen25_hf2mcore.sh)启动命令和配置参数如下：
```bash
# 路径按照真实情况配置
bash examples/ckpt/ckpt_convert_qwen25_hf2mcore.sh
```
配置参数介绍
* `use-mcore-models`：启用 MCore 模型；
* `model-type`：指定模型类型，如 GPT;
* `load-model-type`：指定加载模型的类型，如 hf（Hugging Face）;
* `save-model-type`：指定保存模型的类型，如 mg;
* `target-tensor-parallel-size`：设置目标张量并行大小；
* `target-pipeline-parallel-size`：设置目标流水线并行大小；
* `add-qkv-bias`：是否进行 QKV 偏置；
* `load-dir`：加载 Hugging Face 权重的路径；
* `save-dir`：保存转换后权重的路径；
* `tokenizer-model`：分词器模型文件的路径；
* `model-type-hf`：指定 Hugging Face 模型类型，如 llama2;
* `params-dtype`：指定参数的数据类型，如 bf16。

### mcore 转 hf（可选）
训练结束后，如果需要将生成的mcore格式权重转换回 Hugging Face 格式，可以参照以下[示例脚本](../../examples/ckpt/ckpt_convert_qwen25_mcore2hf.sh)命令及脚本参数：

```bash
# 路径按照真实情况配置
bash examples/ckpt/ckpt_convert_qwen25_mcore2hf.sh
```
配置参数介绍

这里的参数与上文一致，注意以下几个事项即可：
1. 权重转换转回 Hugging Face 格式时，tp 和 pp 配置需配置为1；
2. load-model-type 参数配置为 mg，save-model-type 参数配置为 hf ;
3. save-dir 路径需要填入原始 HF 模型路径，新权重会存于 HF 原始权重文件下的 mg2hg 目录下，如/qwen2.5_7b_hf/mg2hg/
## 启动训练

以 Qwen25 7B 模型为例,在启动训练之前，需要修改[ 启动脚本 ](../../examples/grpo/grpo_trainer_qwen25_7b.sh)的配置：
1. 修改 DEFAULT_YAML 为指定的 yaml，目前已支持的配置文件放置在 configs / 文件夹下，具体参数说明可见 [配置文件参数介绍](../features/grpo_yaml.md)；
2. 根据使用机器的情况，修改 NNODES 、NPUS_PER_NODE 配置， 例如单机 A3 可设置 NNODES 为 1 、NPUS_PER_NODE 为16；
3. 如果是单机，需要保证 MASTER_ADDR 与 CURRENT_IP 一致，如果为多机，需要保证各个机器的 MASTER_ADDR 一致，CURRENT_IP 为各个节点的 IP；
```bash
#上述注意点修改完毕后，可启动脚本开启训练
bash examples/grpo/grpo_trainer_qwen25_7b.sh