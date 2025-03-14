# QWQ-MindSpore_MindIE版

> [QwQ-32B MindSpore版本下载](https://modelers.cn/models/MindSpore-Lab/QwQ-32B)

## 目录
- [模型介绍](#模型介绍)
- [快速开始](#快速开始)
- [声明](#声明)

## 模型介绍
QwQ-32B是千问于2025年3月6日发布的人工智能大型语言模型。这是一款拥有 320 亿参数的模型，其性能可与具备 6710 亿参数（其中 370 亿被激活）的 DeepSeek-R1 媲美。这一成果突显了将强化学习应用于经过大规模预训练的强大基础模型的有效性。QwQ-32B 在一系列基准测试中进行了评估，测试了数学推理、编程能力和通用能力。以下结果展示了 QwQ-32B 与其他领先模型的性能对比，包括 DeepSeek-R1、OpenAI-o1-mini、DeepSeek-R1-Distilled-Llama-70B和DeepSeek-R1-Distilled-Qwen-32B。
![](https://modelscope.cn/api/v1/models/Qwen/QwQ-32B/repo?Revision=master&FilePath=figures%2Fbenchmark.jpg&View=true)

### 下载链接
|社区|下载地址|
|:--:|:--------------------|
|魔乐社区|https://modelers.cn/models/MindSpore-Lab/QwQ-32B|

## 快速开始
QwQ-32B推理至少需要1台（2卡）Atlas 800T A2（64G）服务器服务器（基于BF16权重）。昇思MindSpore提供了QwQ-32B推理可用的Docker容器镜像，供开发者快速体验。

### 下载昇思 MindSpore 推理容器镜像
执行以下 Shell 命令，拉取昇思 MindSpore 推理容器镜像（复用DeepSeek-V3的镜像）：
```bash
docker pull swr.cn-central-221.ovaijisuan.com/mindformers/deepseek_v3_mindspore2.5.0-infer:20250217
```

### 启动容器
执行以下命令创建并启动容器：
```bash
docker stop qwq-32b && docker rm qwq-32b
docker run -itd --privileged  --name=qwq-32b --net=host \
   --shm-size 500g \
   --device=/dev/davinci0 \
   --device=/dev/davinci1 \
   --device=/dev/davinci2 \
   --device=/dev/davinci3 \
   --device=/dev/davinci4 \
   --device=/dev/davinci5 \
   --device=/dev/davinci6 \
   --device=/dev/davinci7 \
   --device=/dev/davinci_manager \
   --device=/dev/hisi_hdc \
   --device /dev/devmm_svm \
   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
   -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
   -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
   -v /usr/local/sbin:/usr/local/sbin \
   -v /etc/hccn.conf:/etc/hccn.conf \
   -v /home/data/qwq32b_mindspore/QwQ-32B:/home/work/QwQ-32B \
   swr.cn-central-221.ovaijisuan.com/mindformers/deepseek_v3_mindspore2.5.0-infer:20250217 \
   bash

docker exec -it qwq-32b bash
```

#### 注意事项：
- 如果部署在多机上，每台机器中容器的hostname不能重复。如果有部分宿主机的hostname是一致的，需要在起容器的时候修改容器的hostname。
- 后续所有操作均在容器内操作。

### 模型下载
执行以下命令为自定义下载路径 `/home/work/QwQ-32B` 添加白名单：
```bash
export HUB_WHITE_LIST_PATHS=/home/work/QwQ-32B
```
执行以下 Python 脚本从魔乐社区下载昇思 MindSpore 版本的 QwQ-32B 文件至指定路径 `/home/work/QwQ-32B` 。下载的文件包含模型代码、权重、分词模型和示例代码，占用约 62GB 的磁盘空间：
```python
from openmind_hub import snapshot_download

snapshot_download(
    repo_id="MindSpore-Lab/QwQ-32B",
    local_dir="./QwQ-32B",
    local_dir_use_symlink=False
)
```

下载完成的 `/home/work/QwQ-32B` 文件夹目录结构如下：

```text
QwQ-32b
  ├── config.json                         # 模型json配置文件
  ├── vocab.json                          # 词表vocab文件
  ├── merges.txt                          # 词表merges文件
  ├── tokenizer.json                      # 词表json文件
  ├── tokenizer_config.json               # 词表配置文件
  ├── predict_qwq_32b.yaml                # 模型yaml配置文件
  ├── qwen2_5_tokenizer.py                # 模型tokenizer文件
  ├── model-xxxxx-of-xxxxx.safetensors    # 模型权重文件
  └── param_name_map.json                 # 模型权重映射文件
```

#### 注意事项：
- `/home/work/QwQ-32B` 可修改为自定义路径，确保该路径有足够的磁盘空间（约 62GB）。

- 模型权重文件和映射文件单独存放到一个文件夹目录下。

- 下载时间可能因网络环境而异，建议在稳定的网络环境下操作。

## 服务化部署
### 1. 修改模型配置文件

在 `predict_qwq_32b.yaml` 中对以下配置进行修改：

```yaml
auto_trans_ckpt: True                         # 打开权重自动切分，自动将权重转换为分布式任务所需的形式
load_checkpoint: '/home/work/QwQ-32B/checkpoint_dir'         # 配置为实际的模型绝对路径
processor:
  tokenizer:
    vocab_file: "/home/work/QwQ-32B/vocab.json"   # 配置为vocab文件的绝对路径
    merges_file: "/home/work/QwQ-32B/merges.txt"  # 配置为merges文件的绝对路径
```

### 2.一键启动MindIE

MindSpore Transformers提供了一键拉起MindIE脚本，脚本中已预置环境变量设置和服务化配置，仅需输入模型文件目录后即可快速拉起服务。
进入 `mindformers/scripts` 目录下，执行MindIE启动脚本

```bash
cd /home/work/mindformers/scripts && bash run_mindie.sh --model-name QwQ-32B --model-path /home/work/QwQ-32B --max-prefill-batch-size 1
```
> 以上进程会以后台方式启动，查看日志: tail -f output.log

终止进程
```
pkill -9 mind
pkill -9 python
```

#### 参数说明
- --model-name：设置模型名称
- --model-path：设置模型目录路径

#### 查看日志：
```bash
tail -f output.log
```
当log日志中出现 `Daemon start success!` ，表示服务启动成功。

### 3. 执行推理请求测试
执行以下命令发送流式推理请求进行测试：
```bash
curl -w "\ntime_total=%{time_total}\n" -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{"inputs": "请介绍一个北京的景点", "parameters": {"do_sample": false, "max_new_tokens": 128}, "stream": false}' http://127.0.0.1:1025/generate_stream
```

## 性能测试
> 参考 [性能测试](https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0012.html)
> [pip依赖参考](https://www.hiascend.com/document/detail/zh/mindie/100/envdeployment/instg/mindie_instg_0023.html)

```
benchmark \
--DatasetPath "/home/work/QwQ-32B/gsm8k" \
--DatasetType "gsm8k" \
--ModelName "QwQ-32B" \
--ModelPath "/home/work/QwQ-32B" \
--TestType client \
--Http http://127.0.0.1:1025 \
--ManagementHttp http://127.0.0.2:1026 \
--Concurrency 10 \
--DoSampling False \
--MaxOutputLen 512
```