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

- 测试triton流式推理接口
```bash
curl -w "\ntime_total=%{time_total}\n" -H "Accept: application/json" -H "Content-type: application/json" -X POST -d '{"inputs": "请介绍一个北京的景点", "parameters": {"do_sample": false, "max_new_tokens": 128}, "stream": false}' http://127.0.0.1:1025/generate_stream
```

- 测试OpenAI 接口
```shell
curl -X POST 127.0.0.1:1025/v1/chat/completions \
  -d "{
    \"messages\": [
        {\"role\": \"system\", \"content\": \"you are a helpful assistant.\"},
        {\"role\": \"user\", \"content\": \"How many r's are in the word \\\"strawberry\\\"\"}
    ],
    \"max_tokens\": 32768,
    \"stream\": false,
    \"do_sample\": true,
    \"repetition_penalty\": 1.00,
    \"temperature\": 0.6,
    \"top_p\": 0.95,
    \"top_k\": 20,
    \"model\": \"QwQ-32B\"
  }"
```

## 性能测试
> 参考 [性能测试](https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0012.html)
> [pip依赖参考](https://www.hiascend.com/document/detail/zh/mindie/100/envdeployment/instg/mindie_instg_0023.html)

### 4卡B3 gsm8k测试结果

```
export MINDIE_LOG_TO_STDOUT="benchmark:1; client:1"
cd /home/work/mindformers
nohup \
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
--MaxOutputLen 512 \
 > /home/work/QwQ-32B/gsm8k/mindspore_mindie_gsm8k_benchmark.log 2>&1 &
tail -f /home/work/QwQ-32B/gsm8k/mindspore_mindie_gsm8k_benchmark.log
```

模型配置

```
seed: 0
output_dir: './output' # path to save checkpoint/strategy
load_checkpoint: "/home/work/QwQ-32B/checkpoint_dir"
src_strategy_path_or_dir: ''
auto_trans_ckpt: True  # If true, auto transform load_checkpoint to load in distributed model
only_save_strategy: False
resume_training: False
use_parallel: True
run_mode: 'predict'
load_ckpt_format: 'safetensors'

# trainer config
trainer:
  type: CausalLanguageModelingTrainer
  model_name: 'qwq_32b'

# runner config
runner_config:
  epochs: 5
  batch_size: 1
  sink_mode: True
  sink_size: 2
runner_wrapper:
  type: MFTrainOneStepCell
  scale_sense:
    type: DynamicLossScaleUpdateCell
    loss_scale_value: 65536
    scale_factor: 2
    scale_window: 1000
  use_clip_grad: True

# default parallel of device num = 8 for Atlas 800T A2
parallel_config:
  data_parallel: 1
  model_parallel: 4
  pipeline_stage: 1
  micro_batch_num: 1
  vocab_emb_dp: False
  gradient_aggregation_group: 4
# when model parallel is greater than 1, we can set micro_batch_interleave_num=2, that may accelerate the train process.
micro_batch_interleave_num: 1

model:
  model_config:
    type: LlamaConfig
    batch_size: 1
    seq_length: 32768
    hidden_size: 5120
    num_layers: 64
    num_heads: 40
    n_kv_heads: 8
    vocab_size: 152064
    intermediate_size: 27648
    max_position_embeddings: 32768
    qkv_has_bias: True
    rms_norm_eps: 1.0e-6
    theta: 1000000.0
    emb_dropout_prob: 0.0
    eos_token_id: [151645,151643]
    pad_token_id: 151643
    bos_token_id: 151643
    compute_dtype: "bfloat16"
    layernorm_compute_type: "float32"
    softmax_compute_type: "float32"
    rotary_dtype: "bfloat16"
    param_init_type: "bfloat16"
    use_past: True
    use_flash_attention: True
    block_size: 32
    num_blocks: 1024
    use_past_shard: False
    offset: 0
    checkpoint_name_or_path: ""
    repetition_penalty: 1.05
    temperature: 0.7
    max_decode_length: 512
    top_k: 20
    top_p: 0.8
    do_sample: True
    is_dynamic: True
    qkv_concat: True
    auto_map:
      AutoTokenizer: [qwen2_5_tokenizer.Qwen2Tokenizer, null]
  arch:
    type: LlamaForCausalLM

processor:
  return_tensors: ms
  tokenizer:
    model_max_length: 131072
    vocab_file: "/home/work/QwQ-32B/vocab.json"
    merges_file: "/home/work/QwQ-32B/merges.txt"
    unk_token: null
    pad_token: "<|endoftext|>"
    eos_token: "<|im_end|>"
    bos_token: null
    chat_template: "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"
    type: Qwen2Tokenizer
  type: Qwen2Processor

# mindspore context init config
context:
  mode: 0 #0--Graph Mode; 1--Pynative Mode
  device_target: "Ascend"
  ascend_config:
    precision_mode: "must_keep_origin_dtype"
  max_call_depth: 10000
  max_device_memory: "59GB"
  save_graphs: False
  save_graphs_path: "./graph"
  device_id: 0

# parallel context config
parallel:
  parallel_mode: 1 # 0-data parallel, 1-semi-auto parallel, 2-auto parallel, 3-hybrid parallel
  gradients_mean: False
  enable_alltoall: False
  full_batch: True
  search_mode: "sharding_propagation"
  enable_parallel_optimizer: False
  strategy_ckpt_save_file: "./ckpt_strategy.ckpt"
  parallel_optimizer_config:
    gradient_accumulation_shard: False
    parallel_optimizer_threshold: 64
```

mindIE配置

```
{
    "Version" : "1.0.0",
    "LogConfig" :
    {
        "logLevel" : "Info",
        "logFileSize" : 20,
        "logFileNum" : 20,
        "logPath" : "logs/mindie-server.log"
    },

    "ServerConfig" :
    {
        "ipAddress": "127.0.0.1",
        "managementIpAddress": "127.0.0.2",
        "port": 1025,
        "managementPort": 1026,
        "metricsPort": 1027,
        "allowAllZeroIpListening" : false,
        "maxLinkNum" : 1000,
        "httpsEnabled": false,
        "fullTextEnabled" : false,
        "tlsCaPath" : "security/ca/",
        "tlsCaFile" : ["ca.pem"],
        "tlsCert" : "security/certs/server.pem",
        "tlsPk" : "security/keys/server.key.pem",
        "tlsPkPwd" : "security/pass/key_pwd.txt",
        "tlsCrlPath" : "security/certs/",
        "tlsCrlFiles" : ["server_crl.pem"],
        "managementTlsCaFile" : ["management_ca.pem"],
        "managementTlsCert" : "security/certs/management/server.pem",
        "managementTlsPk" : "security/keys/management/server.key.pem",
        "managementTlsPkPwd" : "security/pass/management/key_pwd.txt",
        "managementTlsCrlPath" : "security/management/certs/",
        "managementTlsCrlFiles" : ["server_crl.pem"],
        "kmcKsfMaster" : "tools/pmt/master/ksfa",
        "kmcKsfStandby" : "tools/pmt/standby/ksfb",
        "inferMode" : "standard",
        "interCommTLSEnabled" : true,
        "interCommPort" : 1121,
        "interCommTlsCaPath" : "security/grpc/ca/",
        "interCommTlsCaFiles" : ["ca.pem"],
        "interCommTlsCert" : "security/grpc/certs/server.pem",
        "interCommPk" : "security/grpc/keys/server.key.pem",
        "interCommPkPwd" : "security/grpc/pass/key_pwd.txt",
        "interCommTlsCrlPath" : "security/grpc/certs/",
        "interCommTlsCrlFiles" : ["server_crl.pem"],
        "openAiSupport" : "vllm"
    },

    "BackendConfig" : {
        "backendName" : "mindieservice_llm_engine",
        "modelInstanceNumber" : 1,
        "npuDeviceIds" : [[0,1,2,3]],
        "tokenizerProcessNumber" : 8,
        "multiNodesInferEnabled": false,
        "multiNodesInferPort" : 1120,
        "interNodeTLSEnabled" : true,
        "interNodeTlsCaPath" : "security/grpc/ca/",
        "interNodeTlsCaFiles" : ["ca.pem"],
        "interNodeTlsCert" : "security/grpc/certs/server.pem",
        "interNodeTlsPk" : "security/grpc/keys/server.key.pem",
        "interNodeTlsPkPwd" : "security/grpc/pass/mindie_server_key_pwd.txt",
        "interNodeTlsCrlPath" : "security/grpc/certs/",
        "interNodeTlsCrlFiles" : ["server_crl.pem"],
        "interNodeKmcKsfMaster" : "tools/pmt/master/ksfa",
        "interNodeKmcKsfStandby" : "tools/pmt/standby/ksfb",
        "ModelDeployConfig" :
        {
            "maxSeqLen": 2560,
            "maxInputTokenLen": 2048,
            "truncation": false,
            "ModelConfig" : [
                {
                    "modelInstanceType" : "Standard",
                    "modelName": "QwQ-32B",
                    "modelWeightPath": "/home/work/QwQ-32B",
                    "worldSize": 4,
                    "cpuMemSize" : 5,
                    "npuMemSize": 8,
                    "backendType": "ms",
                    "trustRemoteCode" : false
                }
            ]
        },

        "ScheduleConfig" :
        {
            "templateType": "Standard",
            "templateName" : "Standard_LLM",
            "cacheBlockSize" : 16,
            
            "maxPrefillBatchSize": 1,
            "maxPrefillTokens": 8192,
            "prefillTimeMsPerReq" : 150,
            "prefillPolicyType" : 0,
            
            "decodeTimeMsPerReq" : 50,
            "decodePolicyType" : 0,
            "maxBatchSize" : 1,
            "maxIterTimes": 512,
            "maxPreemptCount": 0,
            "supportSelectBatch": false,
            "maxQueueDelayMicroseconds" : 5000
        }
    }
}
```

gsm8k_benchmark 测试结果

```
[2025-03-15 15:14:28.658+00:00] [18521] [281473289023520] [benchmark] [INFO] [output.py:114]
The BenchMark test performance metric result is:
+---------------------+----------------+-----------------+----------------+----------------+----------------+----------------+----------------+------+
|              Metric |        average |             max |            min |            P75 |            P90 |        SLO_P90 |            P99 |    N |
+---------------------+----------------+-----------------+----------------+----------------+----------------+----------------+----------------+------+
|      FirstTokenTime |     88.1944 ms |     798.5878 ms |     58.7664 ms |     94.6069 ms |    101.4279 ms |    101.4279 ms |    117.7145 ms | 8792 |
|          DecodeTime |    264.9084 ms |  221952.7402 ms |       0.011 ms |     26.7239 ms |     28.9207 ms |    473.2623 ms |     48.6279 ms | 8792 |
|      LastDecodeTime |      27.154 ms |      60.4694 ms |           0 ms |     27.6552 ms |     29.8138 ms |     29.8138 ms |      47.966 ms | 8792 |
|       MaxDecodeTime |  107179.843 ms |  221952.7402 ms |  64154.0973 ms | 114097.2967 ms | 119005.0737 ms | 119005.0737 ms | 125618.2632 ms | 8792 |
|        GenerateTime | 118536.9295 ms |  235183.5744 ms |     69.5162 ms | 126327.1709 ms | 131720.5082 ms | 131720.5082 ms | 138796.4644 ms | 8792 |
|         InputTokens |        59.8964 |             212 |             12 |           71.0 |           89.0 |           89.0 |          128.0 | 8792 |
|     GeneratedTokens |       448.1056 |             512 |              1 |          512.0 |          512.0 |          512.0 |          512.0 | 8792 |
| GeneratedTokenSpeed | 3.8218 token/s | 14.3851 token/s | 0.3871 token/s | 4.3693 token/s | 4.6785 token/s | 4.6785 token/s |  5.501 token/s | 8792 |
| GeneratedCharacters |      1510.9037 |            2545 |              0 |        1816.25 |         1967.0 |         1967.0 |         2213.0 | 8792 |
|           Tokenizer |      2.3106 ms |       11.507 ms |      0.7014 ms |      2.6737 ms |       3.729 ms |       3.729 ms |      6.4025 ms | 8792 |
|         Detokenizer |      1.0502 ms |       4.0472 ms |       0.015 ms |      1.2019 ms |      1.2228 ms |      1.2228 ms |      1.2856 ms | 8792 |
|  CharactersPerToken |         3.3718 |               / |              / |              / |              / |              / |              / | 8792 |
|  PostProcessingTime |           0 ms |            0 ms |           0 ms |           0 ms |           0 ms |           0 ms |           0 ms | 8792 |
|         ForwardTime |           0 ms |            0 ms |           0 ms |           0 ms |           0 ms |           0 ms |           0 ms | 8792 |
|    PrefillBatchsize |            1.0 |               1 |              1 |            1.0 |            1.0 |            1.0 |            1.0 | 8792 |
|    DecoderBatchsize |            1.0 |               1 |              1 |            1.0 |            1.0 |              / |            1.0 | 8792 |
|       QueueWaitTime | 238119.3935 μs |    221922274 μs |          10 μs |        55.0 μs |        65.0 μs |           / μs |       142.0 μs | 8792 |
+---------------------+----------------+-----------------+----------------+----------------+----------------+----------------+----------------+------+
[2025-03-15 15:14:28.660+00:00] [18521] [281473289023520] [benchmark] [INFO] [output.py:120]
The BenchMark test common metric result is:
+------------------------+--------------------------+
|          Common Metric |                    Value |
+------------------------+--------------------------+
|            CurrentTime |      2025-03-15 15:14:28 |
|            TimeElapsed |            104279.0743 s |
|             DataSource | /home/work/QwQ-32B/gsm8k |
|                 Failed |                0( 0.0% ) |
|               Returned |           8792( 100.0% ) |
|                  Total |           8792[ 100.0% ] |
|            Concurrency |                       10 |
|              ModelName |                  QwQ-32B |
|                   lpct |                1.4724 ms |
|             Throughput |             0.0843 req/s |
|          GenerateSpeed |          37.7808 token/s |
| GenerateSpeedPerClient |           3.7781 token/s |
|               accuracy |                        / |
+------------------------+--------------------------+
```
