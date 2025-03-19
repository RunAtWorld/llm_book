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

### 4卡B3 gsm8k测试

为了减少测试时间，这里抽取gsm8k的前200条数据测试。
```
export MINDIE_LOG_TO_STDOUT="benchmark:1; client:1"
cd /home/work/mindformers
nohup \
benchmark \
--DatasetPath "/home/work/QwQ-32B/gsm8k-200" \
--DatasetType "gsm8k" \
--ModelName "QwQ-32B" \
--ModelPath "/home/work/QwQ-32B" \
--TestType client \
--Http http://127.0.0.1:1025 \
--ManagementHttp http://127.0.0.2:1026 \
--Concurrency 10 \
--DoSampling False \
--MaxOutputLen 512 \
 > /home/work/QwQ-32B/gsm8k-200/mindspore_mindie_gsm8k_benchmark-200.log 2>&1 &
tail -f /home/work/QwQ-32B/gsm8k-200/mindspore_mindie_gsm8k_benchmark-200.log
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
    "Version" : "1.1.0",
    "LogConfig" :
    {
        "logLevel" : "Info",
        "logFileSize" : 20,
        "logFileNum" : 20,
        "logPath" : "logs/mindservice.log"
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
                    "npuMemSize": 35,
                    "backendType": "ms",
                    "trustRemoteCode" : false
                }
            ]
        },

        "ScheduleConfig" :
        {
            "templateType": "Standard",
            "templateName" : "Standard_LLM",
            "cacheBlockSize" : 128,

            "maxPrefillBatchSize": 50,
            "maxPrefillTokens": 8192,
            "prefillTimeMsPerReq" : 150,
            "prefillPolicyType" : 0,

            "decodeTimeMsPerReq" : 50,
            "decodePolicyType" : 0,

            "maxBatchSize" : 200,
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
[2025-03-18 11:42:57.065+00:00] [35602] [281473348116512] [benchmark] [INFO] [output.py:114]
The BenchMark test performance metric result is:
+---------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+-----+
|              Metric |        average |            max |            min |            P75 |            P90 |        SLO_P90 |            P99 |   N |
+---------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+-----+
|      FirstTokenTime |    106.8642 ms |    766.6764 ms |     61.2204 ms |     89.4486 ms |     96.8093 ms |     96.8093 ms |    678.2679 ms | 200 |
|          DecodeTime |    272.8685 ms | 240991.5447 ms |     18.9562 ms |       26.88 ms |     29.1581 ms |    507.8787 ms |     44.7554 ms | 200 |
|      LastDecodeTime |     27.3458 ms |     44.6792 ms |     23.6018 ms |     28.1029 ms |      30.421 ms |      30.421 ms |     43.4848 ms | 200 |
|       MaxDecodeTime |  110540.058 ms | 240991.5447 ms |  87634.1462 ms | 113301.1768 ms | 119612.0382 ms | 119612.0382 ms | 215188.2593 ms | 200 |
|        GenerateTime | 122445.6956 ms | 254751.9505 ms |  93431.3636 ms | 125816.2365 ms | 131786.2445 ms | 131786.2445 ms | 228628.2242 ms | 200 |
|         InputTokens |          60.89 |            139 |             25 |           71.0 |           87.2 |           87.2 |         110.05 | 200 |
|     GeneratedTokens |         449.32 |            512 |             70 |          512.0 |          512.0 |          512.0 |          512.0 | 200 |
| GeneratedTokenSpeed | 3.7201 token/s | 4.9928 token/s | 0.5706 token/s | 4.3837 token/s | 4.6082 token/s | 4.6082 token/s | 4.9746 token/s | 200 |
| GeneratedCharacters |       1523.965 |           2367 |            258 |         1832.0 |         1952.0 |         1952.0 |        2154.05 | 200 |
|           Tokenizer |      3.0427 ms |      8.4531 ms |      0.8283 ms |      3.8697 ms |       4.942 ms |       4.942 ms |       6.854 ms | 200 |
|         Detokenizer |      1.0629 ms |      1.3137 ms |      0.2007 ms |      1.2119 ms |      1.2252 ms |      1.2252 ms |      1.2969 ms | 200 |
|  CharactersPerToken |         3.3917 |              / |              / |              / |              / |              / |              / | 200 |
|  PostProcessingTime |           0 ms |           0 ms |           0 ms |           0 ms |           0 ms |           0 ms |           0 ms | 200 |
|         ForwardTime |           0 ms |           0 ms |           0 ms |           0 ms |           0 ms |           0 ms |           0 ms | 200 |
|    PrefillBatchsize |            1.0 |              1 |              1 |            1.0 |            1.0 |            1.0 |            1.0 | 200 |
|    DecoderBatchsize |            1.0 |              1 |              1 |            1.0 |            1.0 |              / |            1.0 | 200 |
|       QueueWaitTime | 246183.0655 μs |   240956110 μs |          11 μs |        46.0 μs |        57.0 μs |           / μs |       92.37 μs | 200 |
+---------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+-----+
[2025-03-18 11:42:57.066+00:00] [35602] [281473348116512] [benchmark] [INFO] [output.py:120]
The BenchMark test common metric result is:
+------------------------+------------------------------+
|          Common Metric |                        Value |
+------------------------+------------------------------+
|            CurrentTime |          2025-03-18 11:42:57 |
|            TimeElapsed |                  2502.9557 s |
|             DataSource | /home/work/QwQ-32B/gsm8k-200 |
|                 Failed |                    0( 0.0% ) |
|               Returned |                200( 100.0% ) |
|                  Total |                200[ 100.0% ] |
|            Concurrency |                           10 |
|              ModelName |                      QwQ-32B |
|                   lpct |                     1.755 ms |
|             Throughput |                 0.0799 req/s |
|          GenerateSpeed |              35.9032 token/s |
| GenerateSpeedPerClient |               3.5903 token/s |
|               accuracy |                            / |
+------------------------+------------------------------+
Benchmark task completed successfully!
```

### 性能调优


#### 测试0：4卡，"npuMemSize": 8，"maxPrefillBatchSize": 1
* "maxPrefillBatchSize": 1
* "npuMemSize": 8

```
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

以上结果性能较差，检查后发现，"npuMemSize": 8 比较低，内存没有占满。调整 "npuMemSize": 35

#### 调优1：4卡，"npuMemSize": 35，"maxPrefillBatchSize": 1
* "maxPrefillBatchSize": 1
* "npuMemSize": 35

```
[2025-03-18 11:42:57.065+00:00] [35602] [281473348116512] [benchmark] [INFO] [output.py:114]
The BenchMark test performance metric result is:
+---------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+-----+
|              Metric |        average |            max |            min |            P75 |            P90 |        SLO_P90 |            P99 |   N |
+---------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+-----+
|      FirstTokenTime |    106.8642 ms |    766.6764 ms |     61.2204 ms |     89.4486 ms |     96.8093 ms |     96.8093 ms |    678.2679 ms | 200 |
|          DecodeTime |    272.8685 ms | 240991.5447 ms |     18.9562 ms |       26.88 ms |     29.1581 ms |    507.8787 ms |     44.7554 ms | 200 |
|      LastDecodeTime |     27.3458 ms |     44.6792 ms |     23.6018 ms |     28.1029 ms |      30.421 ms |      30.421 ms |     43.4848 ms | 200 |
|       MaxDecodeTime |  110540.058 ms | 240991.5447 ms |  87634.1462 ms | 113301.1768 ms | 119612.0382 ms | 119612.0382 ms | 215188.2593 ms | 200 |
|        GenerateTime | 122445.6956 ms | 254751.9505 ms |  93431.3636 ms | 125816.2365 ms | 131786.2445 ms | 131786.2445 ms | 228628.2242 ms | 200 |
|         InputTokens |          60.89 |            139 |             25 |           71.0 |           87.2 |           87.2 |         110.05 | 200 |
|     GeneratedTokens |         449.32 |            512 |             70 |          512.0 |          512.0 |          512.0 |          512.0 | 200 |
| GeneratedTokenSpeed | 3.7201 token/s | 4.9928 token/s | 0.5706 token/s | 4.3837 token/s | 4.6082 token/s | 4.6082 token/s | 4.9746 token/s | 200 |
| GeneratedCharacters |       1523.965 |           2367 |            258 |         1832.0 |         1952.0 |         1952.0 |        2154.05 | 200 |
|           Tokenizer |      3.0427 ms |      8.4531 ms |      0.8283 ms |      3.8697 ms |       4.942 ms |       4.942 ms |       6.854 ms | 200 |
|         Detokenizer |      1.0629 ms |      1.3137 ms |      0.2007 ms |      1.2119 ms |      1.2252 ms |      1.2252 ms |      1.2969 ms | 200 |
|  CharactersPerToken |         3.3917 |              / |              / |              / |              / |              / |              / | 200 |
|  PostProcessingTime |           0 ms |           0 ms |           0 ms |           0 ms |           0 ms |           0 ms |           0 ms | 200 |
|         ForwardTime |           0 ms |           0 ms |           0 ms |           0 ms |           0 ms |           0 ms |           0 ms | 200 |
|    PrefillBatchsize |            1.0 |              1 |              1 |            1.0 |            1.0 |            1.0 |            1.0 | 200 |
|    DecoderBatchsize |            1.0 |              1 |              1 |            1.0 |            1.0 |              / |            1.0 | 200 |
|       QueueWaitTime | 246183.0655 μs |   240956110 μs |          11 μs |        46.0 μs |        57.0 μs |           / μs |       92.37 μs | 200 |
+---------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+-----+
[2025-03-18 11:42:57.066+00:00] [35602] [281473348116512] [benchmark] [INFO] [output.py:120]
The BenchMark test common metric result is:
+------------------------+------------------------------+
|          Common Metric |                        Value |
+------------------------+------------------------------+
|            CurrentTime |          2025-03-18 11:42:57 |
|            TimeElapsed |                  2502.9557 s |
|             DataSource | /home/work/QwQ-32B/gsm8k-200 |
|                 Failed |                    0( 0.0% ) |
|               Returned |                200( 100.0% ) |
|                  Total |                200[ 100.0% ] |
|            Concurrency |                           10 |
|              ModelName |                      QwQ-32B |
|                   lpct |                     1.755 ms |
|             Throughput |                 0.0799 req/s |
|          GenerateSpeed |              35.9032 token/s |
| GenerateSpeedPerClient |               3.5903 token/s |
|               accuracy |                            / |
+------------------------+------------------------------+
Benchmark task completed successfully!
```

发现性能基本无提升。将"maxPrefillBatchSize"设为 50

#### 测试2：4卡，"npuMemSize": 35，"maxPrefillBatchSize": 50

* "maxPrefillBatchSize": 50
* "npuMemSize": 35

启动命令改为
```
cd /home/work/mindformers/scripts && bash run_mindie.sh --model-name QwQ-32B --model-path /home/work/QwQ-32B --max-prefill-batch-size 50
```

```
[2025-03-19 08:29:17.778+00:00] [27897] [281473619693600] [benchmark] [INFO] [output.py:114]
The BenchMark test performance metric result is:
+---------------------+-----------------+-----------------+-----------------+-----------------+----------------+----------------+-----------------+-----+
|              Metric |         average |             max |             min |             P75 |            P90 |        SLO_P90 |             P99 |   N |
+---------------------+-----------------+-----------------+-----------------+-----------------+----------------+----------------+-----------------+-----+
|      FirstTokenTime |        97.07 ms |          169 ms |           62 ms |        110.0 ms |       130.1 ms |       130.1 ms |       164.02 ms | 200 |
|          DecodeTime |      29.1162 ms |          212 ms |           24 ms |         29.0 ms |        31.0 ms |     29.9204 ms |         50.0 ms | 200 |
|      LastDecodeTime |       30.005 ms |           99 ms |           26 ms |         30.0 ms |        32.0 ms |        32.0 ms |        90.07 ms | 200 |
|       MaxDecodeTime |       110.82 ms |          212 ms |           42 ms |        123.0 ms |       130.0 ms |       130.0 ms |       184.15 ms | 200 |
|        GenerateTime |    14356.007 ms |     16303.02 ms |     2164.943 ms |   15433.3689 ms |  15628.6762 ms |  15628.6762 ms |   16297.7436 ms | 200 |
|         InputTokens |           60.89 |             139 |              25 |            71.0 |           87.2 |           87.2 |          110.05 | 200 |
|     GeneratedTokens |          482.07 |             512 |              70 |           512.0 |          512.0 |          512.0 |           512.0 | 200 |
| GeneratedTokenSpeed | 33.5801 token/s | 34.9831 token/s | 31.4052 token/s | 34.3461 token/s | 34.453 token/s | 34.453 token/s | 34.7207 token/s | 200 |
| GeneratedCharacters |         1649.17 |            2309 |             258 |         1853.25 |         1979.5 |         1979.5 |         2284.18 | 200 |
|           Tokenizer |       2.4258 ms |       7.7391 ms |       0.7679 ms |       2.7465 ms |       3.667 ms |       3.667 ms |       7.4379 ms | 200 |
|         Detokenizer |       1.1688 ms |       2.1422 ms |       0.2027 ms |       1.2372 ms |      1.2781 ms |      1.2781 ms |       1.4362 ms | 200 |
|  CharactersPerToken |           3.421 |               / |               / |               / |              / |              / |               / | 200 |
|  PostProcessingTime |            0 ms |            0 ms |            0 ms |            0 ms |           0 ms |           0 ms |            0 ms | 200 |
|         ForwardTime |            0 ms |            0 ms |            0 ms |            0 ms |           0 ms |           0 ms |            0 ms | 200 |
|    PrefillBatchsize |          1.8519 |               9 |               1 |            1.25 |            5.0 |            5.0 |            7.93 | 200 |
|    DecoderBatchsize |          9.7255 |              10 |               1 |            10.0 |           10.0 |              / |            10.0 | 200 |
|       QueueWaitTime |     601.6063 μs |       170470 μs |           15 μs |         88.0 μs |       109.0 μs |           / μs |     24682.83 μs | 200 |
+---------------------+-----------------+-----------------+-----------------+-----------------+----------------+----------------+-----------------+-----+
[2025-03-19 08:29:17.779+00:00] [27897] [281473619693600] [benchmark] [INFO] [output.py:120]
The BenchMark test common metric result is:
+------------------------+------------------------------+
|          Common Metric |                        Value |
+------------------------+------------------------------+
|            CurrentTime |          2025-03-19 08:29:17 |
|            TimeElapsed |                   294.5528 s |
|             DataSource | /home/work/QwQ-32B/gsm8k-200 |
|                 Failed |                    0( 0.0% ) |
|               Returned |                200( 100.0% ) |
|                  Total |                200[ 100.0% ] |
|            Concurrency |                           10 |
|              ModelName |                      QwQ-32B |
|                   lpct |                    1.5942 ms |
|             Throughput |                  0.679 req/s |
|          GenerateSpeed |             327.3233 token/s |
| GenerateSpeedPerClient |              32.7323 token/s |
|               accuracy |                            / |
+------------------------+------------------------------+
```

> 注意：原先的版本，把"maxPrefillBatchSize"设置超过1，启动mindie就会报错：
>
> scheduleParam.maxPrefillBatchSize [2] is out of bound [1]
> ERR: Failed to init endpoint! Please check the service log or console output.
>
> 需要更换mindie为 1.0.0版本，且需要修改mindie源码。详细方法参考FAQ

## FAQ

#### 一、更换mindie版本为1.0.0，并修改mindie源码

1. 下载 mindie [安装包](https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/MindIE/MindIE%201.0.0/Ascend-mindie_1.0.0_linux-aarch64.run?response-content-type=application/octet-stream)，安装参考[文档](https://www.hiascend.com/document/detail/zh/mindie/100/envdeployment/instg/mindie_instg_0019.html)

2. 安装后mindie后，修改  `/usr/local/lib/python3.10/dist-packages/mindie_llm/modeling/model_wrapper/ms/mf_model_wrapper.py`两处源码：

   第1处：注释掉disable_custom_fa的异常

   ```
    41 #        try:
    42 #            from mindformers.tools.utils import get_disable_custom_fa
    43 #            self.disable_custom_fa = get_disable_custom_fa()
    44 #        except ImportError:
    45 #            pass
   ```

   第2处：注释掉self.disable_custom_fa分支

   ```
    74         #if model_inputs.is_prefill and self.disable_custom_fa:
    75         if model_inputs.is_prefill:
   ```

   > 如果mindie_llm的安装目录不在/usr/local/lib/python3.10/dist-packages， 查询mindie_llm的安装位置：pip show mindie_llm