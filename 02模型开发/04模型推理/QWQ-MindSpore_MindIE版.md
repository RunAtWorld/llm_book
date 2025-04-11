# QWQ-MindSpore_MindIE版

> [QwQ-32B MindSpore版本下载](https://modelers.cn/models/MindSpore-Lab/QwQ-32B)

## 模型介绍
QwQ-32B是千问于2025年3月6日发布的人工智能大型语言模型。这是一款拥有 320 亿参数的模型，其性能可与具备 6710 亿参数（其中 370 亿被激活）的 DeepSeek-R1 媲美。这一成果突显了将强化学习应用于经过大规模预训练的强大基础模型的有效性。QwQ-32B 在一系列基准测试中进行了评估，测试了数学推理、编程能力和通用能力。以下结果展示了 QwQ-32B 与其他领先模型的性能对比，包括 DeepSeek-R1、OpenAI-o1-mini、DeepSeek-R1-Distilled-Llama-70B和DeepSeek-R1-Distilled-Qwen-32B。
![](https://modelscope.cn/api/v1/models/Qwen/QwQ-32B/repo?Revision=master&FilePath=figures%2Fbenchmark.jpg&View=true)

### 下载链接
|社区|下载地址|
|:--:|:--------------------|
|魔乐社区|https://modelers.cn/models/MindSpore-Lab/QwQ-32B|

## 快速开始
QwQ-32B推理至少需要1台（2卡）Atlas 800T A2（64G）服务器服务器（基于BF16权重）。昇思MindSpore提供了QwQ-32B推理可用的Docker容器镜像，供开发者快速体验。

### 下载MindSpore 推理容器镜像
执行以下命令，拉取昇思 MindSpore 推理容器镜像（复用DeepSeek-V3的镜像）：
```bash
docker pull swr.cn-central-221.ovaijisuan.com/mindformers/deepseek_v3_mindspore2.5.0-infer:20250217
```

各组件版本配套如下：

| 组件       | 版本   |
| :--------- | :----- |
| MindFormers     | deepseek_v3_infer_v1.2分支  |
| MindSpore     | 2.5.0  |
| MindIE     | 1.0.T71  |
| CANN       |  8.0.T63  |
| HDK        | 24.1.rc2 |


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

**注意事项：**

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

**注意事项：**

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

注意：
`run_mindie.sh`会修改MindIE服务配置文件 `/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json`
```
265 sed -i "s/\"backendType\"\s*:\s*\"[^\"]*\"/\"backendType\": \"$BACKEND_TYPE\"/" $CONFIG_FILE
266 sed -i "s/\"modelName\"\s*:\s*\"[^\"]*\"/\"modelName\": \"$MODEL_NAME\"/" $CONFIG_FILE
267 sed -i "s|\"modelWeightPath\"\s*:\s*\"[^\"]*\"|\"modelWeightPath\": \"$MODEL_WEIGHT_PATH\"|" $CONFIG_FILE
268 sed -i "s/\"maxSeqLen\"\s*:\s*[0-9]*/\"maxSeqLen\": $MAX_SEQ_LEN/" "$CONFIG_FILE"
269 sed -i "s/\"maxPrefillTokens\"\s*:\s*[0-9]*/\"maxPrefillTokens\": $MAX_PREFILL_TOKENS/" "$CONFIG_FILE"
270 sed -i "s/\"maxIterTimes\"\s*:\s*[0-9]*/\"maxIterTimes\": $MAX_ITER_TIMES/" "$CONFIG_FILE"
271 sed -i "s/\"maxInputTokenLen\"\s*:\s*[0-9]*/\"maxInputTokenLen\": $MAX_INPUT_TOKEN_LEN/" "$CONFIG_FILE"
272 sed -i "s/\"truncation\"\s*:\s*[a-z]*/\"truncation\": $TRUNCATION/" "$CONFIG_FILE"
273 sed -i "s|\(\"npuDeviceIds\"\s*:\s*\[\[\)[^]]*\(]]\)|\1$NPU_DEVICE_IDS\2|" "$CONFIG_FILE"
274 sed -i "s/\"worldSize\"\s*:\s*[0-9]*/\"worldSize\": $WORLD_SIZE/" "$CONFIG_FILE"
275 sed -i "s/\"httpsEnabled\"\s*:\s*[a-z]*/\"httpsEnabled\": $HTTPS_ENABLED/" "$CONFIG_FILE"
276 sed -i "s/\"templateType\"\s*:\s*\"[^\"]*\"/\"templateType\": \"$TEMPLATE_TYPE\"/" $CONFIG_FILE
277 sed -i "s/\"maxPreemptCount\"\s*:\s*[0-9]*/\"maxPreemptCount\": $MAX_PREEMPT_COUNT/" $CONFIG_FILE
278 sed -i "s/\"supportSelectBatch\"\s*:\s*[a-z]*/\"supportSelectBatch\": $SUPPORT_SELECT_BATCH/" $CONFIG_FILE
279 sed -i "s/\"multiNodesInferEnabled\"\s*:\s*[a-z]*/\"multiNodesInferEnabled\": $MULTI_NODES_INFER_ENABLED/" "$CONFIG_FILE"
280 sed -i "s/\"maxPrefillBatchSize\"\s*:\s*[0-9]*/\"maxPrefillBatchSize\": $MAX_PREFILL_BATCH_SIZE/" "$CONFIG_FILE"
281 sed -i "s/\"ipAddress\"\s*:\s*\"[^\"]*\"/\"ipAddress\": \"$IP_ADDRESS\"/" "$CONFIG_FILE"
282 sed -i "s/\"port\"\s*:\s*[0-9]*/\"port\": $PORT/" "$CONFIG_FILE"
283 sed -i "s/\"managementIpAddress\"\s*:\s*\"[^\"]*\"/\"managementIpAddress\": \"$MANAGEMENT_IP_ADDRESS\"/" "$CONFIG_FILE"
284 sed -i "s/\"managementPort\"\s*:\s*[0-9]*/\"managementPort\": $MANAGEMENT_PORT/" "$CONFIG_FILE"
285 sed -i "s/\"metricsPort\"\s*:\s*[0-9]*/\"metricsPort\": $METRICS_PORT/" $CONFIG_FILE
286 sed -i "s/\"npuMemSize\"\s*:\s*-*[0-9]*/\"npuMemSize\": $NPU_MEM_SIZE/" "$CONFIG_FILE"
```

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
> 参考 [MindIE的性能测试说明](https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0012.html)
>
> 由于官方仓库给的镜像中，没有benchmark的相关依赖，需要自己安装相关[pip依赖](https://www.hiascend.com/document/detail/zh/mindie/100/envdeployment/instg/mindie_instg_0023.html)

### 4卡B3 gsm8k第一次测试

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

#### 第一次测试下性能较差

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

从第一次测试结果来看，性能较差，核心关注两个配置

* "maxPrefillBatchSize": 1
* "npuMemSize": 8

"npuMemSize": 8 比较低，内存没有占满。调整 "npuMemSize": 35

#### 调优1：调整"npuMemSize"到 35
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
增大KV Cache内存后，发现性能基本无提升。怀疑"maxPrefillBatchSize"太小，将 "maxPrefillBatchSize" 设为 50

#### 调优2："maxPrefillBatchSize": 50

增大"maxPrefillBatchSize" 到 50

* "maxPrefillBatchSize": 50
* "npuMemSize": 35

启动命令改为
```
cd /home/work/mindformers/scripts && bash run_mindie.sh --model-name QwQ-32B --model-path /home/work/QwQ-32B --max-prefill-batch-size 50
```

发现启动报错：
```
scheduleParam.maxPrefillBatchSize [50] is out of bound [1]
```
经过与研发确认，原因为该镜像中mindIE版本不支持maxPrefillBatchSize超过1，需要更换mindie为 1.0.0版本，且需要修改mindie源码。详细方法参考FAQ。

修改后，测试结果如下

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

## FAQ

#### 一、运行 benchmark 报错：缺失依赖

**问题现象：**

运行 benchmark 工具报错：缺失依赖

**解决过程：**

由于官方仓库给的镜像中，没有benchmark的相关依赖，需要自己安装相关[pip依赖](https://www.hiascend.com/document/detail/zh/mindie/100/envdeployment/instg/mindie_instg_0023.html)

**解决方法：**

参考官方说明，pip依赖列表如下

```
#
# This file is autogenerated by pip-compile with Python 3.10
# by the following command:
#
#    pip-compile

absl-py==2.1.0
    # via rouge-score
accelerate==0.34.2
    # via -r requirements.in
annotated-types==0.7.0
    # via pydantic
attrs==24.2.0
    # via jsonlines
brotli==1.1.0
    # via geventhttpclient
certifi==2024.8.30
    # via
    #   geventhttpclient
    #   requests
cffi==1.17.1
    # via -r requirements.in
charset-normalizer==3.3.2
    # via requests
click==8.1.7
    # via nltk
cloudpickle==3.0.0
    # via -r requirements.in
colorama==0.4.6
    # via sacrebleu
contourpy==1.3.0
    # via matplotlib
cpm-kernels==1.0.11
    # via -r requirements.in
cycler==0.12.1
    # via matplotlib
decorator==5.1.1
    # via -r requirements.in
et-xmlfile==1.1.0
    # via openpyxl
filelock==3.16.1
    # via
    #   huggingface-hub
    #   icetk
    #   torch
    #   transformers
fonttools==4.54.1
    # via matplotlib
fsspec==2024.9.0
    # via
    #   huggingface-hub
    #   torch
fuzzywuzzy==0.18.0
    # via -r requirements.in
gevent==24.2.1
    # via geventhttpclient
geventhttpclient==2.3.1
    # via -r requirements.in
greenlet==3.1.1
    # via gevent
grpcio==1.66.1
    # via tritonclient
huggingface-hub==0.25.1
    # via
    #   accelerate
    #   tokenizers
    #   transformers
icetk==0.0.4
    # via -r requirements.in
idna==3.10
    # via requests
jieba==0.42.1
    # via -r requirements.in
jinja2==3.1.4
    # via torch
joblib==1.4.2
    # via nltk
jsonlines==4.0.0
    # via -r requirements.in
kiwisolver==1.4.7
    # via matplotlib
latex2mathml==3.77.0
    # via mdtex2html
loguru==0.7.2
    # via -r requirements.in
lxml==5.3.0
    # via sacrebleu
markdown==3.7
    # via mdtex2html
markupsafe==2.1.5
    # via jinja2
matplotlib==3.9.2
    # via -r requirements.in
mdtex2html==1.3.0
    # via -r requirements.in
ml-dtypes==0.5.0
    # via -r requirements.in
mpmath==1.3.0
    # via sympy
networkx==3.3
    # via torch
nltk==3.9.1
    # via rouge-score
numpy==1.26.4
    # via
    #   -r requirements.in
    #   accelerate
    #   contourpy
    #   matplotlib
    #   ml-dtypes
    #   pandas
    #   pyarrow
    #   rouge-score
    #   sacrebleu
    #   scipy
    #   torchvision
    #   transformers
    #   tritonclient
openpyxl==3.1.5
    # via -r requirements.in
packaging==24.1
    # via
    #   accelerate
    #   huggingface-hub
    #   matplotlib
    #   transformers
    #   tritonclient
pandas==2.2.3
    # via -r requirements.in
pathlib2==2.3.7.post1
    # via -r requirements.in
pillow==10.4.0
    # via
    #   matplotlib
    #   torchvision
portalocker==2.10.1
    # via sacrebleu
prettytable==3.11.0
    # via -r requirements.in
protobuf==3.20.0
    # via
    #   -r requirements.in
    #   tritonclient
psutil==6.0.0
    # via accelerate
pyarrow==17.0.0
    # via -r requirements.in
pycparser==2.22
    # via cffi
pydantic==2.9.2
    # via -r requirements.in
pydantic-core==2.23.4
    # via pydantic
pyparsing==3.1.4
    # via matplotlib
python-dateutil==2.9.0.post0
    # via
    #   matplotlib
    #   pandas
python-rapidjson==1.20
    # via tritonclient
pytz==2024.2
    # via pandas
pyyaml==6.0.2
    # via
    #   accelerate
    #   huggingface-hub
    #   transformers
rapidfuzz==3.10.0
    # via thefuzz
regex==2024.9.11
    # via
    #   nltk
    #   sacrebleu
    #   tiktoken
    #   transformers
requests==2.32.3
    # via
    #   huggingface-hub
    #   icetk
    #   tiktoken
    #   torchvision
    #   transformers
rouge==1.0.1
    # via -r requirements.in
rouge-score==0.1.2
    # via -r requirements.in
sacrebleu==2.4.3
    # via -r requirements.in
safetensors==0.4.5
    # via
    #   accelerate
    #   transformers
scipy==1.14.1
    # via -r requirements.in
sentencepiece==0.2.0
    # via icetk
six==1.16.0
    # via
    #   pathlib2
    #   python-dateutil
    #   rouge
    #   rouge-score
sympy==1.13.3
    # via torch
tabulate==0.9.0
    # via sacrebleu
termcolor==2.4.0
    # via -r requirements.in
thefuzz==0.22.1
    # via -r requirements.in
tiktoken==0.7.0
    # via -r requirements.in
tokenizers==0.19.1
    # via transformers
torch==2.1.0
    # via
    #   -r requirements.in
    #   accelerate
    #   torchvision
torchvision==0.16.0
    # via icetk
tornado==6.4.1
    # via -r requirements.in
tqdm==4.66.5
    # via
    #   huggingface-hub
    #   icetk
    #   nltk
    #   transformers
transformers==4.44.0
    # via -r requirements.in
tritonclient[grpc]==2.49.0
    # via -r requirements.in
typing-extensions==4.12.2
    # via
    #   huggingface-hub
    #   pydantic
    #   pydantic-core
    #   torch
tzdata==2024.2
    # via pandas
urllib3==2.2.3
    # via
    #   geventhttpclient
    #   requests
    #   tritonclient
wcwidth==0.2.13
    # via prettytable
wheel==0.44.0
    # via -r requirements.in
zope-event==5.0
    # via gevent
zope-interface==7.0.3
    # via gevent
```

#### 二、更换mindie版本为1.0.0，并修改mindie源码

**问题现象：**

mindie服务启动报错

```
scheduleParam.maxPrefillBatchSize [50] is out of bound [1]
```

**解决过程：**

最开始考虑是maxPrefillBatchSize设置太大，尝试用maxPrefillBatchSize=2启动，依然报错：

```
scheduleParam.maxPrefillBatchSize [2] is out of bound [1]
```

与开发进一步沟通，确认是该镜像中min dIE版本不支持maxPrefillBatchSize超过1，需要更换mindie为 1.0.0版本。

更换mindie 1.0.0版本并安装后，启动报错为

```
Exception: Sync stream error!
EZ9999: Inner Error!
there is an aivec error exception.
```

经过与研发确认，需要进一步修改mindie安装后`mf_model_wrapper.py`的源码。修改后，服务正常启动。

**解决方法：**

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

#### 三、MA平台使用mindie后，无法启动，报错：daemon killed

**问题现象：**

把镜像push到MA平台后，启动服务总是报错：daemon killed。但使用 `while true; do sleep 1; done` 定住容器后，进入容器内启动，可以正常启动。

**解决过程：**

经过反复研究启动脚本 run_mindie.sh，怀疑是进入容器后，自动触发了bash的 `.bashrc`文件中的一些环境变量。

通过查看在MA上部署启动mindie-server启动日志，发现报错为缺少 torch_npu的pip包，这说明通过MA平台启动时，mindie自动走了ATB的后端。进一步验证了以上猜想。在 run_mindie.sh 增加 MINDIE_LLM_FRAMEWORK_BACKEND 环境变量后，启动正常。

**解决方法：**

在 run_mindie.sh 脚本中添加环境变量  

  ```
  export MINDIE_LLM_FRAMEWORK_BACKEND=MS 
  ```

#### 四、请求响应中首个token是 `<|file_sep|>`而不是`<think>`或普通字符

**问题现象：**

调用mindIE兼容openAI接口 `/v1/chat/completions`后，发现请求响应中首个token是 `<|file_sep|>`而不是`<think>`或普通字符。

**解决过程：**

通过走读[MindFormer的源码qwen2_5_tokenizer.py](https://gitee.com/mindspore/mindformers/blob/br_infer_deepseek_os/research/qwen2_5/qwen2_5_tokenizer.py) 和[魔乐社区qwen2_5_tokenizer.py](https://modelers.cn/models/MindSpore-Lab/QwQ-32B/blob/main/qwen2_5_tokenizer.py)发现，发现261行代码存在bug:

```
261         tool_response_start_token = AddedToken(
262             FILESEP, lstrip=False, rstrip=False, special=True, normalized=False)
263         tool_response_end_token = AddedToken(
264             FILESEP, lstrip=False, rstrip=False, special=True, normalized=False)
265         think_start_token = AddedToken(
266             FILESEP, lstrip=False, rstrip=False, special=True, normalized=False)
267         think_end_token = AddedToken(
268             FILESEP, lstrip=False, rstrip=False, special=True, normalized=False)
```

经过与研发确认该bug，并提单跟踪。

**解决方法：**

将以上[魔乐社区qwen2_5_tokenizer.py](https://modelers.cn/models/MindSpore-Lab/QwQ-32B/blob/main/qwen2_5_tokenizer.py)代码修改为以下代码

```
261         tool_response_start_token = AddedToken(
262             TOOLRESPONSESTART, lstrip=False, rstrip=False, special=True, normalized=False)
263         tool_response_end_token = AddedToken(
264             TOOLRESPONSEEND, lstrip=False, rstrip=False, special=True, normalized=False)
265         think_start_token = AddedToken(
266             THINKSTART, lstrip=False, rstrip=False, special=True, normalized=False)
267         think_end_token = AddedToken(
268             THINKEND, lstrip=False, rstrip=False, special=True, normalized=False)
```

## 其他思考
### 模型支持的最长序列和哪些因素有关?
参考 [长序列特性介绍](https://www.hiascend.com/document/detail/zh/mindie/100/mindiellm/llmdev/mindie_llm0295.html),长序列定义为序列长度超过32K甚至可到达1M级别的文本。长序列特性的主要要求是在输入文本超长的场景下，模型回答的效果及性能也可以同时得到保障。在长序列场景下，由**Attention和KV Cache部分**造成的显存消耗会快速的成倍增长。因此对这部分显存的优化便是长序列特性的关键技术点。其中涉及到诸如KV Cache量化，kv多头压缩，训短推长等关键算法技术。

强行增加序列长度到64K，则报错如下
```
EZ9999: Inner Error!
EZ9999: [PID: 2168] 2025-04-11-07:47:24.603.424 The error from device(chipId:3, dieId:0), serial number is 2, there is an aivec error exception, core id is 22, error code = 0x800000, dump info: pc start: 0x12c0c019d560, current: 0x12c0c0194714, vec error info: 0xc308de2314, mte error info: 0x51031459c0, ifu error info: 0x1714a13144e80, ccu error info: 0xa00a45750817d3b0, cube error info: 0, biu error info: 0, aic error mask: 0x6500020bd00028c, para base: 0x12c100412000.[FUNC:ProcessStarsCoreErrorInfo][FILE:device_error_proc.cc][LINE:1417][THREAD:3463]
        TraceBack (most recent call last):
       The extend info: errcode:(0x800000, 0, 0) errorStr: The DDR address of the MTE instruction is out of range. fixp_error0 info: 0x31459c0, fixp_error1 info: 0x51 fsmId:0, tslot:1, thread:0, ctxid:0, blk:23, sublk:0, subErrType:4.[FUNC:ProcessStarsCoreErrorInfo][FILE:device_error_proc.cc][LINE:1429][THREAD:3463]

    The error from device(chipId:3, dieId:0), serial number is 2, there is an aivec error exception, core id is 12, error code = 0x800000, dump info: pc start: 0x12c0c019d560, current: 0x12c0c0194714, vec error info: 0x7112f7423a, mte error info: 0x51031459c0, ifu error info: 0xf438331e0f40, ccu error info: 0xce7e0d8602810c90, cube error info: 0, biu error info: 0, aic error mask: 0x6500020bd00028c, para base: 0x12c100412000.[FUNC:ProcessStarsCoreErrorInfo][FILE:device_error_proc.cc][LINE:1417][THREAD:3463]
    
    The extend info: errcode:(0x800000, 0, 0) errorStr: The DDR address of the MTE instruction is out of range. fixp_error0 info: 0x314b9a4, fixp_error1 info: 0x57 fsmId:1, tslot:1, thread:0, ctxid:0, blk:9, sublk:0, subErrType:4.[FUNC:ProcessStarsCoreErrorInfo][FILE:device_error_proc.cc][LINE:1429][THREAD:7333]
       
    The error from device(chipId:1, dieId:0), serial number is 4, there is an aivec error exception, core id is 13, error code = 0x800000, dump info: pc start: 0x12c0c019d560, current: 0x12c0c0195ea4, vec error info: 0xe117ffaf80, mte error info: 0x570314a9a4, ifu error info: 0x7cffd73cadbc0, ccu error info: 0x1f405e207d59311f, cube error info: 0, biu error info: 0, aic error mask: 0x6500020bd00028c, para base: 0x12c100410800.[FUNC:ProcessStarsCoreErrorInfo][FILE:device_error_proc.cc][LINE:1417][THREAD:7333]
```
以上报错，说明瓶颈在aivec和mte的硬件或算子。通过多次实验，在不考虑KV Cache的情况，极限序列长度是55K。