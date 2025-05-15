# QWQ-ATB_MindIE版
> 官方说明：https://modelscope.cn/models/Qwen/QwQ-32B

## 镜像

> 镜像zip包：https://mindx.sdk.obs.cn-north-4.myhuaweicloud.com/MindIE/docker/mindie_2.0.T3-20250212-800I-A2-py311-openeuler24.03-lts-aarch64.tar.gz

下载镜像使用下面命令完成加载镜像

```shelll
docker load -i mindie:1.0.0-800I-A2-py311-openeuler24.03-lts(下载的镜像名称与标签)
```

镜像中各组件版本配套如下：

| 组件       | 版本   |
| :--------- | :----- |
| MindIE     | 1.0.0  |
| CANN       | 8.0.0  |
| PTA        | 6.0.0  |
| MindStudio | 7.0.0  |
| HDK        | 24.1.0 |

## 模型权重

下载模型权重： https://modelers.cn/models/Models_Ecosystem/QwQ-32B

## 创建容器

目前提供的 MindIE 镜像预置了 QwQ-32B 模型推理脚本，无需再额外下载魔乐仓库承载的模型适配代码，直接新建容器即可。

执行以下启动命令（参考）：如果您使用的是 root 用户镜像（例如从 Ascend Hub 上取得），并且可以使用特权容器，请使用以下命令启动容器：

```shell
docker run -it -d --net=host --shm-size=10g \
    --privileged \
    --name qwq32b \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /home/data/qwq32b_mindie/QwQ-32B:/weights/qwq32b_mindie/QwQ-32B \
    mindie:2.0.T3-20250212-800I-A2-py311-openeuler24.03-lts-aarch64 bash
```

如果您希望使用自行构建的普通用户镜像，并且规避容器相关权限风险，可以使用以下命令指定用户与设备：

```shell
docker run -it -d --net=host --shm-size=1g \
    --user mindieuser:<HDK-user-group> \
    --name <container-name> \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /path-to-weights:/path-to-weights:ro \
    mindie:1.0.0-800I-A2-py311-openeuler24.03-lts bash
```

注意，以上启动命令仅供参考，请根据需求自行修改再启动容器，尤其需要注意：

1. --user，如果您的环境中 HDK 是通过普通用户安装（例如默认的HwHiAiUser，可以通过id HwHiAiUser命令查看该用户组 ID），请设置好对应的用户组，例如用户组 1001 可以使用 HDK，则--user mindieuser:1001，镜像中默认使用的是用户组 1000。如果您的 HDK 是由 root 用户安装，且指定了--install-for-all参数，则无需指定--user参数。

2. 设定容器名称--name与镜像名称，800I A2 和 300I DUO 各自使用对应版本的镜像，例如 800I A2 服务器使用mindie:1.0.0-py3.11-800I-A2-aarch64-Ubuntu22.04。

3. 设定想要使用的卡号--device。

4. 设定权重挂载的路径，-v /path-to-weights:/path-to-weights:ro，注意，如果使用普通用户镜像，权重路径所属应为镜像内默认的 1000 用户，且权限可设置为 750。可使用以下命令进行修改：

    ```
    chown -R 1000:1000 /path-to-weightschmod -R 755 /path-to-weights
    ```

5. 在普通用户镜像中，注意所有文件均在 /home/mindieuser 下，请勿直接挂载 /home 目录，以免宿主机上存在相同目录，将容器内文件覆盖清除。 

进入容器

```shell
docker exec -it qwq32b bash
```

## 执行推理
### 纯模型推理

#### 对话测试
> 如果需要设置可见卡 export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
进入 llm_model 路径

`ATB_SPEED_HOME_PATH` 默认 `/usr/local/Ascend/llm_model`,以情况而定

执行对话测试

```shell
cd $ATB_SPEED_HOME_PATH
torchrun --nproc_per_node 2 \
         --master_port 20037 \
         -m examples.run_pa \
         --model_path /weights/qwq32b_mindie/QwQ-32B \
         --max_output_length 20
```

### 服务化推理

- 打开配置文件

```shell
vim /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
```

- 更改配置文件

```json
{
...
  "ServerConfig": {
      "_comment": "服务器相关配置",
      "port": 1040,           // 自定义主服务端口
      "managementPort": 1041, // 自定义管理端口
      "metricsPort": 1042,    // 自定义指标端口
      "httpsEnabled": false  //关闭https验证
    },
  "BackendConfig": {
    "_comment": "后端服务相关配置",
    "npuDeviceIds": [[0, 1, 2, 3]],
    "ModelDeployConfig": {
      "truncation": false,
      "ModelConfig": [
        {
          "_comment": "模型配置项",
          "modelName": "QwQ-32B", //修改模型名称
          "modelWeightPath": "/weights/qwq32b_mindie/QwQ-32B", //修改模型权重路径
          "worldSize": 4
        }
      ]
    }
  }
}
```

- 拉起服务化

```shell
cd /usr/local/Ascend/mindie/latest/mindie-service/bin
./mindieservice_daemon
```
后台启动
```shell
cd /usr/local/Ascend/mindie/latest/mindie-service/bin
nohup ./mindieservice_daemon > /root/service.log 2>&1 &
```

终止后台进程
```
pkill -9 mind
pkill -9 python
```

- 测试OpenAI 接口

```shell
curl -X POST 127.0.0.1:1040/v1/chat/completions \
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

> 注 1: 服务化推理的更多信息请参考[MindIE Service 用户指南](https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0001.html)
>
> 注 2：仅在 Atlas 800I A2 64G 服务器下使用 4 卡进行 BF16 推理测试，尚未在其他条件下进行测试。


## 参考
1. [QwQ-32B昇腾魔乐社区部署](https://modelers.cn/models/Models_Ecosystem/QwQ-32B)

## FAQ
1. 关于权限:在在宿主机上修改模型权重权限为 640
  ```
  chmod -R 640 /home/data/qwq32b/QwQ-32B  #在宿主机上修改
  ```

  在容器中修改配置文件为 640
  ```
  chmod 640 /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
  ```

## 性能测试
> 参考 [性能测试](https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0012.html)

### 4卡B3 gsm8k测试结果
为了减少测试时间，这里抽取gsm8k的前200条数据测试。
```
export MINDIE_LOG_TO_STDOUT="benchmark:1; client:1"
nohup benchmark \
--DatasetPath "/weights/qwq32b_mindie/QwQ-32B/gsm8k-200" \
--DatasetType "gsm8k" \
--ModelName "QwQ-32B" \
--ModelPath "/weights/qwq32b_mindie/QwQ-32B" \
--TestType client \
--Http http://127.0.0.1:1040 \
--ManagementHttp http://127.0.0.2:1041 \
--Concurrency 10 \
--DoSampling False \
--MaxOutputLen 512 > /weights/qwq32b_mindie/QwQ-32B/atb_mindie_gsm8k_benchmark-200.log 2>&1 &
tail -f /weights/qwq32b_mindie/QwQ-32B/atb_mindie_gsm8k_benchmark-200.log
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
        "ipAddress" : "127.0.0.1",
        "managementIpAddress" : "127.0.0.2",
        "port" : 1040,
        "managementPort" : 1041,
        "metricsPort" : 1042,
        "allowAllZeroIpListening" : false,
        "maxLinkNum" : 1000,
        "httpsEnabled" : false,
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
        "npuDeviceIds" : [[4,5,6,7]],
        "tokenizerProcessNumber" : 8,
        "multiNodesInferEnabled" : false,
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
            "maxSeqLen" : 2560,
            "maxInputTokenLen" : 2048,
            "truncation" : false,
            "ModelConfig" : [
                {
                    "modelInstanceType" : "Standard",
                    "modelName" : "QwQ-32B",
                    "modelWeightPath" : "/weights/qwq32b_mindie/QwQ-32B",
                    "worldSize" : 4,
                    "cpuMemSize" : 5,
                    "npuMemSize" : -1,
                    "backendType" : "atb",
                    "trustRemoteCode" : false
                }
            ]
        },

        "ScheduleConfig" :
        {
            "templateType" : "Standard",
            "templateName" : "Standard_LLM",
            "cacheBlockSize" : 128,

            "maxPrefillBatchSize" : 50,
            "maxPrefillTokens" : 8192,
            "prefillTimeMsPerReq" : 150,
            "prefillPolicyType" : 0,

            "decodeTimeMsPerReq" : 50,
            "decodePolicyType" : 0,
            "maxBatchSize" : 200,
            "maxIterTimes" : 512,
            "maxPreemptCount" : 0,
            "supportSelectBatch" : false,
            "maxQueueDelayMicroseconds" : 5000
        }
    }
}
```

gsm8k_benchmark 测试结果

```
[2025-03-17 17:32:16.757+08:00] [12089] [281473357974080] [benchmark] [INFO] [output.py:115]
The BenchMark test performance metric result is:
+---------------------+-----------------+-----------------+-----------------+-----------------+-----------------+-----------------+-----------------+------+
|              Metric |         average |             max |             min |             P75 |             P90 |         SLO_P90 |             P99 |    N |
+---------------------+-----------------+-----------------+-----------------+-----------------+-----------------+-----------------+-----------------+------+
|      FirstTokenTime |      64.7768 ms |     848.3901 ms |      49.9074 ms |      69.1345 ms |       74.367 ms |       74.367 ms |      93.7642 ms | 8792 |
|          DecodeTime |      27.0037 ms |     788.9588 ms |       0.0176 ms |      27.0996 ms |      28.3229 ms |      27.6082 ms |      63.7563 ms | 8792 |
|      LastDecodeTime |      27.5678 ms |     101.4566 ms |            0 ms |      28.0333 ms |      29.2665 ms |      29.2665 ms |      63.0869 ms | 8792 |
|       MaxDecodeTime |      81.6564 ms |     788.9588 ms |      28.2619 ms |      83.5511 ms |      99.0384 ms |      99.0384 ms |     172.0248 ms | 8792 |
|        GenerateTime |   12176.1428 ms |    15100.317 ms |      53.4904 ms |   13929.2935 ms |    14138.473 ms |    14138.473 ms |   14394.8389 ms | 8792 |
|         InputTokens |         59.8964 |             212 |              12 |            71.0 |            89.0 |            89.0 |           128.0 | 8792 |
|     GeneratedTokens |        449.2916 |             512 |               1 |           512.0 |           512.0 |           512.0 |           512.0 | 8792 |
| GeneratedTokenSpeed | 36.7786 token/s | 40.9125 token/s | 11.9284 token/s | 37.3645 token/s | 37.5995 token/s | 37.5995 token/s | 38.0678 token/s | 8792 |
| GeneratedCharacters |       1492.5876 |            4086 |               0 |          1779.0 |          1930.9 |          1930.9 |         2179.09 | 8792 |
|           Tokenizer |       1.1126 ms |      14.7884 ms |       0.2434 ms |       1.3033 ms |        1.662 ms |        1.662 ms |       2.8289 ms | 8792 |
|         Detokenizer |       4.3607 ms |      13.1407 ms |       0.0107 ms |       4.9663 ms |       4.9882 ms |       4.9882 ms |       5.0328 ms | 8792 |
|  CharactersPerToken |          3.3221 |               / |               / |               / |               / |               / |               / | 8792 |
|  PostProcessingTime |            0 ms |            0 ms |            0 ms |            0 ms |            0 ms |            0 ms |            0 ms | 8792 |
|         ForwardTime |            0 ms |            0 ms |            0 ms |            0 ms |            0 ms |            0 ms |            0 ms | 8792 |
|    PrefillBatchsize |          1.0258 |              10 |               1 |             1.0 |             1.0 |             1.0 |             2.0 | 8792 |
|    DecoderBatchsize |          9.9792 |              20 |               1 |            10.0 |            10.0 |               / |            10.0 | 8792 |
|       QueueWaitTime |     893.4933 μs |       175689 μs |           18 μs |         92.0 μs |        114.0 μs |            / μs |      37317.0 μs | 8792 |
+---------------------+-----------------+-----------------+-----------------+-----------------+-----------------+-----------------+-----------------+------+
[2025-03-17 17:32:16.758+08:00] [12089] [281473357974080] [benchmark] [INFO] [output.py:121]
The BenchMark test common metric result is:
+------------------------+--------------------------------------+
|          Common Metric |                                Value |
+------------------------+--------------------------------------+
|            CurrentTime |                  2025-03-17 17:32:16 |
|            TimeElapsed |                         10714.4325 s |
|             DataSource | /weights/qwq32b_mindie/QwQ-32B/gsm8k |
|                 Failed |                            0( 0.0% ) |
|               Returned |                       8792( 100.0% ) |
|                  Total |                       8792[ 100.0% ] |
|            Concurrency |                                   10 |
|              ModelName |                              QwQ-32B |
|                   lpct |                            1.0815 ms |
|             Throughput |                         0.8206 req/s |
|          GenerateSpeed |                     368.6777 token/s |
| GenerateSpeedPerClient |                      36.8678 token/s |
|               accuracy |                                    / |
+------------------------+--------------------------------------+
```

