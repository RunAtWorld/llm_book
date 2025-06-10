# BGE模型
> 模型链接 https://modelscope.cn/models/BAAI/bge-m3

## 下载权重并修改配置
安装modelscope
```
pip install modelscope -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

下载权重
```
modelscope download --model BAAI/bge-m3
```

修改config配置
```
"_name_or_path": "/home/HwHiAiUser/bge-m3",
"auto_map": {
    "AutoModel": "/home/HwHiAiUser/Ascend/llm_model/examples/models/embedding/xlm_roberta--modeling_xlm_roberta.XLMRobertaModel"
  }
```

## 创建容器
启动容器
```
docker run -itd \
--security-opt seccomp=unconfined \
-v /sys/fs/cgroup:/sys/fs/cgroup \
--user=1001:1000 \
--shm-size=4G \
--network=host  \
--name=bge \
--device=/dev/davinci5 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/sbin/:/usr/local/sbin/ \
-v /root/lpf/bge-m3:/home/HwHiAiUser/bge-m3 \
--entrypoint /bin/bash \
cmb_bge:1.0.0
```
> 启动容器前，要先修改 `/home/HwHiAiUser/bge/bge-m3` 的文件属主为 HiHwHiAiUser, 使用命令： `chown -R 1001:1000 /root/lpf/bge-m3`
> 
> 如果本地没有这个1000的用户，则创建一个用户: `groupadd -g 1001 ma1  && useradd -d /home/ma1 -m -u 1001 -g 1000 -s /bin/bash ma1`
>
> bge-m3目录的权限设置为750 `chmod -R 750 /root/lpf/bge-m3`

## 启动服务
进入容器
```
docker exec -it bge bash
```

进入容器后启动服务
```
export PATH=$PATH:~/.cargo/bin/
# 设置TEI运行计算卡id号与模型后端 
export TEI_NPU_DEVICE=0  # 按需设置计算卡id,若有只有一张卡，就设置为0
export TEI_NPU_BACKEND=atb  # 按需选择mindietorch或atb
# 本地模型权重路径或在Huggingface代码仓中的模型id
# model_path_embedding=/home/HwHiAiUser/bge-m3

# 以下启动方式及参数名与原生TEI一致，请按需选择拉起Embedding或Reranker模型
# Embedding模型
text-embeddings-router --model-id /home/HwHiAiUser/bge-m3  --dtype float16 --pooling cls --max-concurrent-requests 2048 --max-batch-requests 2048 --max-batch-tokens 1100000 --max-client-batch-size 256 --port 12347
```

API访问测试
```
curl 127.0.0.1:12347/embed \ 
-X POST \
-d '{"inputs": ["What is Deep Learning?"]}' \     
-H 'Content-Type: application/json'
```

## FAQ

### 1. 启动容器时，需要根据容器中HwHiAiUser用户和用户组的，配置对应的参数
镜像中HwHiAiUser的属组为： 1001:1000

```
--security-opt seccomp=unconfined \
-v /sys/fs/cgroup:/sys/fs/cgroup \
--user=1001:1000 \
```

### 2. 挂载 bge-m3 目录时，文件目录权限和属主必须对应

启动容器前，要先修改 `/home/HwHiAiUser/bge/bge-m3` 的文件属主为 HiHwHiAiUser, 使用命令： `chown -R 1000:1001 /root/lpf/bge-m3`

如果本地没有这个1000的用户，则创建一个用户: `groupadd -g 1001 ma1  && useradd -d /home/ma1 -m -u 1001 -g 1001 -s /bin/bash ma1`

bge-m3目录的权限设置为750 `chmod -R 750 /root/lpf/bge-m3`