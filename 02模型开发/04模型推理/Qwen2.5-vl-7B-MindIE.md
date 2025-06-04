# Qwen2.5-vl-7B-MindIE

## 下载MindIE的镜像和权重

下载MindIE的镜像
```
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.0.RC1-800I-A2-py311-openeuler24.03-lts
```

下载权重
```
pip install modelscope

cd /home/nvme1n1/weights

mkdir Qwen2.5-VL-7B
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir Qwen2.5-VL-7B

mkdir Qwen2.5-VL-3B
modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct --local_dir Qwen2.5-VL-3B
```

## 启动服务Qwen2.5-VL-7B

### 启动服务
启动 Qwen2.5-VL-7B 容器
```
docker run -it -d --net=host --shm-size=100g \
    --privileged \
    --name Qwen2.5-VL-7B \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /home/nvme1n1/weights/Qwen2.5-VL-7B:/root/Qwen2.5-VL-7B \
    -v /home/nvme1n1/data:/root/data \
    swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.0.RC1-800I-A2-py311-openeuler24.03-lts \
    bash

docker exec -it Qwen2.5-VL-7B bash
cd /root
```

生效环境变量
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/mindie/set_env.sh
source /usr/local/Ascend/atb-models/set_env.sh
```

替换pip包
```
av==13.1.0
huggingface-hub>=0.26.0
tokenizers>=0.21,<0.22
transformers==4.49.0
```

修改服务配置
```bash
MODEL_NAME="Qwen2.5-VL-7B"
MODEL_WEIGHT_PATH="/root/Qwen2.5-VL-7B"

CONFIG_FILE=/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json
sed -i "s/\"modelName\"\s*:\s*\"[^\"]*\"/\"modelName\": \"$MODEL_NAME\"/" $CONFIG_FILE
sed -i "s|\"modelWeightPath\"\s*:\s*\"[^\"]*\"|\"modelWeightPath\": \"$MODEL_WEIGHT_PATH\"|" $CONFIG_FILE
sed -i "s/\"worldSize\"\s*:\s*[0-9]*/\"worldSize\": 1/" "$CONFIG_FILE"
sed -i "s/\"httpsEnabled\"\s*:\s*[a-z]*/\"httpsEnabled\": false/" "$CONFIG_FILE"
sed -i "s/\"backendType\"\s*:\s*\"[^\"]*\"/\"backendType\": \"atb\"/" $CONFIG_FILE

# sed -i "s|\(\"npuDeviceIds\"\s*:\s*\[\[\)[^]]*\(]]\)|\1$0\2|" "$CONFIG_FILE"

grep -i -rn modelName "$CONFIG_FILE"
grep -i -rn modelWeightPath "$CONFIG_FILE"
grep -i -rn worldSize "$CONFIG_FILE"
grep -i -rn npuDeviceIds "$CONFIG_FILE"
grep -i -rn backendType "$CONFIG_FILE"
grep -i -rn npuMemSize "$CONFIG_FILE"

chmod 640 $MODEL_NAME/config.json
```

启动服务

```
cd /usr/local/Ascend/mindie/latest/mindie-service/bin
./mindieservice_daemon
```

### 测试接口
```shell
curl -X POST http://127.0.0.1:1025/v1/chat/completions -d '{
"model": "Qwen2.5-VL-7B",
"messages": [{
    "role": "user",
    "content": [
        {"type": "text", "text": "Explain the contents of the picture."},
        {"type": "image_url", "image_url": "/root/data/cat.png"}
            ]
}],
"max_tokens": 512,
"stream": false
}'
```

- **OpenAI接口**
```shell
curl http://127.0.0.1:1025/v1/chat/completions -d '{
"model": "Qwen2.5-VL-7B",
"messages": [{
    "role": "user",
    "content": [
                {
                    "type": "text",
                    "text": "Explain the contents of the picture."
                },
                {"type": "image_url", "image_url": "/root/data/cat.png"}
            ]
}],
"max_tokens": 512,
"stream": false
}'
```

- **vLLM接口**
```shell
curl localhost:${端口号，与起服务化时config.json中的'port'保持一致}/generate -d '{
    "prompt": [
        {"type": "text", "text": "Explain the contents of the picture."},
        {
            "type": "image_url",
            "image_url": ${图片路径}
        }
    ],
    "max_tokens": 512,
    "do_sample": false,
    "stream": false,
    "model": "qwen2_vl"
}'

## 测试可用镜像

https://poc-resource.obs.cn-south-1.myhuaweicloud.com:443/qwen2.5-vl/qwen2.5-vl-7b_250604.tar?AccessKeyId=HST3UNGQ62AKNPK4B2TL&Expires=1749100987&x-obs-security-token=ggpjbi1ub3J0aC00TxN7ImFjY2VzcyI6IkhTVDNVTkdRNjJBS05QSzRCMlRMIiwibWV0aG9kcyI6WyJ0b2tlbiJdLCJyb2xlIjpbXSwicm9sZXRhZ2VzIjpbXSwidGltZW91dF9hdCI6MTc0OTA5NzAwNTQ5OCwidXNlciI6eyJkb21haW4iOnsiaWQiOiI1MDc3ZmQ5MWEzOTE0NGYzYTM5YWQwODQwYmJiMDlkYiIsIm5hbWUiOiJhc2NlbmRfcG9jX2h1YXdlaV9jbG91ZCJ9LCJpZCI6IjIzZjk1ZTA0ZGY3MzRiZDliOGJlZWRjYTlmMjE0MTRhIiwibmFtZSI6InRlbGVfcG9jIiwicGFzc3dvcmRfZXhwaXJlc19hdCI6IiIsInVzZXJfdHlwZSI6MTZ9fR6EqbwDt2i_zMJp9dmma-Lkcs0tq-smY-33oekyB1XSsJOQ1-_oHpvwIkd5b9UecyYXOGkiy14_fc9b2piZKfCvnYlfyBpToay0oszprfK885H5nLl7GgL5HKuFeG2hpSxZ1wtMDe57odvH_JCacPlDufVgHY-hn6akYi9pL_7vbcNPKpePZh7oADpTidejB6dDoiOh_s-RqKCe57y1-GPFluuvKIkYwdY6VDNkSs8dA66LndDsxpG_DBcGlbBGD7ieHlioJfW2Z7sQF_DY0HmXaC5Fdu6fJIIQxNJDZuUwPlyQP-5jioquQ66BhjuHYiU13vYF8aeGNAsFnFAZmDY%3D&Signature=KrE/ehgb934mt142FfYmcqVZdmM%3D


docker login -u cn-north-4@HST3W3XVQTBZ9GK4O4EA -p 415a2c08786c4bcddaf1900f681b0bf7eb3aaad6bbbfdf59c221fcb242dc4a9a swr.cn-north-4.myhuaweicloud.com
docker pull swr.cn-north-4.myhuaweicloud.com/tomy/qwen2.5-vl-7b:250604