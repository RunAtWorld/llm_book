# MindSpeed-RL实践

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
