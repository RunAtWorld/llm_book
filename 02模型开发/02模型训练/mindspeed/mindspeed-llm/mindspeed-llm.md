# MindSpeed-LLM

## 获取镜像与拉起容器
```
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mindspeed-llm:2025.rc1-arm
```

```bash
docker run -dit --ipc=host --network host --name 'llm_test' --privileged -v /usr/local/Ascend/driver:/usr/local/Ascend/driver  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware  -v /usr/local/sbin/:/usr/local/sbin/ -v /home/:/home/ mindspeed-llm:tag
```