# 在NPU上创建容器



A3 创建容器

```
export IMAGE=quay.io/ascend/cann:8.5.0-a3-ubuntu22.04-py3.11
docker run --rm \
--name cann_85_lpf \
--shm-size=1g \
--device /dev/davinci14 \
--device /dev/davinci15 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-it $IMAGE bash
```

测试

