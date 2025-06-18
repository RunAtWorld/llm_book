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

### PTA安装

```shell
# 安装torch和torch_npu
wget https://download.pytorch.org/whl/cpu/torch-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install torch-2.5.1-cp310-cp310-*.whl
pip install torch_npu-2.5.1.*.manylinux2014_aarch64.whl

# apex for Ascend 构建参考 https://gitee.com/ascend/apex
pip install --upgrade apex-0.1+ascend-*.whl
```

[验证 torch_npu 安装是否成功](https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/configandinstg/instg/insg_0001.html)
```
python3 -c "import torch;import torch_npu;print(torch_npu.npu.is_available())"
```

### vllm及相关依赖安装：
> 注：环境中需要安装git，因为vllm的安装过程依赖git
> 安装包在: /usr/local/python3.10.17/lib/python3.10/site-packages
```shell
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout -b v0.7.3
pip install -r requirements-build.txt
VLLM_TARGET_DEVICE=empty pip install .
```

> 如果依赖安装较慢，换源
> pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
> pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
> pip config set global.extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple/
> 
> ~/.config/pip/pip.conf文件内容如下
> [global]
> index-url = https://pypi.tuna.tsinghua.edu.cn/simple
> extra-index-url = https://mirrors.aliyun.com/pypi/simple/
### vllm_ascend安装
```shell
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout -b v0.7.3-dev
git checkout 0713836e95fe993feefe334945b5b273e4add1f1
pip install -e .
```

### 高性能内存库 jemalloc 安装
为了确保 Ray 进程能够正常回收内存，需要安装并使能 jemalloc 库进行内存管理。
#### Ubuntu 操作系统
通过操作系统源安装jemalloc（注意： 要求ubuntu版本>=20.04）：
```shell
sudo apt install libjemalloc2
```
在启动任务前执行如下命令通过环境变量导入jemalloc：
```shell
# arm64架构
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2 $LD_PRELOAD
# x86_64架构
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 $LD_PRELOAD
```

#### OpenEuler 操作系统

执行如下命令重操作系统源安装jemalloc
```shell
yum install jemalloc
```
如果上述方法无法正常安装，可以通过源码编译安装
前往jamalloc官网下载最新稳定版本，官网地址:https://github.com/jemalloc/jemalloc/releases/
```shell
tar -xvf jemalloc-{version}.tar.bz2
cd jemalloc-{version}
./configure --prefix=/usr/local
make
make install
```
在启动任务前执行如下命令通过环境变量导入jemalloc：
```shell
export LD_PRELOAD=/usr/local/lib/libjemalloc.so.2 $LD_PRELOAD
```

> 如以上安装过程出现错误，可以通过提出issue获得更多解决建议。

## 准备源码
```shell
git clone https://gitee.com/ascend/MindSpeed-RL.git 

git clone https://gitee.com/ascend/MindSpeed.git 
cd MindSpeed
git checkout 0dfa0035ec54d9a74b2f6ee2867367df897299df  # 参考MindSpeed-LLM依赖版本
pip install -r requirements.txt 
cp -r mindspeed ../MindSpeed-RL/
cd ..

git clone https://github.com/NVIDIA/Megatron-LM.git  # Megatron从github下载，请确保网络能访问
cd Megatron-LM
git checkout -b core_r0.8.0
cp -r megatron ../MindSpeed-RL/
cd ..

git clone https://gitee.com/ascend/MindSpeed-LLM.git
cd MindSpeed-LLM
git checkout 421ef7bcb83fb31844a1efb688cde71705c0526e
cp -r mindspeed_llm ../MindSpeed-RL/
cd ..

cd ./MindSpeed-RL
pip install -r requirements.txt
pip install antlr4-python3-runtime==4.7.2 --no-deps 
```

## 保存Docker镜像

保存镜像

```
docker commit -m "mindspeed-rl env" -a "lpf"  llm_rl mindspeed-rl:0618
```

镜像save为镜像.tar

```
docker save -o mindspeed-rl_0618.tar mindspeed-rl:0618
```

