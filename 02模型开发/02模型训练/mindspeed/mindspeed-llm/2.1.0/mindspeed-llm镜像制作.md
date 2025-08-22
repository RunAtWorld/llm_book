# Mindspeed-LLM镜像制作教程

> 参考 https://gitee.com/ascend/MindSpeed-LLM/blob/2.1.0/docs/pytorch/install_guide.md

镜像制作流程如下

```mermaid
graph LR
    1获取镜像 --> 2启动容器 --> 3.1重装CANN --> 3.2重装PTA --> 3.3安装MindSpeed-LLM --> 4保存镜像
```

## 1.获取镜像

当前基础镜像为

```
registry-cn-guian.midea.com/aios/a3_llm_release:v2.1.3
```

## 2.启动容器

```
docker run -it -u root --privileged --ipc=host \
    --name 'qwen3_img' \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    --device=/dev/davinci8 \
    --device=/dev/davinci9 \
    --device=/dev/davinci10 \
    --device=/dev/davinci11 \
    --device=/dev/davinci12 \
    --device=/dev/davinci13 \
    --device=/dev/davinci14 \
    --device=/dev/davinci15 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/sbin/:/usr/local/sbin/ \
    -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
    -v /apps/shared_storage/hw_test/hw:/apps/hw \
    registry-cn-guian.midea.com/aios/a3_llm_release:v2.1.3  bash
```

> * 宿主机挂载目录映射为：/apps/shared_storage/hw_test/hw，容器内对应目录为 /apps/hw
>
> * 相关组件已经下载好放到共享存储目录 /apps/shared_storage/hw_test/hw/pkg/train25.1.0

## 3.安装组件

相关组件已经下载好在以下目录：

```
/apps/shared_storage/hw_test/hw/pkg/train25.1.0
```

### 3.1 CANN安装

下载[CANN](https://www.hiascend.com/developer/download/community/result?module=cann)的`cann-toolkit`、`cann-kernel`和`cann-nnal`。选择参考[CANN安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/softwareinst/instg/instg_quick.html)或执行以下命令安装：

```shell
# 因为版本迭代，包名存在出入，根据实际修改
chmod +x Ascend-cann-toolkit_8.2.RC1_linux-aarch64.run
./Ascend-cann-toolkit_8.2.RC1_linux-aarch64.run --install
chmod +x Atlas-A3-cann-kernels_8.2.RC1_linux-aarch64.run
./Atlas-A3-cann-kernels_8.2.RC1_linux-aarch64.run --install
chmod +x Ascend-cann-nnal_8.2.RC1_linux-aarch64.run
./Ascend-cann-nnal_8.2.RC1_linux-aarch64.run --install
# 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### 3.2 安装PTA

准备[torch_npu](https://www.hiascend.com/developer/download/community/result?module=pt)和[apex](https://gitee.com/ascend/apex)，参考[Ascend Extension for PyTorch 配置与安装](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0001.html)执行以下命令安装：

```shell
# 安装torch和torch_npu 构建参考 https://gitee.com/ascend/pytorch/releases
pip install torch-2.6.0+cpu-cp310-cp310-manylinux_2_28_aarch64.whl
pip install torch_npu-2.6.0-cp310-cp310-manylinux_2_28_aarch64.whl

# apex for Ascend 构建参考 https://gitee.com/ascend/apex
pip install apex-0.1+ascend-cp310-cp310-linux_aarch64.whl
```

### 3.3 安装MindSpeed-LLM

```shell
# 使能环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 安装MindSpeed加速库
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 2c085cc9  # checkout commit from MindSpeed core_r0.8.0 in 2025.04.01
pip install -r requirements.txt 
pip3 install -e .
cd ..

# 准备MindSpeed-LLM及Megatron-LM源码
git clone https://gitee.com/ascend/MindSpeed-LLM.git 
git clone https://github.com/NVIDIA/Megatron-LM.git  # megatron从github下载，请确保网络能访问
cd Megatron-LM
git checkout core_r0.8.0
cp -r megatron ../MindSpeed-LLM/
cd ../MindSpeed-LLM
git checkout 2.1.0

pip install -r requirements.txt  # 安装其余依赖库
```
>注意: qwen3系列模型依赖transformers 4.51.0,  需要在环境配置完成后手动执行pip install transformers==4.51.0

## 4.保存镜像

以后组件安装后，保存镜像

```
docker commit -a "MD" -m "add new cann"  qwen3_img  registry-cn-guian.midea.com/aios/a3_llm:v2.1.4
docker push registry-cn-guian.midea.com/aios/a3_llm:v2.1.4
```

