# 安装910A-ubuntu20.04
# 安装操作系统
ubuntu20.04: https://mirrors.aliyun.com/oldubuntu-releases/releases/20.04/?spm=a2c6h.25603864.0.0.10eb7ff35r3X2f
安装指南: https://support.huawei.com/enterprise/zh/doc/EDOC1100258049/47513e18?idPath=23710424|251366513|22892968|252309113|250702818

配置网络

查询网卡
```
ip addr
```

网络配置
```
network:
  version: 2
  renderer: networkd
  ethernets:
    enp189s0f0:
      dhcp4: no
      addresses: [168.170.140.82/16]
      gateway4: 168.170.1.1
      nameservers:
          addresses: [114.114.114.114]
```

生效网络
```
netplan apply
```

更换apt的源
```
cp -a /etc/apt/sources.list /etc/apt/sources.list.bak
sed -i "s@http://.*archive.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list
sed -i "s@http://.*security.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list
```

apt的源文件
```
deb https://repo.huaweicloud.com/ubuntu-ports/ bionic main restricted universe multiverse
deb-src https://repo.huaweicloud.com/ubuntu-ports/ bionic main restricted universe multiverse

deb https://repo.huaweicloud.com/ubuntu-ports/ bionic-security main restricted universe multiverse
deb-src https://repo.huaweicloud.com/ubuntu-ports/ bionic-security main restricted universe multiverse

deb https://repo.huaweicloud.com/ubuntu-ports/ bionic-updates main restricted universe multiverse
deb-src https://repo.huaweicloud.com/ubuntu-ports/ bionic-updates main restricted universe multiverse

deb https://repo.huaweicloud.com/ubuntu-ports/ bionic-backports main restricted universe multiverse
deb-src https://repo.huaweicloud.com/ubuntu-ports/ bionic-backports main restricted universe multiverse

## Not recommended
# deb https://repo.huaweicloud.com/ubuntu-ports/ bionic-proposed main restricted universe multiverse
# deb-src https://repo.huaweicloud.com/ubuntu-ports/ bionic-proposed main restricted universe multiverse
```


# 安装驱动
> 参考：https://support.huawei.com/enterprise/zh/doc/EDOC1100349467/2645a51f?idPath=23710424|251366513|22892968|252764743

安装依赖
```
apt install -y make 
apt install -y gcc
apt install -y dkms 
apt install -y linux-header
apt install -y net-tools
```

以root用户登录服务器。
执行如下命令，创建运行用户。
```
groupadd HwHiAiUser
useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
```

执行如下命令，切换至root用户。
```
su - root
```

执行如下命令，进入软件包所在路径（如“/opt”）。
```
cd /opt
```

执行如下命令，增加软件包的可执行权限。
```
chmod +x Ascend-hdk-910-npu-driver_23.0.0_linux-aarch64.run 
```

执行如下命令，校验run安装包的一致性和完整性。
```
./Ascend-hdk-910-npu-driver_23.0.0_linux-aarch64.run  --check
```

若出现如下回显信息，表示软件包校验成功。
Verifying archive integrity...  100%   SHA256 checksums are OK. All good.

执行如下命令，完成驱动安装，软件包默认安装路径为“/usr/local/Ascend”。
```
./Ascend-hdk-910-npu-driver_23.0.0_linux-aarch64.run  --full  --install-for-all
```

如果指定root用户为运行用户，则需要与--install-for-all参数配合使用，如下所示，该场景下权限控制可能存在安全风险。`--install-username=root --install-usergroup=root --install-for-all`

# 安装固件
> 参考：https://support.huawei.com/enterprise/zh/doc/EDOC1100349467/2645a51f?idPath=23710424|251366513|22892968|252764743


执行如下命令，切换至root用户。
```
su - root
```
执行如下命令，进入软件包所在路径（如“/opt”）。
```
cd /opt
```

执行如下命令，增加软件包的可执行权限。
```
chmod +x *.run
```

执行如下命令，校验run安装包的一致性和完整性。
```
./Ascend-hdk-910-npu-firmware_7.1.0.3.220.run --check
```

出现如下回显信息，表示软件包校验成功。

Verifying archive integrity...  100%   SHA256 checksums are OK. All good.

执行如下命令，完成安装。

```
./Ascend-hdk-910-npu-firmware_7.1.0.3.220.run  --full
```

若系统出现如下关键回显信息，表示固件安装成功，并根据提示信息决定是否立即重启系统。

Firmware package installed successfully!


执行如下命令，查看芯片固件版本号。
```
/usr/local/Ascend/driver/tools/upgrade-tool --device_index -1 --component -1 --version
```
若与固件软件包版本号一致，则说明安装成功。

# 安装CANN
> 参考: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/700alpha003/softwareinstall/instg/instg_000002.html


安装依赖

```
sudo apt-get install -y gcc g++ make cmake zlib1g zlib1g-dev openssl libsqlite3-dev libssl-dev libffi-dev unzip pciutils net-tools libblas-dev gfortran libblas3
```
检查系统是否安装满足版本要求的python开发环境（具体要求请参见依赖列表，此步骤以环境上需要使用python 3.7.x为例进行说明）。
执行命令python3 --version，如果返回信息满足python版本要求，则直接进入下一步。

否则可参考如下方式安装python3.7.5。

使用wget下载python3.7.5源码包，可以下载到安装环境的任意目录，命令为：
```
wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz
```
进入下载后的目录，解压源码包，命令为：
```
tar -zxvf Python-3.7.5.tgz
```
进入解压后的文件夹，执行配置、编译和安装命令：
```
cd Python-3.7.5
./configure --prefix=/usr/local/python3.7.5 --enable-loadable-sqlite-extensions --enable-shared
make
make install
```
其中“--prefix”参数用于指定python安装路径，用户根据实际情况进行修改。“--enable-shared”参数用于编译出libpython3.7m.so.1.0动态库。“--enable-loadable-sqlite-extensions”参数用于加载libsqlite3-dev依赖。

本手册以--prefix=/usr/local/python3.7.5路径为例进行说明。执行配置、编译和安装命令后，安装包在/usr/local/python3.7.5路径，libpython3.7m.so.1.0动态库在/usr/local/python3.7.5/lib/libpython3.7m.so.1.0路径。

设置python3.7.5环境变量。
```
#用于设置python3.7.5库文件路径
export LD_LIBRARY_PATH=/usr/local/python3.7.5/lib:$LD_LIBRARY_PATH
#如果用户环境存在多个python3版本，则指定使用python3.7.5版本
export PATH=/usr/local/python3.7.5/bin:$PATH
```
通过以上export方式设置环境变量，该种方式设置的环境变量只在当前窗口有效。您也可以通过将以上命令写入~/.bashrc文件中，然后执行source ~/.bashrc命令，使上述环境变量永久生效。注意如果后续您有使用环境上其他python版本的需求，则不建议将以上命令写入到~/.bashrc文件中。

安装完成之后，执行如下命令查看安装版本，如果返回相关版本信息，则说明安装成功。
```
python3 --version
pip3 --version
```

安装前请先使用pip3 list命令检查是否安装相关依赖，若已经安装，则请跳过该步骤；若未安装，则安装命令如下

```
pip3 install attrs
pip3 install numpy
pip3 install decorator
pip3 install sympy
pip3 install cffi
pip3 install pyyaml
pip3 install pathlib2
pip3 install psutil
pip3 install protobuf
pip3 install scipy
pip3 install requests
pip3 install absl-py
```

> 可以使用aliyun的源
> ```
> pip3 config set global.index-url http://mirrors.huaweicloud.com/repository/pypi/simple
> pip3 config set trusted-host mirrors.huaweicloud.com
> ```

~/.pip/pip.conf
```
[global]
index-url = https://mirrors.huaweicloud.com/repository/pypi/simple
trusted-host = mirrors.huaweicloud.com
timeout = 120
```

加对软件包的可执行权限。
```
chmod +x 软件包名.run
```
软件包名.run表示开发套件包Ascend-cann-toolkit_{version}_linux-{arch}.run，请根据实际包名进行替换。

执行如下命令校验软件包安装文件的一致性和完整性。
```
./Ascend-cann-toolkit_7.0.RC1_linux-aarch64.run --check
```
执行以下命令安装软件（以下命令支持--install-path=<path>等参数，具体参数说明请参见参数说明）。
```
./Ascend-cann-toolkit_7.0.RC1_linux-aarch64.run --install
```

静默安装
```
Ascend310P-opp_kernel-7.5.0.1.129-linux.aarch64.run --full --quiet --nox11
```

如果用户未指定安装路径，则软件会安装到默认路径下，默认安装路径如下。

root用户：`/usr/local/Ascend` ; 非root用户：`${HOME}/Ascend`, 其中`${HOME}`为当前用户目录。

用户需签署华为企业业务最终用户许可协议（EULA）后进入安装流程
```
#配置为英文
export LANG=en_US.UTF-8
```
安装完成后，若显示如下信息，则说明软件安装成功：
xxx install success
xxx表示安装的实际软件包名。


配置环境变量，使用前要导入环境变量
```
-  To take effect for all users, you can add "source /usr/local/Ascend/ascend-toolkit/set_env.sh" to /etc/profile.
-  To take effect for current user, you can exec command below: source /usr/local/Ascend/ascend-toolkit/set_env.sh or add "source /usr/local/Ascend/ascend-toolkit/set_env.sh" to ~/.bashrc.
```

CANN软件提供进程级环境变量设置脚本，供用户在进程中引用，以自动完成环境变量设置。用户进程结束后自动失效。示例如下（以root用户默认安装路径为例）：

安装toolkit包后配置变量
```
. /usr/local/Ascend/ascend-toolkit/set_env.sh
```
用户也可以通过修改~/.bashrc文件方式设置永久环境变量，操作如下：
以运行用户在任意目录下执行vi ~/.bashrc命令，打开.bashrc文件，在文件最后一行后面添加上述内容。
执行:wq!命令保存文件并退出。
执行source ~/.bashrc命令使其立即生效。

```
-  To take effect for all users, you can add "source /usr/local/Ascend/ascend-toolkit/set_env.sh" to /etc/profile.
-  To take effect for current user, you can exec command below: source /usr/local/Ascend/ascend-toolkit/set_env.sh or add "source /usr/local/Ascend/ascend-toolkit/set_env.sh" to ~/.bashrc.
```

# 安装conda 

安装miniconda
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# rm -rf ~/miniconda3/miniconda.sh

# 如需初始化bash或zsh
# ~/miniconda3/bin/conda init bash
# ~/miniconda3/bin/conda init zsh
```

创建环境
```
conda create -n ms2.2py39 python=3.9
conda activate ms2.2py39
pip3 install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0/MindSpore/unified/aarch64/mindspore-2.2.0-cp39-cp39-linux_aarch64.whl
```

# 安装Pytorch（非必选）
```
apt-get install -y patch build-essential libbz2-dev libreadline-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev liblzma-dev m4 dos2unix libopenblas-dev git 
apt-get install -y gcc==7.3.0 cmake==3.12.0 #gcc7.3.0版本及以上，cmake3.12.0版本及以上。若用户要安装1.11.0版本PyTorch，则gcc需为7.5.0版本以上。
```
```
pip3 install torch==2.0.1+cpu  
```

# 安装Mindspore
> 版本配套参考： https://mindformers.readthedocs.io/zh-cn/latest/Version_Match.html
> 选择：MindFormers 0.8  MindSpore 2.2.0  CANN 7.0.RC.beta1: aarch64 x86_64


配置环境变量
如果昇腾AI处理器配套软件包没有安装在默认路径，安装好MindSpore之后，需要导出Runtime相关环境变量，下述命令中LOCAL_ASCEND=/usr/local/Ascend的/usr/local/Ascend表示配套软件包的安装路径，需注意将其改为配套软件包的实际安装路径。
```
. /usr/local/Ascend/ascend-toolkit/set_env.sh

# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# Environment variables that must be configured
## TBE operator implementation tool path
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe
## OPP path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp
## AICPU path
export ASCEND_AICPU_PATH=${ASCEND_OPP_PATH}/..
## TBE operator compilation tool path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/compiler/ccec_compiler/bin/:${PATH}
## Python library that TBE implementation depends on
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}
```

验证是否成功安装
方法一：
```
python -c "import mindspore;mindspore.set_context(device_target='Ascend');mindspore.run_check()"
```
如果输出：

MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed on platform [Ascend] successfully!
说明MindSpore安装成功了。

方法二：
```
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_context(device_target="Ascend")
x = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x, y))
```
如果输出：

[[[[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]

  [[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]

  [[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]]]
说明MindSpore安装成功了。

## 日常使用

激活环境
```
conda activate py39ms2.2
```

导入环境变量
```
#安装toolkit包时配置
. /usr/local/Ascend/ascend-toolkit/set_env.sh

# Mindspore环境变量
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# Environment variables that must be configured
## TBE operator implementation tool path
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe
## OPP path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp
## AICPU path
export ASCEND_AICPU_PATH=${ASCEND_OPP_PATH}/..
## TBE operator compilation tool path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/compiler/ccec_compiler/bin/:${PATH}
## Python library that TBE implementation depends on
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}
```

> 用户也可以通过修改~/.bashrc文件方式设置永久环境变量，操作如下：
> 1.以运行用户在任意目录下执行vi ~/.bashrc命令，打开.bashrc文件，在文件最后一行后面添加上述内容。
> 2.执行:wq!命令保存文件并退出。
> 3.执行source ~/.bashrc命令使其立即生效。

# 安装MindFormers
> 参考 https://gitee.com/mindspore/mindformers

## 安装
方式1：Linux源码编译安装
支持源码编译安装，用户可以执行下述的命令进行包的安装
```
git clone -b r0.8 https://gitee.com/mindspore/mindformers.git
cd mindformers
bash build.sh
```

方式2：镜像
docker下载命令
```
docker pull swr.cn-central-221.ovaijisuan.com/mindformers/mindformers0.8.0_mindspore2.2.0:aarch_20231025
```
创建容器
```
# --device用于控制指定容器的运行NPU卡号和范围
# -v 用于映射容器外的目录
# --name 用于自定义容器名称

docker run -dit -u root \
--ipc=host \
--network host \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /etc/localtime:/etc/localtime \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /var/log/npu/:/usr/slog \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /root/src:/root/src \
--name mf_8d-lpf \
swr.cn-central-221.ovaijisuan.com/mindformers/mindformers0.8.0_mindspore2.2.0:aarch_20231025 \
/bin/bash
```