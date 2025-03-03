# 安装CANN环境
设置HDK用户和安装依赖
```
groupadd HwHiAiUser
useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash

apt-get update
apt-get install -y dkms gcc linux-headers-$(uname -r)
```
> [参考](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/softwareinst/instg/instg_0004.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)

安装CANN
```
echo "ulimit -u unlimited" >> /etc/profile
source /etc/profile

apt-get install -y gcc make net-tools python3 python3-dev python3-pip
python3 --version
pip3 --version

pip3 install attrs cython numpy==1.24.0 decorator sympy cffi pyyaml pathlib2 psutil protobuf==3.20 scipy requests absl-py
chmod +x Ascend-cann-toolkit_*_linux-*.run

./Ascend-cann-toolkit_8.0.0_linux-x86_64.run --install-for-all
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /usr/local/Ascend/ascend-toolkit/latest/<arch>-linux
cat ascend_toolkit_install.info

echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> ~/.bashrc
```
> [参考](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/softwareinst/instg/instg_0007.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)

# 验证环境

验证cpu的开发环境是否OK
```
bash run.sh -r cpu
```
> [参考](https://www.hiascend.com/developer/courses/detail/1691696509765107713)