# 安装驱动-openEuler

参考: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/softwareinst/instg/instg_0005.html?Mode=PmIns&OS=openEuler&Software=cannToolKit

物理机和容器部署场景，只需要在物理机安装NPU驱动和固件。

### 创建HwHiAiUser用户

创建HwHiAiUser用户和用户属组：
```
groupadd HwHiAiUser
useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash
```

#### 安装说明

- 首次安装场景：硬件设备刚出厂时未安装驱动，或者硬件设备前期安装过驱动和固件但是当前已卸载，上述场景属于首次安装场景，需按照**“驱动 > 固件”**的顺序安装驱动和固件。

- 覆盖安装场景：硬件设备前期安装过驱动和固件且未卸载，当前要再次安装驱动和固件，此场景属于覆盖安装场景，需按照

  “固件>驱动”

  的顺序安装。

  用户可使用如下命令查询当前环境是否安装驱动，若返回驱动相关信息说明已安装。

  ```
  npu-smi info
  ```

#### 安装驱动和固件

1. 以**root**用户登录安装环境，将驱动包和固件包上传到服务器任意目录如“/home”。

2. 安装驱动所需依赖。

   1. 执行如下命令检查源是否可用。

      openEuler系列（包含openEuler、CentOS、Kylin、BCLinux、BC-Linux-for-Euler、UOS20 1050e、UOS20 1020e、UOSV20、AntOS、CTyunOS、CULinux、Tlinux操作系统）：

      ```
      yum makecache
      ```

      如果命令执行报错或者等待时间过长，则检查网络是否连接或修改**“/etc/yum.repos.d/xxxx.repo”**文件为可用源（以配置华为镜像源为例，可参考[华为开源镜像站](https://mirrors.huaweicloud.com/)中镜像源对应的配置方法操作）。

   2. 执行命令安装所需依赖

      openEuler系列：

      ```
      yum install -y make dkms gcc kernel-headers-$(uname -r) kernel-devel-$(uname -r)
      ```

      如果出现报错或者依赖不存在，请参考[安装驱动源码编译所需依赖](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/softwareinst/instg/instg_0051.html#ZH-CN_TOPIC_0000002229014544)解决。

3. 进入软件包所在目录，执行如下命令增加执行权限和校验软件包的一致性和完整性。

   ```
   chmod +x Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run
   chmod +x Ascend-hdk-<chip_type>-npu-firmware_<version>.run
   ./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --check
   ./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --check
   ```

   出现如下回显信息，表示软件包校验成功。

   ```
   Verifying archive integrity...  100%   SHA256 checksums are OK. All good.
   ```

4. 安装驱动和固件，软件包默认安装路径为“/usr/local/Ascend”。

   - 执行如下命令安装驱动。

     ```
     ./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --install-for-all
     ```

     若系统出现如下关键回显信息，则表示驱动安装成功。

     ```
     Driver package installed successfully!
     ```

     若执行以上命令出现缺失部分Linux工具，请根据安装过程中回显信息提示自行安装。若出现dkms和缺少依赖等相关报错，请参考[驱动安装出现报错](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/softwareinst/instg/instg_0069.html#ZH-CN_TOPIC_0000002229174388)解决。

   - 执行如下命令安装固件。

     ```
     ./Ascend-hdk-<chip_type>-npu-firmware_<version>.run --full
     ```

     若系统出现如下关键回显信息，表示固件安装成功。

     ```
     Firmware package installed successfully! Reboot now or after driver installation for the installation/upgrade to take effect
     ```

   说明

   如果驱动和固件运行用户和运行用户属组不是HwHiAiUser时，则在安装驱动和固件包时必须指定运行用户和属组，示例命令如下：

   ```
   ./Ascend-hdk-<chip_type>-npu-driver_<version>_linux-<arch>.run --full --install-username=<username> --install-usergroup=<usergroup>
   ```
5. 根据系统提示信息决定是否重启系统，若需要重启，请执行以下命令；否则，请跳过此步骤。

   ```
   reboot
   ```

6. 执行如下命令查看驱动加载是否成功

   ```
   npu-smi info
   ```