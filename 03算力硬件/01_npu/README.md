# 昇腾NPU
## 资源
1. 制作昇腾镜像参考仓库：https://gitee.com/ascend/ascend-docker-image
2. 昇腾镜像仓库：https://www.hiascend.com/developer/ascendhub
3. MF镜像仓库：http://mirrors.cn-central-221.ovaijisuan.com/detail/154.html
4. 制作MS容器参考： https://gitee.com/Maigee/create_docker
5. 

## 常见查询方法
1. 查看每张卡驱动的命令

  ```
  /usr/local/Ascend/driver/tools/upgrade-tool --device_index -1 --component -1 --version | grep 'Get component version' | awk -F'(' '{print $2}' | awk -F')' '{print $1}'
  ```

2. 查看pci中有几张昇腾卡
```
lspci | grep d802  #910B
lspci | grep d500  #310P
```
3. 允许两个容器占用同样的npu卡，启动容器时添加配置 --privileged=true。赋予容器额外权限，使其能够访问和使用宿主机的磁盘、驱动设备，加载内核模块，操作硬件资源。如果不加该选项，容器启动后，查询npu-smi info，会显示：
DrvMngGetConsoleLogLevel failed. (g_conLogLevel=3)
dcmi model initialized failed, because the device is used. ret is -8020

4. 昇腾各个组件查询方法
| **组件**         | **版本**                                                     | **查看方法**                                                 | **备注** |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------- |
| 驱动             | 23.0.3                                                       | cat /usr/local/Ascend/driver/version.info或者ascend-dmi -c #更全面 |          |
| 固件             | 7.1.0.5.220                                                  | cat /usr/local/Ascend/firmware/version.info                  |          |
| 宿主机OS         | CTyunOS release 22.06.2                                      | cat /etc/*release                                            |          |
| 容器OS           | EulerOS release 2.0 (SP10)                                   | cat /etc/*release                                            |          |
| 容器内CANN       | 8.0.RC1                                                      | ll /usr/local/Ascend/ascend-toolkit/latest或者cd /usr/local/Ascend/ascend-toolkit/latest/<arch>-linuxcat [ascend_toolkit_install.info](https://ascend_toolkit_install.info/) |          |
| 容器内MindSpore  | 2.3.0rc6                                                     | pip list \| grep mindspore                                   |          |
| 容器内MindFormer | new_telechat_r1_1_0_0527                                     | 进入mindformer的目录，执行git branch                         |          |
| 容器内Python版本 | Python 3.9.10                                                | 进入容器，执行python                                         |          |
| docker镜像       | [harbor.telecom-ai.com.cn/ai-nlp/mf1.1_ms23rc5:v0629_formal_v5](https://harbor.telecom-ai.com.cn/ai-nlp/mf1.1_ms23rc5:v0629_formal_v5) |                                                              |          |

## 常见环境设置

1. 启动容器的命令
    ```
    docker run -it -u 0 --ipc=host  --network host \
    --name mf1.3_ma-0.4 \
    --privileged \
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
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/sbin/:/usr/local/sbin/ \
    -v /home/pae101/:/home/pae101/ \
    mf1.3_ma:0.4  \
    /bin/bash
    ```
    -u 0 以root启动容器；--privileged特权方式驱动。

2. 装好驱动后，生效驱动，设置环境变量
    ```
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH}
    ```

3. 安装好CANN-Toolkit后，设置环境变量
    ```
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```

    其中set_env.sh的文件内容如下
    
    ```
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
    export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
    export LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/$(arch):$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/tools/aml/lib64:${ASCEND_TOOLKIT_HOME}/tools/aml/lib64/plugin:$LD_LIBRARY_PATH
    export PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
    export PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:${ASCEND_TOOLKIT_HOME}/tools/ccec_compiler/bin:$PATH
    export ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}
    export ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
    export TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit
    export ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}
    ```

4. 设置环境变量，让模型只能在固定的几张卡可见
    ```
    export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
    ```
5. 指定模型在某张NPU卡上运行，设置环境变量：
    ```
    export DEVICE_ID=3
    ```
