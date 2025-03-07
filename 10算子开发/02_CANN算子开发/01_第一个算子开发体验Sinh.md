# Sinh昇腾算子开发体验

> 背景： [Ascend C算子开发能力认证（中级）](https://www.hiascend.com/edu/certification/exam/34bf904cb410497cb9c582be6c047ff7)

### 题目要求

实现Ascend C算子Sinh,算子命名为SinhCustom，编写其kernel侧代码、host侧代码，并完成aclnn算子调用测试。
相关算法：sinh(x) = (exp(x) - exp(-x)) / 2.0

代码工程：[SinhCustom.tar.gz](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/AscendC/SinhCustom.tar.gz)

### 代码编写

修改文件 SinhCustom\op_host\sinh_custom_tiling.h，增加tiling定义

```
BEGIN_TILING_DATA_DEF(SinhCustomTilingData)
  //考生自行定义tiling结构体成员变量
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;
```

修改文件 SinhCustom\op_host\sinh_custom.cpp，实现tiling

```
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SinhCustomTilingData tiling;
    //考生自行填充
    const uint32_t BLOCK_DIM = 8;
    const uint32_t TILE_NUM = 8;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
```

> 这两部分可以参考 [samples仓库的AddCustom](https://gitee.com/ascend/samples/tree/master/operator/ascendc/tutorials/AddCustomSample/FrameworkLaunch/AddCustom) 

修改 SinhCustom\op_kernel\sinh_custom.cpp，实现**KernelSinh**核函数

```
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelSinh {
public:
    __aicore__ inline KernelSinh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        //考生补充初始化代码
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ half *)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half *)y + this->blockLength * GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuffer1, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuffer3, this->tileLength * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        //考生补充对“loopCount”的定义，注意对Tiling的处理
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        //考生补充算子代码
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        //考生补充算子计算代码
        // sinh(x) = (exp(x) - exp(-x)) / 2.0 
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        LocalTensor<half> tmpTensor1 = tmpBuffer1.Get<half>();
        LocalTensor<half> tmpTensor2 = tmpBuffer2.Get<half>();
        LocalTensor<half> tmpTensor3 = tmpBuffer3.Get<half>();
        half negOne = -1;
        // 计算 exp(-x)
        Muls(tmpTensor1, xLocal, negOne, this->tileLength);
        Exp(tmpTensor2, tmpTensor1, this->tileLength);
         // 计算 exp(x)
        Exp(tmpTensor1, xLocal, this->tileLength);
        // 计算 (exp(x) - exp(-x)) 
        Sub(tmpTensor3, tmpTensor1, tmpTensor2, this->tileLength);
        // 计算 (exp(x) - exp(-x)) / 2.0 
        half oneHalf = 0.5;
        Muls(yLocal, tmpTensor3, oneHalf, this->tileLength);
        // printf("yLocal:%f",yLocal);
        outQueueY.EnQue<half>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        //考生补充算子代码
        LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        // free output tensor for reuse
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> tmpBuffer1, tmpBuffer2, tmpBuffer3;
    //create queue for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    //create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;

    //考生补充自定义成员变量
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void sinh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSinh op;
    //补充init和process函数调用内容
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
```

> 其中算子的核心实现是在  __aicore__ inline void Compute(int32_t progress) 函数中，有两个点注意：
>
> 1. 核函数的实现都是在AI_Cube或AI_Vector上操作tensor，使用函数时要特别注意
> 2. Mul是矢量双目运算，Muls才是可以传入标量的标量双目运算。

### **编译算子**

执行命令

```
cd SinhCustom/SinhCustom
bash build.sh
./build_out/custom_opp_ubuntu_aarch64.run
```

过程日志如下

```
root@19:/home/tomy/src/SinhCustom/SinhCustom# bash build.sh
Preset CMake variables:

  ASCEND_CANN_PACKAGE_PATH:PATH="/usr/local/Ascend/ascend-toolkit/latest"
  ASCEND_COMPUTE_UNIT:STRING="ascend910b"
  ASCEND_PYTHON_EXECUTABLE:STRING="python3"
  CMAKE_BUILD_TYPE:STRING="Release"
  CMAKE_CROSS_PLATFORM_COMPILER:PATH="/usr/bin/aarch64-linux-gnu-g++"
  CMAKE_INSTALL_PREFIX:PATH="/home/tomy/src/SinhCustom/SinhCustom/build_out"
  ENABLE_BINARY_PACKAGE:BOOL="True"
  ENABLE_CROSS_COMPILE:BOOL="False"
  ENABLE_SOURCE_PACKAGE:BOOL="True"
  ENABLE_TEST:BOOL="True"
  vendor_name:STRING="customize"

-- The C compiler identification is GNU 11.4.0
-- The CXX compiler identification is GNU 11.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Opbuild generating sources
-- Opbuild generating sources - done
-- Configuring done
-- Generating done
CMake Warning:
  Manually-specified variables were not used by the project:

    CMAKE_CROSS_PLATFORM_COMPILER


-- Build files have been written to: /home/tomy/src/SinhCustom/SinhCustom/build_out
[  7%] Generating scripts/install.sh, scripts/upgrade.sh
[ 15%] Building CXX object framework/tf_plugin/CMakeFiles/cust_tf_parsers.dir/tensorflow_sinh_custom_plugin.cc.o
[ 38%] Building CXX object op_host/CMakeFiles/cust_op_proto.dir/sinh_custom.cpp.o
[ 38%] Building CXX object op_host/CMakeFiles/cust_op_proto.dir/__/autogen/op_proto.cc.o
[ 38%] Building CXX object op_host/CMakeFiles/cust_optiling.dir/sinh_custom.cpp.o
[ 38%] Built target gen_version_info
[ 69%] Generating tbe/op_info_cfg/ai_core/npu_supported_ops.json
[ 69%] Generating tbe/.impl_timestamp
[ 69%] Generating tbe/op_info_cfg/ai_core/ascend910b/aic-ascend910b-ops-info.json
[ 69%] Building CXX object op_host/CMakeFiles/cust_opapi.dir/__/autogen/aclnn_sinh_custom.cpp.o
/home/tomy/src/SinhCustom/SinhCustom/build_out/autogen /home/tomy/src/SinhCustom/SinhCustom/build_out/op_kernel/tbe/op_info_cfg/ai_core


==============check valid for ops info start==============
==============check valid for ops info end================


Compile op info cfg successfully.
[ 69%] Built target modify_vendor
[ 69%] Built target ops_info_gen_ascend910b
[INFO] Succed generated /home/tomy/src/SinhCustom/SinhCustom/build_out/op_kernel/tbe/op_info_cfg/ai_core/npu_supported_ops.json
[ 69%] Built target ascendc_impl_gen
[ 69%] Built target npu_supported_ops
[ 76%] Linking CXX shared library libcust_opapi.so
[ 76%] Built target cust_opapi
[ 84%] Linking CXX shared library libcust_tf_parsers.so
[ 84%] Built target cust_tf_parsers
[ 92%] Linking CXX shared library libcust_opsproto_rt2.0.so
[100%] Linking CXX shared library libcust_opmaster_rt2.0.so
[100%] Built target cust_op_proto
[100%] Built target cust_optiling
[100%] Built target optiling_compat
Run CPack packaging tool...
CPack: Create package using External
CPack: Install projects
CPack: - Run preinstall target for: opp
CPack: - Install project: opp []
CPack: Create package

About to compress 444 KB of data...
Adding files to archive named "custom_opp_ubuntu_aarch64.run"...
./help.info
./install.sh
./packages/vendors/customize/framework/tensorflow/libcust_tf_parsers.so
./packages/vendors/customize/framework/tensorflow/npu_supported_ops.json
./packages/vendors/customize/op_api/include/aclnn_sinh_custom.h
./packages/vendors/customize/op_api/lib/libcust_opapi.so
./packages/vendors/customize/op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info.json
./packages/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic/sinh_custom.cpp
./packages/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic/sinh_custom.py
./packages/vendors/customize/op_impl/ai_core/tbe/kernel/ascend910b/sinh_custom/
./packages/vendors/customize/op_impl/ai_core/tbe/kernel/config/ascend910b/
./packages/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so
./packages/vendors/customize/op_impl/ai_core/tbe/op_tiling/liboptiling.so
./packages/vendors/customize/op_proto/inc/op_proto.h
./packages/vendors/customize/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so
./packages/vendors/customize/version.info
./upgrade.sh
CRC: 1731669034
SHA256: 0801b43d39ce03e58e4626b09f03b1a138833d1bf47ce5262a825ce70eac03cd
Skipping md5sum at user request

Self-extractable archive "custom_opp_ubuntu_aarch64.run" successfully created.
Copy /home/tomy/src/SinhCustom/SinhCustom/build_out/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/custom_opp_ubuntu_aarch64.run to /home/tomy/src/SinhCustom/SinhCustom/build_out/
CPack: - package: /home/tomy/src/SinhCustom/SinhCustom/build_out/custom_opp_ubuntu_aarch64.run.json generated.
CPack: - package: /home/tomy/src/SinhCustom/SinhCustom/build_out/custom_opp_ubuntu_aarch64.run generated.
Verifying archive integrity...  100%   SHA256 checksums are OK. All good.
Uncompressing version:1.0  100%
[runtime] [2025-03-05 10:48:52] [INFO] copy uninstall sh success
[ops_custom]upgrade framework
tensorflow [runtime] [2025-03-05 10:48:52] [INFO] replace or merge old ops framework files .g.....
[runtime] [2025-03-05 10:48:52] copy new ops framework files ......
[ops_custom]upgrade op proto
inc lib [runtime] [2025-03-05 10:48:52] [INFO] replace or merge old ops op_proto files .g.....
[runtime] [2025-03-05 10:48:52] copy new ops op_proto files ......
[ops_custom]upgrade version.info
[runtime] [2025-03-05 10:48:52] copy new version.info files ......
[ops_custom]upgrade op impl
ai_core [runtime] [2025-03-05 10:48:52] [INFO] replace or merge old ops op_impl files .g.....
[runtime] [2025-03-05 10:48:52] copy new ops op_impl files ......
[ops_custom]upgrade op api
include lib [runtime] [2025-03-05 10:48:52] [INFO] replace or merge old ops op_api files .g.....
[runtime] [2025-03-05 10:48:52] copy new ops op_api files ......
[runtime] [2025-03-05 10:48:52] [INFO] no need to upgrade custom.proto files
SUCCESS
[100%] Built target ascendc_impl_gen
[100%] Built target ascendc_bin_ascend910b_sinh_custom_copy
[100%] Built target ascendc_bin_ascend910b
[Ascend910B1] Generating SinhCustom_00b2b0b8ab8f50db439d6cb44263785b ...
Opc tool start working now, please wait for a moment.
start compile Ascend C operator SinhCustom. kernel name is SinhCustom_00b2b0b8ab8f50db439d6cb44263785b
[Ascend910B1] Generating SinhCustom_00b2b0b8ab8f50db439d6cb44263785b Done
/usr/bin/gmake
[100%] Built target ascendc_bin_ascend910b_sinh_custom_0
[100%] Built target ascendc_bin_ascend910b_gen_ops_config
[100%] Built target binary
[  7%] Built target modify_vendor
[  7%] Built target gen_version_info
[ 15%] Built target ops_info_gen_ascend910b
[ 76%] Built target npu_supported_ops
[ 76%] Built target ascendc_impl_gen
[ 84%] Built target cust_opapi
[ 84%] Built target cust_optiling
[ 92%] Built target cust_op_proto
[100%] Built target cust_tf_parsers
[100%] Built target optiling_compat
Run CPack packaging tool...
CPack: Create package using External
CPack: Install projects
CPack: - Run preinstall target for: opp
CPack: - Install project: opp []
CPack: Create package

About to compress 464 KB of data...
Adding files to archive named "custom_opp_ubuntu_aarch64.run"...
./help.info
./install.sh
./packages/vendors/customize/framework/tensorflow/libcust_tf_parsers.so
./packages/vendors/customize/framework/tensorflow/npu_supported_ops.json
./packages/vendors/customize/op_api/include/aclnn_sinh_custom.h
./packages/vendors/customize/op_api/lib/libcust_opapi.so
./packages/vendors/customize/op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info.json
./packages/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic/sinh_custom.cpp
./packages/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic/sinh_custom.py
./packages/vendors/customize/op_impl/ai_core/tbe/kernel/ascend910b/sinh_custom/SinhCustom_00b2b0b8ab8f50db439d6cb44263785b.json
./packages/vendors/customize/op_impl/ai_core/tbe/kernel/ascend910b/sinh_custom/SinhCustom_00b2b0b8ab8f50db439d6cb44263785b.o
./packages/vendors/customize/op_impl/ai_core/tbe/kernel/config/ascend910b/binary_info_config.json
./packages/vendors/customize/op_impl/ai_core/tbe/kernel/config/ascend910b/sinh_custom.json
./packages/vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so
./packages/vendors/customize/op_impl/ai_core/tbe/op_tiling/liboptiling.so
./packages/vendors/customize/op_proto/inc/op_proto.h
./packages/vendors/customize/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so
./packages/vendors/customize/version.info
./upgrade.sh
CRC: 1368852360
SHA256: 4016cd996fa25b7a7b5e63fc16bb9ed6c019283d01d4caf8c0dc498d2c5f1047
Skipping md5sum at user request

Self-extractable archive "custom_opp_ubuntu_aarch64.run" successfully created.
Copy /home/tomy/src/SinhCustom/SinhCustom/build_out/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/custom_opp_ubuntu_aarch64.run to /home/tomy/src/SinhCustom/SinhCustom/build_out/
CPack: - package: /home/tomy/src/SinhCustom/SinhCustom/build_out/custom_opp_ubuntu_aarch64.run.json generated.
CPack: - package: /home/tomy/src/SinhCustom/SinhCustom/build_out/custom_opp_ubuntu_aarch64.run generated.
root@19:/home/tomy/src/SinhCustom/SinhCustom# ./build_out/custom_opp_ubuntu_aarch64.run
Verifying archive integrity...  100%   SHA256 checksums are OK. All good.
Uncompressing version:1.0  100%
[runtime] [2025-03-05 10:50:31] [INFO] copy uninstall sh success
[ops_custom]upgrade framework
tensorflow [runtime] [2025-03-05 10:50:31] [INFO] replace or merge old ops framework files .g.....
[runtime] [2025-03-05 10:50:31] copy new ops framework files ......
[ops_custom]upgrade op proto
inc lib [runtime] [2025-03-05 10:50:31] [INFO] replace or merge old ops op_proto files .g.....
[runtime] [2025-03-05 10:50:31] copy new ops op_proto files ......
[ops_custom]upgrade version.info
[runtime] [2025-03-05 10:50:31] copy new version.info files ......
[ops_custom]upgrade op impl
ai_core [runtime] [2025-03-05 10:50:31] [INFO] replace or merge old ops op_impl files .g.....
[runtime] [2025-03-05 10:50:31] copy new ops op_impl files ......
[ops_custom]upgrade op api
include lib [runtime] [2025-03-05 10:50:31] [INFO] replace or merge old ops op_api files .g.....
[runtime] [2025-03-05 10:50:31] copy new ops op_api files ......
[runtime] [2025-03-05 10:50:31] [INFO] no need to upgrade custom.proto files
SUCCESS
```

### **AclNN调用算子**

执行命令

```
cd SinhCustom/AclNNInvocation
bash run.sh
```

过程日志如下

```
root@19:/home/tomy/src/SinhCustom# cd AclNNInvocation/
root@19:/home/tomy/src/SinhCustom/AclNNInvocation# bash run.sh
INFO: generate input data success!
-- The C compiler identification is GNU 11.4.0
-- The CXX compiler identification is GNU 11.4.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /usr/bin/cc - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- env INC_PATH: /usr/local/Ascend/ascend-toolkit/latest
-- env LIB_PATH: /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64
-- Configuring done
-- Generating done
-- Build files have been written to: /home/tomy/src/SinhCustom/AclNNInvocation/build
INFO: cmake success!
[ 20%] Building CXX object CMakeFiles/execute_sinh_op.dir/operator_desc.cpp.o
[ 40%] Building CXX object CMakeFiles/execute_sinh_op.dir/op_runner.cpp.o
[ 60%] Building CXX object CMakeFiles/execute_sinh_op.dir/main.cpp.o
[ 80%] Building CXX object CMakeFiles/execute_sinh_op.dir/common.cpp.o
[100%] Linking CXX executable /home/tomy/src/SinhCustom/AclNNInvocation/output/execute_sinh_op
[100%] Built target execute_sinh_op
INFO: make success!
INFO: execute op!
[INFO]  Set device[0] success
[INFO]  Get RunMode[1] success
[INFO]  Init resource success
[INFO]  Set input success
[INFO]  Copy input[0] success
[INFO]  Create stream success
[INFO]  Synchronize stream success
[INFO]  Copy output[0] success
[INFO]  Write output success
[INFO]  Run op success
[INFO]  Reset Device success
[INFO]  Destory resource success
INFO: acl executable run success!
golden:[2.715e+02 1.843e+00 2.348e+00 ... 3.148e+02 7.336e+03 6.562e+00] real_result:[2.715e+02 1.843e+00 2.348e+00 ... 3.148e+02 7.336e+03 6.559e+00] result:[0. 0. 0. ... 0. 0. 0.003906] test pass
```



### FAQ

1、**"AddCustom do not registe tiling struct!!! "  或 error: use of undeclared identifier 'tiling_data' 错误**

**问题**：在 `bash build.sh` 编译算子过程中，报  `error: use of undeclared identifier 'tiling_data' `错误。

**解决**：参考[昇腾论坛](https://www.hiascend.com/forum/thread-0215170325203931082-1-1.html) 去掉ASCEND_CUSTOM_OPP_PATH后，编译通过。

```
unset ASCEND_CUSTOM_OPP_PATH
```

**补充分析**：

当前ASCEND_CUSTOM_OPP_PATH 的内容如下

```
ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/mindie/latest/mindie-rt/ops/vendors/aie_ascendc:/usr/local/Ascend/mindie/latest/mindie-rt/ops/vendors/customize:
```

参考 [CANN算子包部署](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha002/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0069.html) ，`算子默认安装场景，不配置--install-path参数，安装成功后会将编译生成的自定义算子相关文件部署到${INSTALL_DIR}/opp/vendors/*<vendor_name>*目录。${INSTALL_DIR}请替换为CANN软件安装后文件存储路径。例如，若安装的Ascend-cann-toolkit软件包，安装后文件存储路径示例为：$HOME/Ascend/ascend-toolkit/latest。` 

确实在 `/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic` 目录中看到

```
root@19:/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_impl/ai_core/tbe/customize_impl/dynamic# ll -h
total 36K
drwxr-xr-x 2 root root 4.0K Mar  5 10:21 ./
drwxr-xr-x 3 root root 4.0K Mar  5 10:16 ../
-rw-r--r-- 1 root root 3.7K Mar  5 10:22 add_custom.cpp
-rw-r--r-- 1 root root 8.6K Mar  5 10:22 add_custom.py
-rw-r--r-- 1 root root 3.8K Mar  5 10:16 sinh_custom.cpp
-rw-r--r-- 1 root root 7.3K Mar  5 10:16 sinh_custom.py
```

在 `/usr/local/Ascend/ascend-toolkit/8.0.RC3/opp/vendors/customize/op_impl/ai_core/tbe/kernel`目录中看到

```
root@19:/usr/local/Ascend/ascend-toolkit/8.0.RC3/opp/vendors/customize/op_impl/ai_core/tbe/kernel# ll
drwxr-xr-x 7 root root 4096 Mar  5 10:21 ./
drwxr-xr-x 6 root root 4096 Mar  5 10:16 ../
drwxr-xr-x 3 root root 4096 Mar  5 10:21 ascend310b/
drwxr-xr-x 3 root root 4096 Mar  5 10:21 ascend310p/
drwxr-xr-x 3 root root 4096 Mar  5 10:21 ascend910/
drwxr-xr-x 4 root root 4096 Mar  5 10:21 ascend910b/
drwxr-xr-x 6 root root 4096 Mar  5 10:21 config/
```

在`/usr/local/Ascend/ascend-toolkit/8.0.RC3/opp/vendors/customize/op_impl/ai_core/tbe/kernel/ascend910b`目录中看到

```
root@19:/usr/local/Ascend/ascend-toolkit/8.0.RC3/opp/vendors/customize/op_impl/ai_core/tbe/kernel# find . -iname "*sin*"
./config/ascend910b/sinh_custom.json
./ascend910b/sinh_custom
./ascend910b/sinh_custom/SinhCustom_00b2b0b8ab8f50db439d6cb44263785b.json
./ascend910b/sinh_custom/SinhCustom_00b2b0b8ab8f50db439d6cb44263785b.o
```

本来猜测 ASCEND_CUSTOM_OPP_PATH 中加入这个路径即可，如下

```
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp/vendors/customize/:${ASCEND_CUSTOM_OPP_PATH}
```

测试后不行，还是会报 error: use of undeclared identifier 'tiling_data'

以上说明：

1. 算子在 `bash build.sh`后，将 `sinh_custom.cpp`和 `sinh_custom.py` 文件安装到了 `${ASCEND_TOOLKIT_HOME}/opp/vendors/customize/`下
2. 在 `./build_out/custom_opp_ubuntu_aarch64.run` 后，把 `SinhCustom_*.json`和 `SinhCustom_*.o` 文件安装到 `${ASCEND_TOOLKIT_HOME}/opp/vendors/customize/op_impl/ai_core/tbe/kernel`下

### 参考

1. [CANN Samples仓库](https://gitee.com/ascend/samples)
2. [CANN文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha002/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0064.html)
3. [CANN算子包部署](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha002/devguide/opdevg/ascendcopdevg/atlas_ascendc_10_0069.html)
3. [CANN API参考](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha002/apiref/ascendcopapi/atlasascendc_api_07_0037.html)
4. https://blog.csdn.net/m0_74120525/article/details/142027001
5. https://blog.csdn.net/qq_25745625/article/details/133888849
6. https://www.hiascend.com/forum/thread-0215170325203931082-1-1.html