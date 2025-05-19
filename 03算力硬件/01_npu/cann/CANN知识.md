# 昇腾GANN

## CANN安装
```
2.1G    Ascend-cann-kernels-310p_8.0.RC3_linux-aarch64.run
1.9G    Ascend-cann-toolkit_8.0.RC3_linux-aarch64.run
```

### 安装 cann-toolkit
执行以下命令安装软件（以下命令支持--install-path=<path>等参数，具体参数说明请参见参数说明）。
```
./Ascend-cann-toolkit_8.0.RC3_linux-aarch64.run --install
```

静默安装
```
./Ascend-cann-toolkit_8.0.RC3_linux-aarch64.run --install --quiet --nox11
```

### 安装 cann-kernels
```
./Ascend-cann-kernels-310p_8.0.RC3_linux-aarch64.run --install
```

静默安装
```
./Ascend-cann-kernels-310p_8.0.RC3_linux-aarch64.run --install --quiet --nox11
```

## CANN目录下文件
```
root@HW-Ascend-172376:/usr/local/Ascend/ascend-toolkit/8.0.RC3# du -sh *
1.5G    aarch64-linux
0       acllib
0       arm64-linux
28K     atc
0       bin
4.0K    cann_uninstall.sh
24K     combo_script
1.5M    compiler
0       conf
8.0K    create_version_softlink.sh
28K     fwkacllib
536K    hccl
0       include
0       lib64
228K    mindstudio-toolkit
7.3G    opp #算子实现文件，
864K    opp_kernel
0       ops
28K     pyACL
81M     python
50M     runtime
0       test-ops
257M    toolkit
603M    tools
8.0K    var
```

cann-kernels包主要安装在 `/usr/local/Ascend/ascend-toolkit/8.0.RC3/opp/built-in/op_impl/ai_core/tbe/kernel`。安装后，在该目录下会多一个 ascend310p 或 ascend910b 的目录

![image-20250515145947660](CANN知识.assets/image-20250515145947660.png)