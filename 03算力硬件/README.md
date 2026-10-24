# 算力硬件

LLM 一切能力的物质基础。本仓库聚焦国产昇腾 NPU，兼顾 GPU 对比。

## NPU（昇腾）
- [NPU 总览](01_npu/README.md) — 镜像制作、容器启动、常见命令、环境变量
- [昇腾硬件架构](01_npu/昇腾硬件架构.md) — 达芬奇架构、AI Core、HBM/Cache 层级
- CANN：[CANN 知识](01_npu/cann/CANN知识.md) / [hccn_tool](01_npu/cann/hccn_tool.md) / [相关工具](01_npu/cann/相关工具.md)
- 驱动安装：[openEuler](01_npu/driver/install_driver_openEuler.md) / [910A Ubuntu](01_npu/install_910A_ubuntu20.04.md)
- [NPU 上构建 Docker](01_npu/create_docker_on_npu.md)

## GPU
- [GPU 硬件管控](02_gpu/GPU硬件管控.md)
- [硬件对比](02_gpu/01硬件对比.md) — H20 / H200 / Blackwell 定位

## TODO（待补全）
- 昇腾 910B / 910C 与 H100 / H200 算力与互联对比
- 集群网络（RoCE / IB / HCCN）
- 国产算力横向对比（昇腾 / 海光 / 寒武纪 / 燧原）
- 存储与 IO 瓶颈（训练数据吞吐、Checkpoint 读写）
