# LLM 大模型 学习之旅

> 一份围绕大语言模型（LLM）系统工程的学习路径指导，持续完善中。
>
> LLM 三大核心要素：**数据、算法、算力**。本仓库按"算力硬件 → 数据工程 → 模型开发（结构/训练/评测/推理）→ 应用生态"的顺序组织，力求一眼看清 LLM 系统工程全貌。

---

## 一、算力硬件

LLM 一切能力的物质基础。本仓库聚焦国产昇腾 NPU，兼顾 GPU 对比。

| 主题 | 文档 | 说明 |
| --- | --- | --- |
| NPU 总览 | [昇腾 NPU](03算力硬件/01_npu/README.md) | 镜像制作、容器启动、常见命令、环境变量 |
| 硬件架构 | [昇腾硬件架构](03算力硬件/01_npu/昇腾硬件架构.md) | 达芬奇架构、AI Core、HBM/Cache 层级 |
| CANN | [CANN 知识](03算力硬件/01_npu/cann/CANN知识.md) / [hccn_tool](03算力硬件/01_npu/cann/hccn_tool.md) | 镜像、集合通信 |
| 驱动安装 | [openEuler](03算力硬件/01_npu/driver/install_driver_openEuler.md) / [910A Ubuntu](03算力硬件/01_npu/install_910A_ubuntu20.04.md) | 驱动安装流程 |
| Docker | [NPU 上构建 Docker](03算力硬件/01_npu/create_docker_on_npu.md) | 容器制作 |
| GPU | [GPU 硬件管控](03算力硬件/02_gpu/GPU硬件管控.md) / [硬件对比](03算力硬件/02_gpu/01硬件对比.md) | H20/H200/Blackwell 定位 |

**TODO**：昇腾 910B/910C 与 H100/H200 算力对比 · 集群网络（RoCE/IB/HCCN） · 国产算力横向对比 · 存储与 IO 瓶颈

---

## 二、数据工程

"数据决定上限，算法逼近上限"。目前内容较少，是重点补全方向。

| 主题 | 文档 | 说明 |
| --- | --- | --- |
| numpy | [transpose 与 permute](01数据工程/numpy/transpose和permute.md) | PyTorch 转置函数 |

**TODO**：数据采集与清洗（去重/过滤/脱敏） · 分词器（BPE/SentencePiece/WordPiece） · 数据配比与课程学习 · SFT/RLHF 数据构造 · 多模态数据对齐 · 数据质量评测

---

## 三、模型开发

### 1. 基础知识

LLM 的理论地基，重点在 Transformer 与注意力机制。

| 主题 | 文档 | 说明 |
| --- | --- | --- |
| Transformer | [模型详解](02模型开发/00基础知识/Transformer模型详解.md) / [Attention Is All You Need](02模型开发/00基础知识/Attention%20Is%20All%20You%20Need.md) / [transfomer](02模型开发/00基础知识/transfomer.md) | Encoder/Decoder、整体结构 |
| Flash Attention | [flash_attention](02模型开发/00基础知识/flash_attention.md) | IO-Aware、分块计算、HBM/SRAM |
| Norm | [Norm](02模型开发/00基础知识/Norm.md) | Layer Norm / RMS Norm / Deep Norm、Pre/Post-Norm |
| 优化器 | [优化器](02模型开发/00基础知识/优化器.md) / [权重衰减](02模型开发/00基础知识/ml基础/weight_deacy.md) | Adam、warmup |
| 上下文长度 | [LLM 中的上下文长度](02模型开发/00基础知识/LLM中的上下文长度.md) | 长上下文挑战 |
| 计算公式 | [MFU 计算公式](02模型开发/00基础知识/大模型%20MFU%20计算公式.md) / [线性度公式](02模型开发/00基础知识/线性度公式.md) | GPT/LLaMA/MoE 计算量、集群扩展性 |

**TODO**：位置编码（RoPE/ALiBi） · 激活函数（GELU/SwiGLU） · KV Cache 与 PagedAttention · 混合精度训练（FP16/BF16/FP8） · 梯度累加与 ZeRO 优化

### 2. 模型结构

主流 LLM 架构解读。

| 模型 | 文档 | 说明 |
| --- | --- | --- |
| DeepSeek V3 | [deepseek_v3](02模型开发/01模型结构/01LLM模型/deepseek/deepseek_v3.md) | MoE、MLA、无辅助损失负载均衡 |
| Qwen3 | [qwen3](02模型开发/01模型结构/01LLM模型/qwen/qwen3/qwen3.md) | — |

**TODO**：LLaMA 系列架构演进 · Mistral / Mixtral MoE · 多模态模型结构（LLaVA、Qwen-VL） · Embedding 模型结构（BGE、M3）

### 3. 模型训练

预训练、微调、强化学习的工程实践，以昇腾 MindSpeed 体系为主线。

**训练基础**

| 主题 | 文档 | 说明 |
| --- | --- | --- |
| 并行策略 | [模型并行知识](02模型开发/02模型训练/模型并行知识.md) | DP/TP/PP/CP 关系 |
| 内存计算 | [内存计算](02模型开发/02模型训练/内存计算.md) | A100 算力、静态/动态内存 |
| 确定性计算 | [固定随机性和确定性计算](02模型开发/02模型训练/固定随机性和确定性计算.md) / [具身模型](02模型开发/02模型训练/具身模型固定确定性计算.assets/) | 固定随机性、排查 loss 差异 |
| 数据加载 | [数据 shuffle](02模型开发/02模型训练/数据shuffle.md) | shuffle 对训练的影响 |
| FAQ | [FAQ](02模型开发/02模型训练/FAQ.md) | 训练速度、IB/RoCE、K/V 矩阵 |

**MindSpeed 体系**

| 子库 | 文档 | 说明 |
| --- | --- | --- |
| 总览 | [MindSpeed](02模型开发/02模型训练/mindspeed/README.md) | 与 Megatron/DeepSpeed 对比 |
| MindSpeed-LLM | [2.0.0](02模型开发/02模型训练/mindspeed/mindspeed-llm/2.0.0/mindspeed-llm-2.0.0.md) / [Docker](02模型开发/02模型训练/mindspeed/mindspeed-llm/2.0.0/docker_guide.md) / [Qwen2.5-7B](02模型开发/02模型训练/mindspeed/mindspeed-llm/2.0.0/qwen2.5-7B.md) / [Qwen3-14B](02模型开发/02模型训练/mindspeed/mindspeed-llm/2.1.0/qwen3-14B.md) / [开源数据集](02模型开发/02模型训练/mindspeed/mindspeed-llm/2.1.0/qwen3-14B-开源数据集.md) / [精度对齐](02模型开发/02模型训练/mindspeed/mindspeed-llm/2.1.0/qwen3-14B精度对齐.md) / [问题排查](02模型开发/02模型训练/mindspeed/mindspeed-llm/2.1.0/qwen3-14B问题.md) | LLM 预训练实践 |
| MindSpeed-MM | [多模态](02模型开发/02模型训练/mindspeed/mindspeed-mm/mindspeed-mm.md) | 多模态训练 |
| MindSpeed-RL | [安装](02模型开发/02模型训练/mindspeed/mindspeed-rl/install_guide.md) / [实践](02模型开发/02模型训练/mindspeed/mindspeed-rl/mindspeed-rl_practice.md) / [GRPO 算法](02模型开发/02模型训练/mindspeed/mindspeed-rl/algorithms/grpo.md) / [GRPO YAML](02模型开发/02模型训练/mindspeed/mindspeed-rl/features/grpo_yaml.md) / [R1-Zero 7B](02模型开发/02模型训练/mindspeed/mindspeed-rl/solutions/r1_zero_qwen25_7b.md) / [R1-Zero 32B](02模型开发/02模型训练/mindspeed/mindspeed-rl/solutions/r1_zero_qwen25_32b.md) / [数据模块](02模型开发/02模型训练/mindspeed/mindspeed-rl/features/data_module_design.md) / [worker](02模型开发/02模型训练/mindspeed/mindspeed-rl/features/integrated_worker.md) / [profiler](02模型开发/02模型训练/mindspeed/mindspeed-rl/features/profiler.md) / [msprobe](02模型开发/02模型训练/mindspeed/mindspeed-rl/features/msprobe.md) / [wandb](02模型开发/02模型训练/mindspeed/mindspeed-rl/features/logging_wandb_tensorboard.md) | 强化学习 |
| 辅助工具 | [PTA](02模型开发/02模型训练/mindspeed/pta/pta.md) / [Apex](02模型开发/02模型训练/mindspeed/pta/apex.md) / [模型迁移](02模型开发/02模型训练/mindspeed/模型迁移.md) / [精度对齐](02模型开发/02模型训练/mindspeed/精度对齐.md) | 迁移与对齐 |
| MindSpore | [接入 MindSpeed](02模型开发/02模型训练/mindspore/MindSpore接入MindSpeed.md) / [并行](02模型开发/02模型训练/mindspore/ms_并行.md) | MindSpore 生态 |
| 算法 | [GRPO 原理](02模型开发/00基础知识/grpo.md) | 从 PPO 到 GRPO |

**TODO**：预训练全流程（数据配比/loss/checkpoint） · 微调方法（LoRA/QLoRA/Adapter/Prefix Tuning） · RLHF 全链路（SFT→RM→PPO/DPO） · 分布式通信（NCCL/HCCL、AllReduce 代价） · 断点续训与容错 · 训练性能调优

### 4. 模型评测

| 主题 | 文档 | 说明 |
| --- | --- | --- |
| 目录 | [模型评测](02模型开发/03模型评测/RAEDME.md) | 待完善 |

**TODO**：评测基准（MMLU/CMMLU/C-Eval/HumanEval/GSM8K） · 评测框架（OpenCompass/lm-eval-harness） · 评测指标（困惑度/准确率/pass@k） · 人类对齐评测

### 5. 模型推理

核心命题："存得下" + "跑得快"。以昇腾 MindIE 为主线。

| 主题 | 文档 | 说明 |
| --- | --- | --- |
| 推理总览 | [模型推理](02模型开发/04模型推理/RAEDME.md) | MindIE 代码/镜像/文档 |
| MindIE | [安装](02模型开发/04模型推理/MindIE/README.md) / [推理接口](02模型开发/04模型推理/MindIE/MindIE推理接口.md) / [性能调优](02模型开发/04模型推理/MindIE/性能调优.md) | MindIE 全流程 |
| BGE | [BGE 模型部署](02模型开发/04模型推理/BGE.md) | Embedding 模型 |
| QWQ | [ATB-MindIE 版](02模型开发/04模型推理/QWQ-ATB_MindIE版.md) / [MindSpore-MindIE 版](02模型开发/04模型推理/QWQ-MindSpore_MindIE版.md) | QWQ 推理 |
| 多模态 | [Qwen2-7B-Audio](02模型开发/04模型推理/Qwen2-7B-Audio_ATB-MindIE.md) / [Qwen2.5-vl-7B](02模型开发/04模型推理/Qwen2.5-vl-7B-MindIE.md) | 音频/视觉模型推理 |

**TODO**：推理优化（量化 GPTQ/AWQ/SmoothQuant、剪枝、蒸馏） · 解码策略（Greedy/Beam/Sampling/Speculative） · 推理框架对比（vLLM/TGI/TensorRT-LLM/MindIE） · 服务化部署（Continuous Batching） · 性能指标（TTFT/TPOT/Throughput）

---

## 四、应用生态

| 主题 | 说明 |
| --- | --- |
| 框架 | TODO：LangChain / LlamaIndex |
| Agent | TODO：ReAct / AutoGPT / 多智能体 |
| RAG | TODO：检索增强生成 |
| Prompt | TODO：Prompt Engineering |
| 工具调用 | TODO：Function Calling / Tool Use |
| 编排 | TODO：LangGraph / 工作流编排 |

---

## 五、参考资源

| 类型 | 资源 |
| --- | --- |
| 论文 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) · [FlashAttention](https://arxiv.org/abs/2205.14135) · [DeepSeekMath (GRPO)](https://arxiv.org/pdf/2402.03300) · [DeepSeek V3](https://arxiv.org/abs/2412.19437) · [GLM](https://arxiv.org/pdf/2103.10360.pdf) |
| 课程 | [清华 NLP 公开课](https://www.bilibili.com/video/BV1UG411p7zv) · [ChatGLM3 教程](https://www.bilibili.com/video/BV1Hc411q76k) · [大模型微调实战](https://www.bilibili.com/video/BV1dN4y167vX) |
| 仓库 | [llm-action](https://github.com/liguodongiot/llm-action) · [MindIE-LLM](https://gitee.com/ascend/MindIE-LLM.git) · [MindSpeed](https://www.hiascend.com/software/mindspeed) |

---

## 全貌速览

```
LLM 系统工程
├── 算力硬件   NPU(GPU) 架构 / 驱动 / CANN / 集群网络
├── 数据工程   采集 / 清洗 / 分词 / 配比 / 标注
├── 模型开发
│   ├── 基础知识   Transformer / Attention / Norm / 优化器 / 上下文长度
│   ├── 模型结构   DeepSeek / Qwen / LLaMA / MoE / 多模态
│   ├── 模型训练   预训练 / 微调 / RLHF / 并行策略 / MindSpeed
│   ├── 模型评测   基准 / 框架 / 指标
│   └── 模型推理   MindIE / 量化 / 解码 / 服务化
└── 应用生态   LangChain / Agent / RAG / Prompt
```

---

> 如果觉得还不错，关注 【飞见AI】 公众号获得更多优质文章和免费资料。
>
> ![](assets/wechat_code.jpg)
