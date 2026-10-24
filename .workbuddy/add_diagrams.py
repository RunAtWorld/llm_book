# -*- coding: utf-8 -*-
"""给 17 篇文档批量插入 Mermaid 图示（放在 '## 图示速览' 小节）。"""
import os

ROOT = r"E:/src/my/llm_book"

diagrams = {
    # ---------------- 基础知识 ----------------
    "02模型开发/00基础知识/位置编码 RoPE与ALiBi.md": (
        "**位置编码演进全景**：从绝对（Sinusoidal / 可学习）走向相对（RoPE 旋转注入、ALiBi 分数偏置），"
        "RoPE/ALiBi 因天然相对性与强外推成为主流。\n\n"
        "```mermaid\n"
        "graph LR\n"
        "  S[Sinusoidal<br/>固定 sin/cos] --> L[可学习 Embedding<br/>随训 难外推]\n"
        "  L --> R[RoPE<br/>旋转注入 q/k<br/>相对距离]\n"
        "  L --> A[ALiBi<br/>分数加线性偏置<br/>原生外推]\n"
        "  R -.->|NTK / PI / YaRN| RE[长上下文外推]\n"
        "  A -.->|仅依赖距离| RE\n"
        "```\n"
    ),
    "02模型开发/00基础知识/KV Cache与PagedAttention.md": (
        "**PagedAttention 核心思想**：逻辑 KV 序列按块切分，再映射到不连续的物理显存块池，"
        "按需分配、不同序列可共享同一物理块（前缀缓存），从而消除碎片。\n\n"
        "```mermaid\n"
        "flowchart LR\n"
        "  Seq[逻辑 KV 序列] -->|按块切分| Blocks[逻辑块 0,1,2,3]\n"
        "  Blocks -->|映射到| Phy[物理显存块池 p0..pN]\n"
        "  Phy -->|按需分配 不连续| NoFrag[消除内部/外部碎片]\n"
        "  Note1[不同序列可共享同一物理块<br/>前缀缓存/复用] --> Phy\n"
        "```\n"
    ),
    "02模型开发/00基础知识/混合精度训练 FP16 BF16 FP8.md": (
        "**数值精度选型决策**：硬件支持 FP8 且追求吞吐可上 FP8；否则 BF16 是训练默认首选，"
        "推理阶段再按需做权重量化进一步省显存。\n\n"
        "```mermaid\n"
        "flowchart TD\n"
        "  Start[选择数值精度] --> Q1{硬件支持 FP8?}\n"
        "  Q1 -->|是 追求吞吐/省显存| FP8[FP8 训练<br/>E4M3 / E5M2]\n"
        "  Q1 -->|否| BF16[BF16 训练<br/>默认首选]\n"
        "  BF16 --> Q2{推理要更省?}\n"
        "  Q2 -->|W8A8 / INT4| Quant[权重量化部署]\n"
        "  Q2 -->|保持| Keep[FP16 / BF16 推理]\n"
        "  FP8 --> Q3{稳定性问题?}\n"
        "  Q3 -->|loss spike| BF16\n"
        "```\n"
    ),
    "02模型开发/00基础知识/ZeRO与优化器状态切分.md": (
        "**ZeRO 分片递进**：ZeRO-1 只切优化器状态，ZeRO-2 再加梯度，ZeRO-3 进一步切参数。"
        "切得越细单卡显存越低，但跨卡通信越多。\n\n"
        "```mermaid\n"
        "graph TD\n"
        "  D[完整副本: Opt + Grad + Param] --> Z1[ZeRO-1: 仅切分优化器状态]\n"
        "  Z1 --> Z2[ZeRO-2: + 切分梯度]\n"
        "  Z2 --> Z3[ZeRO-3: + 切分参数]\n"
        "  Z3 --> F[单卡显存 ↓ 通信 ↑]\n"
        "```\n"
    ),
    "02模型开发/00基础知识/序列并行与长上下文训练.md": (
        "**并行策略谱系**：序列并行（Megatron SP / Ulysses / Ring Attention）与 TP、PP、DP 并列，"
        "专门解决超长序列下激活显存与注意力计算的分片问题。\n\n"
        "```mermaid\n"
        "graph TB\n"
        "  Root[模型并行] --> DP[数据并行 DP]\n"
        "  Root --> TP[张量并行 TP<br/>层内按列/行切]\n"
        "  Root --> PP[流水线并行 PP<br/>层间切]\n"
        "  Root --> SP[序列并行 SP]\n"
        "  SP --> Meg[Megatron SP<br/>切序列维度, 配 TP]\n"
        "  SP --> Uly[Ulysses<br/>head 维 all-to-all]\n"
        "  SP --> Ring[Ring Attention<br/>分块注意力 + 环通信]\n"
        "```\n"
    ),
    "02模型开发/00基础知识/激活函数 GELU与SwiGLU.md": (
        "**SwiGLU 门控结构**：输入 split 后一支过门控（SiLU 激活）与另一支逐元素乘，"
        "再做下投影。相比普通 FFN 多一个门控矩阵，是现代 LLM 的标准 MLP。\n\n"
        "```mermaid\n"
        "flowchart LR\n"
        "  X[输入 x] --> Split{{split}}\n"
        "  Split --> Up[上投影 W_up·x]\n"
        "  Split --> Gate[门控 W_gate·x]\n"
        "  Gate --> Sig[SiLU]\n"
        "  Sig --> Mul((×))\n"
        "  Up --> Mul\n"
        "  Mul --> Down[下投影 W_down]\n"
        "  Down --> Y[输出]\n"
        "```\n"
    ),
    # ---------------- 模型结构 ----------------
    "02模型开发/01模型结构/01LLM模型/llama/LLaMA系列架构演进.md": (
        "**LLaMA 架构演进时间线**：从 dense 走向 MoE，上下文长度从 4K 一路扩展到 10M，"
        "并引入原生多模态（LLaMA-4）。\n\n"
        "```mermaid\n"
        "timeline\n"
        "  title LLaMA 架构演进\n"
        "  2023-02 : LLaMA-1 : 7/13/33/65B : 纯 dense : RoPE+SwiGLU\n"
        "  2023-07 : LLaMA-2 : 引入 Chat/RLHF : 上下文 4K\n"
        "  2024-04 : LLaMA-3 : 15T token : GQA\n"
        "  2024-07 : LLaMA-3.1 : 405B : 上下文 128K\n"
        "  2025-04 : LLaMA-4 : MoE : Scout 10M / Maverick 1M : 原生多模态\n"
        "```\n"
    ),
    "02模型开发/01模型结构/01LLM模型/mistral/Mistral与Mixtral MoE.md": (
        "**MoE 路由与滑动窗口**：Router 对每 token 选 top-k 专家，专家输出加权求和；"
        "注意力采用滑动窗口只在局部窗口内计算，跨窗口靠缓存传递全局信息。\n\n"
        "```mermaid\n"
        "flowchart LR\n"
        "  Tok[Token 表征] --> Router[Router 门控]\n"
        "  Router -->|top-k| E1[Expert 1]\n"
        "  Router -->|top-k| E2[Expert 2]\n"
        "  Router -->|top-k| E3[Expert 3 ... N]\n"
        "  E1 --> Out[加权求和输出]\n"
        "  E2 --> Out\n"
        "  E3 --> Out\n"
        "  SW[滑动窗口: 仅看邻近 w 个 token] -.注意力.-> Attn[局部+全局跨窗口]\n"
        "```\n"
    ),
    "02模型开发/01模型结构/01LLM模型/deepseek/DeepSeek V3.1与R1 推理模型.md": (
        "**DeepSeek-R1 四阶段训练管线**：冷启动 SFT → 推理 RL（GRPO）→ 拒绝采样+SFT → 全场景 RL。"
        "R1-Zero 作为对照，证明纯 RL（无 SFT）也能涌现长链推理。\n\n"
        "```mermaid\n"
        "flowchart TD\n"
        "  S1[阶段1: 冷启动 SFT<br/>少量高质量 CoT] --> S2[阶段2: 推理 RL<br/>GRPO 强化数学/代码]\n"
        "  S2 --> S3[阶段3: 拒绝采样 + SFT<br/>融合推理/通用]\n"
        "  S3 --> S4[阶段4: 全场景 RL<br/>对齐偏好/安全]\n"
        "  R0[R1-Zero: 纯 RL 无 SFT<br/>自发涌现长链推理] -.对照.-> S2\n"
        "```\n"
    ),
    "02模型开发/01模型结构/01LLM模型/kimi/Kimi K2 MoE.md": (
        "**Kimi K2 架构要点**：MuonMoE（384 专家、激活 32）配合 MLA 压缩 KV，"
        "优化器用 MuonClip（权重 Muon + 注意力/嵌入 AdamW）稳定大规模训练。\n\n"
        "```mermaid\n"
        "flowchart TB\n"
        "  In[输入 Tokens] --> Emb[Embedding]\n"
        "  Emb --> Blocks[共 N 层 Decoder]\n"
        "  Blocks --> MLA[MLA 注意力<br/>压缩 KV]\n"
        "  Blocks --> MoE[MuonMoE<br/>384 专家 / 选 32 激活]\n"
        "  MoE --> Router[Router]\n"
        "  Router -->|top-32| Act[激活专家]\n"
        "  Blocks --> Out[LM Head]\n"
        "  Opt[MuonClip 优化器<br/>权重用 Muon, 注意力/嵌入用 AdamW] --> Blocks\n"
        "```\n"
    ),
    "02模型开发/01模型结构/01LLM模型/multimodal/多模态模型结构 LLaVA Qwen-VL InternVL.md": (
        "**多模态融合骨架**：图像经视觉编码器（ViT）提取特征，经投影层变成视觉 Token，"
        "与文本 Token 一起送入 LLM 完成跨模态理解与生成。\n\n"
        "```mermaid\n"
        "flowchart LR\n"
        "  Img[图像] --> ViT[视觉编码器 ViT]\n"
        "  ViT --> Proj[投影层 Projector<br/>MLP / Resampler]\n"
        "  Proj --> Tok[视觉 Token]\n"
        "  Tok --> LLM[大语言模型 LLM]\n"
        "  Q[文本] --> LLM\n"
        "  LLM --> Ans[多模态回答]\n"
        "```\n"
    ),
    "02模型开发/01模型结构/01LLM模型/embedding/BGE与M3 Embedding模型.md": (
        "**BGE-M3 三模态检索**：同一编码器同时产出稠密 / 稀疏 / 多向量三种表示并融合，"
        "召回结果送 LLM 做生成（典型 RAG 链路）。\n\n"
        "```mermaid\n"
        "flowchart LR\n"
        "  Q[查询] --> M3[BGE-M3 编码器]\n"
        "  M3 --> D[稠密向量 Dense]\n"
        "  M3 --> S[稀疏向量 Sparse]\n"
        "  M3 --> MV[多向量 Multi-vector]\n"
        "  D --> Retr[融合检索]\n"
        "  S --> Retr\n"
        "  MV --> Retr\n"
        "  Retr --> Gen[LLM 生成 基于召回片段]\n"
        "```\n"
    ),
    # ---------------- 推理优化 ----------------
    "02模型开发/04模型推理/量化推理 GPTQ AWQ SmoothQuant FP8.md": (
        "**量化方法谱系**：按粒度分 W8A8/FP8（权重+激活）与 W4A16（仅权重）；"
        "PTQ 中 GPTQ 用二阶补偿、AWQ 保护显著激活；SmoothQuant 把激活分布平滑到权重。\n\n"
        "```mermaid\n"
        "flowchart TD\n"
        "  Q[量化推理] --> W8A8[W8A8 / FP8<br/>权重+激活 8bit]\n"
        "  Q --> W4[W4A16 INT4<br/>仅权重量化]\n"
        "  Q --> PT[训练后量化 PTQ]\n"
        "  PT --> GPTQ[GPTQ<br/>二阶补偿]\n"
        "  PT --> AWQ[AWQ<br/>激活 salient 保护]\n"
        "  W8A8 --> SS[SmoothQuant<br/>平滑激活分布到权重]\n"
        "```\n"
    ),
    "02模型开发/04模型推理/推理框架对比 vLLM SGLang TensorRT-LLM MindIE.md": (
        "**框架选型决策树**：有昇腾优先 MindIE；否则按极致性能 / 结构化与长上下文 / 通用上手逐层分流。\n\n"
        "```mermaid\n"
        "flowchart TD\n"
        "  Q0{有昇腾 NPU?} -->|是| M[MindIE<br/>原生对接]\n"
        "  Q0 -->|否| Q1{极致性能且 NVIDIA?}\n"
        "  Q1 -->|是| T[TensorRT-LLM]\n"
        "  Q1 -->|否| Q2{结构化/agent/长上下文多?}\n"
        "  Q2 -->|是| S[SGLang]\n"
        "  Q2 -->|否| V[vLLM<br/>通用默认]\n"
        "```\n"
    ),
    "02模型开发/04模型推理/解码策略与投机解码.md": (
        "**投机解码时序**：Draft 小模型一次并行生成 k 个候选，Target 大模型一次前向全部验证；"
        "全部接受时等效 1 次前向产出 k token，显著加速。\n\n"
        "```mermaid\n"
        "sequenceDiagram\n"
        "  participant D as Draft 小模型\n"
        "  participant T as Target 大模型\n"
        "  participant O as 输出\n"
        "  D->>T: 并行生成 k 个候选 token\n"
        "  T->>T: 一次性验证 k 个\n"
        "  alt 全部接受\n"
        "    T->>O: 输出 k token (1 次前向)\n"
        "  else 部分接受\n"
        "    T->>O: 输出前 m 个 + 自校 1 个\n"
        "  end\n"
        "```\n"
    ),
    "02模型开发/04模型推理/服务化部署 Continuous Batching.md": (
        "**静态批处理 vs 连续批处理**：静态批需等整批最慢请求完成才释放，短请求空转；"
        "连续批中请求随时插入、各自完成即释放，GPU 利用率更高。\n\n"
        "```mermaid\n"
        "flowchart LR\n"
        "  subgraph Static[静态批处理]\n"
        "    A1[请求A 12 tok] --> Wait[等所有完成]\n"
        "    Wait --> A2[请求B 3 tok 空转]\n"
        "  end\n"
        "  subgraph Cont[连续批处理]\n"
        "    B1[请求A 进行中] --> B2[请求B 即时插入]\n"
        "    B2 --> B3[各自完成即释放]\n"
        "  end\n"
        "```\n"
    ),
    "02模型开发/04模型推理/推理性能指标 TTFT TPOT Throughput.md": (
        "**关键延迟指标时序**：TTFT 为首 token 延迟（Prefill 阶段），TPOT 为相邻生成 token 间隔"
        "（Decode 阶段），Throughput 为总 token / 总时长。\n\n"
        "```mermaid\n"
        "sequenceDiagram\n"
        "  participant U as 用户\n"
        "  participant S as 服务\n"
        "  U->>S: 提交请求 (Prefill)\n"
        "  Note over S: TTFT = 首 token 延迟\n"
        "  S-->>U: token 1\n"
        "  S-->>U: token 2\n"
        "  S-->>U: token 3 ...\n"
        "  Note over U,S: TPOT = 相邻 token 间隔\n"
        "  Note over U,S: Throughput = 总 token / 总时长\n"
        "```\n"
    ),
}

ANCHOR = "---\n\n## 1."

for rel, body in diagrams.items():
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        print("MISSING:", rel)
        continue
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if "## 图示速览" in content:
        print("SKIP (already has 图示速览):", rel)
        continue
    insert = "\n## 图示速览\n\n" + body + "\n"
    idx = content.find(ANCHOR)
    if idx == -1:
        # 兜底：直接插在第一个 '## 1.' 之前
        h1 = content.find("## 1.")
        if h1 == -1:
            print("NO ANCHOR:", rel)
            continue
        content = content[:h1] + "## 图示速览\n\n" + body + "\n\n" + content[h1:]
    else:
        cut = idx + len("---\n")
        content = content[:cut] + insert + content[cut:]
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("OK:", rel)

print("DONE")
