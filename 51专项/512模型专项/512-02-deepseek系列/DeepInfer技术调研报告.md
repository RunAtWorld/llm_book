# DeepInfer深度调研：真正让推理走向普惠

## 0. 摘要

- **定位**：DeepInfer 不是又一个通用推理框架，而是 **DeepSeek 自家大规模云服务的生产级引擎**。它基于 vLLM 主体二次开发（借鉴 SGLang / TensorRT-LLM 思路），但砍掉了通用抽象层，只为 **一个模型（DeepSeek-V4.1-Flash）** 和 **两类旗舰硬件（NVIDIA SM100 / 华为昇腾 950）** 做极致优化。
- **发布是"一对"，不是"一个"**：**DeepInfer**（Apache-2.0，推理引擎）+ **deepseek-recipe**（MIT，生产级 token 编解码 / 三协议转换）。前者解决"算得快"，后者解决"算得对、答得像官方"——少了 recipe，"第三方部署对齐官方 API"这句话就不成立。
- **一句话结论**：这次开源真正值钱的不是代码，而是**"模型架构 × 缓存生命周期 × 硬件调度"三者协同设计的方法论**——它把推理成本降低的路径，从"堆卡 + 调 kernel"改写成了"改模型结构 + 重定义缓存层级"。
- **官方自己的叙事骨架是三条线**（本文沿用该结构组织全文）：
  - **吞吐线**——把 GPU 空转压到最低：节点内 EP、Prefill 小 EP / Decode Wide-EP 的相反取值、四类负载均衡器、DSpark 变长验证、异步调度、容错。
  - **成本线**——把重复 prefill 降到最低：KV 留存 >72h、Block KV 落独立 3FS、SWA 不落盘 + Bounded Replay。
  - **延迟线**——TTFT 与 TBT 分别优化：E/P/D 分离、chunk 按时间预算反解、分词分段并行、边生成边返回、工具参数片段直出。
- **最值得关注的创新**（按对算力侧的影响排序）：
  1. **SWA Bounded Replay**：用"只重放 128 token 的近似重建"换掉 SSD 上的 SWA 持久化，decoder 侧 prefill 计算量近乎减半，持久化 KV 降到 V4-Flash 的 1/8。
  2. **反直觉并行布局**：Prefill 走**小** EP/DP（2~8），Decode 走**Wide-EP**（64~128，单通信域）。decode 的瓶颈从算力转向**互联**。
  3. **KVCache 升格为存储系统**：独立 3FS 集群 ~100 PiB SSD、TTL 3~7 天、峰值读 10 TiB/s，让"任意 prefill 机器命中任意前缀"成为可能，代价是放弃前缀亲和路由。
  4. **四类负载均衡器共用一张二次代价模型 `cost(q,c)`**：把调度从启发式变成"对代价模型反解"，线上 moe_lb 压到 1.03~1.09。
  5. **昇腾是一等公民**：`tilelang_ops/` 下 **cuda 30 个 / asc 30 个算子一一对应**，并 vendor 了 DeepGEMM-Ascend、DeepEP-Ascend。
- **能力画像**：吞吐与单位算力产值是当前公开数据里的天花板（单集群 155 万亿 tokens/天、约 38 USD/GPU·h 收入）；但**模型覆盖度=1、无在线 serving 外壳、硬件仅两代旗舰**，不是 vLLM/SGLang 的替代品。
- **开放边界**：Apache-2.0 开源的是**引擎 + 离线推理 + 协议层**；**KVCache 存储系统、E/P/D 分离编排、DP 副本间负载均衡**因与内部基础设施耦合，**未完整开源**（只有设计文档）。

---

## 0.2 三条优化主线

DeepSeek 对外讲这套系统时，用的是"吞吐 / 成本 / 延迟"三条线，而不是"模块清单"。**这个组织方式本身值得抄**——它把每个技术动作都锚定到一个可度量的经营指标上。

| 主线 | 要解决的真问题 | 关键动作（官方口径） | 官方自报收益 | 本文对应 |
|---|---|---|---|---|
| **吞吐线** | GPU 空转 = 直接烧钱 | ① 节点内 MoE 专家并行（EP）做大 batch；② 并行度反向取值：Prefill EP2~8 取小，Decode EP64~128 取大；③ 四类 LB 共用一张代价模型；④ DSpark 置信度变长验证替代固定 5 token；⑤ 异步调度填满 CPU/GPU 缝隙；⑥ 容错靠"多留机器 + 请求重路由 + 降级重启" | moe_lb 1.0320（EP2DP2）/ 1.0618（EP4DP4）/ 1.090（Decode）；变长验证 **+6%**；LB 达完美均衡的 ~80% | §2.2 / §2.3 / §2.6 |
| **成本线** | 重复 prefill = 白算 | ① KV 长期留存，设计目标 **>72h**（线上 agent 3 天 / chatbot 7 天）；② Block KV **不进本机内存**，128 token 切块落独立 3FS（~100 PiB / 读 10 TiB/s / 写 ~850 GiB/s）；③ SWA **20 层 × 134 token 的 ring**，只留本机内存、不持久化（TTL ~180s，命中 >98%）；④ V4.1 弃用"SWA 落盘"，改为 Bounded Replay | decoder 侧总 prefill 计算量**近乎减半**；持久化 KV 降至 V4-Flash 的 **1/8**；SWA 释放近百 PiB 级容量 | §2.4 / §2.5 |
| **延迟线** | 首字慢、吐字抖 | ① Encoder/Prefill/Decode 分离部署，TTFT 与 TBT 解耦；② chunk size 由目标时延**反解**；③ 分词从线性改分段并行（special token 切段独立编码后按序拼接，核心路径 Rust）；④ 跨 chunk 状态机缓冲不完整 UTF-8、KMP 增量匹配分隔符；⑤ 工具参数按 DSML 边界直出 JSON 片段，不构造完整对象；⑥ 投机解码顺带压延迟（按接受率分配验证算力） | 线上 TTFT ~400ms；平均输出 ~230 tok/s（代码 ~300） | §2.1 / §2.6 / §2.7 / §2.9 |

**读法提醒**：三条线不是并列的三块工作，而是**互相抢资源**的三角——KV 留存越久，成本线越省但存储越贵；decode batch 越大，吞吐越高但单用户速度越低；chunk 越大，吞吐越高但 TTFT 越差。DeepInfer 的实质贡献是把这个三角上的**操作点**用一张代价模型显式标定出来了（详见 §2.3）。

---

## 0.3 官方三个设计目标

这套系统的所有架构选择，都能回溯到发布时写下的三个目标。**看目标比看结论更能迁移**：

| # | 官方目标 | 原文要义 | 直接推导出的架构选择 |
|---|---|---|---|
| 1 | **支持多厂商加速器** | 在不同硬件平台上跑起来，并吃满各自性能 | 平台抽象层 `hardware/{nvidia,ascend}.py`；`tilelang_ops/cuda` 与 `asc` 各 30 算子一一对应；vendor DeepEP-Ascend / DeepGEMM-Ascend（§2.8） |
| 2 | **围绕商品化硬件组织 P/D** | prefill 用普通 8 卡机，decode 用普通 64~128 卡超节点，计算/通信/调度按两阶段负载特征分别设计 | 并行度反向取值（小 EP vs Wide-EP）；EPD 三级分离；TTFT/TBT 解耦优化（§2.1、§2.2） |
| 3 | **KV 留存 >72 小时** | 给上下文复用一个长窗口，命中就省掉重复 prefill，同时承担相应的容量与管理开销 | KV 从"显存 buffer"变为**独立存储系统**；TTL 分级（agent 3 天 / chatbot 7 天）；SWA 因生命期只有分钟级而**被赶出 SSD**（§2.4、§2.5） |

> **目标 3 是整份报告里最容易被低估的一句。** 它意味着"是否值得为缓存付存储费"这个决策，被提前到了**模型架构设计阶段**：正因为承诺了 72h 留存，才必须把全局 KV 压到 890 B/token、把持久化 KV 压到 1/8、把 SWA 踢出持久层。反之，如果只做 1 小时缓存，V4 那套"SWA 落盘"反而更简单。**缓存策略不是运维参数，是架构约束。**

---

## 1. 定位与发布背景

| 维度 | 说明 |
|---|---|
| 开发者 | DeepSeek-AI |
| 发布时间 | 2026-10-20（同日开源 `deepseek-recipe`） |
| 定位 | 大规模云服务推理框架（生产同款引擎，非实验室 demo） |
| 开源协议 | DeepInfer: Apache-2.0；deepseek-recipe: MIT |
| 代码基线 | "originally built primarily on **vLLM**"，借鉴 **SGLang** / **TensorRT-LLM** |
| 支持模型 | 仅 **DeepSeek-V4.1-Flash**（默认 checkpoint `DeepSeek-V4.1-Flash-1017`） |
| 支持硬件 | **NVIDIA SM100**（B200/GB200 级，compute capability 10.0）、**华为昇腾 950** |
| 硬件组网 | 常见商品化组网：**8 卡机做 prefill + 64~128 卡超节点做 decode**，兼容多厂商芯片 |
| 软件栈要求 | Linux、GCC ≥ 13、Python 3.12、CUDA 13（NVIDIA）/ **CANN 9.2（昇腾）** |
| 运行形态 | **仅离线推理**（vLLM 风格 in-process `LLM.generate()`）；协议转换由 recipe 以**库**形式提供，**无 HTTP server** |
| 实测规模 | 单集群 **155 万亿 tokens/天**；约 **38 USD/GPU·h**（谷价收入口径） |

### 1.1 经济性换算

官方给的是一句"38 美元/卡时（谷价）"，换算成国内团队能直接对标的口径如下：

| 口径 | 数值 | 折算说明 |
|---|---|---|
| 卡时收入 | **38 USD / GPU·h** | 按 1 USD ≈ 6.68 CNY → **≈ 253.8 元/卡时** |
| 单卡日收入 | — | 253.8 × 24 ≈ **6,091 元/卡·天** |
| 单卡月收入 | — | 6,091 × 30 ≈ **18.3 万元/卡·月** |
| 报道口径 | 18.57 万元/卡·月 | 智东西原文数字，反推汇率约 **6.79**（与卡时那处的 6.68 不一致） |

> **三条引用红线（写好汇报再发出去前请逐条核）**：
> 1. **这是"收入"不是"毛利"**。官方原文写的是 "overall revenue of US$38 per GPU-hour (at valley pricing)"。部分二手报道标题用了"毛利"，但正文与官方口径均无毛利数据——**不要写成"毛利 38 美元/卡时"**。
> 2. **"谷价"意味着这是最低档定价**。谷价时段的单位算力收入，被当成全天平均售价会系统性高估。
> 3. **汇率口径不统一**。同一篇报道里，卡时换算用 ≈6.68、月收入隐含 ≈6.79，差额约 1.6%。内部对比时统一用哪个都可以，但要在脚注写清楚。

**发布逻辑**：DeepSeek 已经把模型、TileLang、推理算子库（FlashMLA / DeepGEMM / DeepEP / TileKernels / DeepSelect）陆续开源，这次补齐最后两环——把它们串成一条可复现的**执行路径**（DeepInfer），以及让第三方部署**在 API 行为上与官方逐字节对齐**的协议层（deepseek-recipe）。官方表述是"为善与众行之"，实际效果是：**第三方私有化部署第一次可以同时对齐官方 API 行为与官方级性能基线**。

**为什么是"框架专用化"而不是"给 vLLM 提 PR"？** 通用框架要兼容数百种架构和消费级硬件，必然引入泛化抽象；DeepInfer 只服务 MLA/CED + MoE + Wide-EP + 两代旗舰卡，可以做到"零冗余抽象、直通 kernel"。这也是它敢公布 38 USD/GPU·h 单位经济性的底层原因。

---

## 2. 核心架构与创新点

> **关键判断**：DeepInfer 的能力有一半来自**模型侧**。V4.1-Flash 的 CED 架构、CSA2 跨层 KV 复用、FP4 主 KV 都是"为部署而设计"的产物（arXiv:2609.19969）。**只读框架代码而不看模型结构，会误判这些优化的来源。** 下表先给出模型侧的硬约束，再逐条拆框架侧实现。

### 2.0 模型侧前提

| 项 | 数值 | 对推理系统的含义 |
|---|---|---|
| 参数 | 552B 主干 MoE + 196B Engram（口径见备注） | 单机放不下，必须 EP 切分 |
| 激活参数 | **prefill 8B / decode 16B**（非对称） | Agent "读多写少"负载下，prefill 便宜一半 |
| 层结构 | 40 层 = 20 层 Causal Encoder + 20 层 Decoder | prefill 只跑 encoder，decoder 全局 KV 由 encoder 末层隐状态**投影**得到 |
| MoE | 1 共享 + 384 路由专家，每 token 激活 6 个 | 384+32 冗余专家 → decode 侧 EP64/EP128 的切分粒度 |
| 全局 KV | **890 B/token**（≈ V4-Flash 的 1/4） | 1M 上下文 ≈ 890 MB，80 GB 卡可承载 80+ 条百万上下文会话 |
| 持久化 KV | ≈ V4-Flash 的 **1/8** | 直接决定 SSD 采购量与 TTL 策略 |
| 低精度契约 | 主 KV：per-16 **e4m3**（类 NVFP4，去掉全局 scale）；Indexer KV / MoE 路由权重：per-32 **UE8M0**（MXFP4）；SWA KV：**MXFP8 528 B/token** | **硬件不支持这些格式就出局**（详见 §8） |
| 上下文 | 1M 输入，最大 384K 输出 | 倒逼 chunked prefill 与分层缓存 |

> 口径备注：552B 为主干参数量；含 Engram 条件记忆模块的总参口径在不同二手源中为 748B~763B，本文统一采用论文口径"552B backbone + 196B Engram"。

### 2.1 三级分离架构

> 主线归属：延迟线

```
API 请求 → [deepseek-recipe 编解码] → Entrypoint(路由/前缀匹配)
                                          ├→ Vision Encoder（单卡，便宜的卡）
                                          ├→ Prefill 引擎（8 卡机，EP2~8/DP2~8，控 TTFT）
                                          ├→ Decode 引擎（64~128 卡超节点，Wide-EP，控 TBT）
                                          └→ 3FS 集群（Block KV，~100 PiB，TTL 3~7 天）
```

- **三级分离（EPD）**：Vision Encoder / Prefill / Decode 独立部署、独立扩缩容，分别优化 TTFT 与 TBT。ATTN 层走 **DP（#DP=#EP）**，MoE 层走 **EP**。
- **TTFT 与 TBT 解耦**：这是"为什么要分离"的核心理由——prefill 是算力密集（大 batch 高效），decode 是访存密集（大 batch 反而拖慢单用户速度），两者的最优并行度方向**相反**。
- **注意这条链路的第一跳**：请求进引擎之前先过 recipe。它不是"可选的 SDK"，而是官方给出的**协议边界**——协议转换、prompt 编码、流式解析都在这一层完成，引擎只见 token（详见 §2.9）。

### 2.2 并行布局：小 EP 与 Wide-EP

> 主线归属：吞吐线

| 阶段 | 并行配置 | 取值原则 | 原因 |
|---|---|---|---|
| Prefill | **EP2~8 / DP2~8** | 显存够就用**最小**的 EP/DP | 批内工作量足（一次几千到几万 token），小 EP 通信开销低、计算效率高 |
| Decode | **EP64~128 / DP64~128** | 卡够就用**最大**的 EP/DP | 每步单请求工作量极少，必须靠跨请求大 batch + 专家切分降低**每步访存量** |
| Vision Encoder | 单卡 | 用可用范围内**最便宜**的卡 | 纯单卡计算，无通信 |

- **Wide-EP 的收益**（官方 EP8 vs EP64 曲线，解码、ctx 128K、nextn=3）：在 **200 tokens/s/user** 这一档，EP64 约 **19k tokens/s/GPU**，EP8 仅约 **10.5k**，**接近 2 倍**；且 EP8 在 32 req/rank 之后就急剧衰减，EP64 一直到 256 req/rank 仍在爬升。
- **"商品化硬件"这个设计目标的落点**就在这里：不是买最贵的机器，而是**用两类普通机器拼出不同的互联域规模**——8 卡机做 prefill，64~128 卡超节点做 decode。这条选择的代价是集群形态变成"哑铃型"（详见 §6 趋势 4）。
- **对算力团队的含义**：decode 的性能天花板**不在单卡算力，而在 scale-up 域的集合通信能力**（all-to-all、专家权重搬移）。这也是"超节点"叙事在推理侧第一次有了硬数据支撑。

### 2.3 四类负载均衡器

> 主线归属：吞吐线

核心思想：把注意力子层的计算量拟合为 **`cost(q, c)`**（`q` = 未命中 token 数，`c` = 已算完的前缀长度，二次代价模型），所有调度决策最终都归约到 **compute-quota** 这一个标量上。

**Prefill Load Balancer（四层）**
1. **分桶**：按输入总长分 4 档（<16K / 16–64K / 64–256K / ≥256K），每次按上一轮 compute-quota 需求占比动态重分引擎台数；
2. **DP 组间**：按剩余 quota 拉取，**不做前缀亲和路由**（因为任何引擎都能命中任何前缀——这是 3FS 独立存储换来的自由）；
3. **DP 组内**：逐请求分给当前 quota 负载最低的 rank；
4. **单卡内**：FIFO 出队，**chunk size 不固定**——每步给定纳秒级时间预算，对 `cost(q,c)` **反解二次方程**，取恰好填满预算的最大 `q`（上限 32768 token），组内再 all-reduce MIN 收敛预算。

**其余三类**
- **Decode LB**：跨所有 decode 实例的所有卡统一路由，同时均衡**请求数**（决定批大小）与 **KV 总长**（决定注意力代价）；短请求发给"并发低但 KV 总长偏大"的卡，长请求发给"KV 总长小"的卡（参考 DualPath, SIGCOMM'26 组内调度）。
- **Vision Encoder LB**：以机器为粒度，按 `(w/patch)×(h/patch)` 估算 patch 数，分给当前 patch 最少的卡。
- **Expert-Parallel LB（EPLB）**：每卡常驻专家之外多备**冗余专家**放热门专家副本，统计窗口 N 步、每 M 步重排，权重在 scale-up 网络内经 `symm_mem` 组内搬移。

| EPLB 实测（线上） | Prefill | Decode |
|---|---|---|
| 冗余专家起点 | 4 个/卡 | 1 个/卡（EP64 取整后 7 个/卡，不含共享专家） |
| 统计窗 / 重排周期 | 100 步 / 1000 步 | 1000 步 / 10000 步 |
| `moe_lb`（最慢 rank/均值） | **1.045（EP2DP2）/ 1.074（EP4DP4）** | **1.090** |

> 注：两个来源的 prefill 读数不一致——智东西报道沿用知乎原文的 **1.0320（EP2DP2）/ 1.0618（EP4DP4）**，官方 Q&A 文档给的是 **1.045 / 1.074**（疑为不同版本统计口径）。两版均记录于此，引用时建议标注来源文档。
> 整体 LB 效果：达到"完美均衡"离线基准的 **约 80%**。

### 2.4 KVCache 存储系统

> 主线归属：成本线

传统框架把 KV 当"显存里的一块 buffer"；DeepInfer 把它拆成**生命周期完全不同的三类对象**，分别放在三种介质上：

| 对象 | 介质 | 副本 | 生命周期 | 管理方 |
|---|---|---|---|---|
| 运行中 Block KV + State KV | 推理卡 HBM | — | 请求级 | P/D 实例 |
| **SWA KV**（存于 state cache 的 `StateBlock`） | 本机 host 内存 | 单副本（3FS 上没有），跨机命中走 400G VLAN 拉取 | **TTL ≈ 180 s，命中率 > 98%** | P/D 实例 + Entrypoint 内存目录 |
| **Block KV（分页）** | **独立 3FS 集群** | 应用层无副本，靠 3FS 自身冗余（EC 10+2） | **agent 3 天 / chatbot 7 天** | `kvcache-store` |

**Block KV 关键数字**
- 128 token / block，存 **4 层 main KV + indexer K**；block 粒度约 **112 KiB**；
- 容量 **~100 PiB SSD**；峰值聚合读 **10 TiB/s**、写 **约 850 GiB/s**（针对 112 KiB 随机读写专门优化）；
- 存储配比经验值：**约 14.4 TB 逻辑容量 / 每张推理卡**；单存储服务器建议 20×30.73 TB SSD，≥40 GB/s 读、5 GB/s 写；
- TTL 代价：3 天 TTL 会让 chatbot 流量命中率下降 **0.5%~1%**（chatbot 长尾命中更重——少量 KV 被读很多次，丢了整体命中率掉得明显）。

**两个容易被漏掉的运维细节**
- **`kvcache-store` 本身没有过期逻辑**，回收完全交给 3FS 自身的 GC。所以"TTL"实际上是"GC 容忍窗口"，容量规划做错时表现为**GC 压力**而不是**过期报错**。
- **decode 侧不做 KV offload、也不启用 Hi-Sparse**（decode 不受缓存容量约束）；但 decode 阶段产生的 KV **仍然写盘**，纯粹是为了抬命中率。这是一个很"反直觉但省钱"的选择：写放大约 850 GiB/s 的量，换回来的是下一轮请求的 prefill 省掉。

**设计取舍（值得学）**：用**独立存储集群**而不是本机/多级缓存，换来三件事——① 任意 prefill 机器可命中任意前缀，避免"前缀亲和 vs 负载均衡"的路由冲突；② 存储与计算的扩缩容解耦；③ 缓存不受计算节点故障影响（计算节点故障率比存储高一个数量级以上）。代价是放弃前缀亲和路由、多一次网络往返。

**命中路径三段，只有第一段在关键路径上**
1. **匹配**：Entrypoint 在本地 `kvcache-store`(CHECK 角色) 做哈希链前缀匹配，只查元数据——但命中长度必须在路由前算出，**无法掩盖**；
2. **取 KV**：独立进程异步，把 KV 从 3FS 搬到宿主机 shm；
3. **换入**：shm → HBM，engine 进程内线程做**逐层 H2D 与前向计算重叠**；因为 CUDA copy engine 对 KV block 这种细粒度拷贝性能差，改用 **GDR（RDMA 网卡打满 PCIe 带宽，且不占 SM）**。

### 2.5 SWA Bounded Replay

> 主线归属：成本线

**问题**：V4.1 用 128 token 窗口的 SWA，窗口外 KV 会丢。要精确重建一个窗口，需要回放 `层数 × 窗口` 个 token（decoder 侧 20 层 × 128 = **2560 token** 的完整前向）。V4 的做法是**把 SWA 落盘 + 未命中精确重算**——代价是 SWA 占去持久 KV 近一半容量，却换不到对应命中收益（SWA 只有分钟级生命期，却占着天级 SSD）。V4 论文提的 "Zero SWA Caching"（完全不落盘、只靠精确重算）在生产上又太贵，**V4.1 实际没采用**。

**方案**：只重放**最近 128 个 token**，并把注意力窗口截断在重放段内——从位置 `s` 起回放，位置 `i` 的 query 只与 `max(s, i-127) ~ i` 的 key 计算，得到**近似**的 SWA KV。代价从"与上下文长度和层数相关"降为**常数**。

两处落地：
- **Encoder SWA Bounded Replay**：前缀缓存命中长度从此只依赖 Block KV；SWA 缺失时重算命中前缀最后 128 token，与未命中后缀一起前向，重算段**只重建 SWA**，Block KV 原样复用、不重算不覆写。
- **Decoder SWA Bounded Replay**：decoder 的 Block KV 本就由 encoder 投影得到，只有 SWA 需自算；改为每次 prefill 在 decoder 层只重算 prompt 末尾 **128 token**，产出仅供 decode、不进前缀缓存 → **总 prefill 计算量近乎减半**。

**SWA 的物理形态：一个 20 层 × 134 token 的 ring**

| 项 | 数值 / 规则 |
|---|---|
| 每序列存储量 | `20 层 × 134 token × 528 B = 1,415,040 B ≈ 1.35 MiB`（528 B/token 即 MXFP8 SWA 格式） |
| 134 的构成 | **128 token 窗口 + 6 token 投机回退**；ring buffer 比窗口多出 6 个槽位，正好吸收"decode 阶段拷贝位置最多超出 128 边界 5 个 token"的溢出 |
| 存放位置 | state cache 的固定大小槽位 `StateBlock`，按 `[已存区: SWA] [丢弃区: YOCO consumer SWA]` 切成两段；丢弃区里还有 C2 压缩边界累加器 `ca_state`/`ica_state`（压缩窗口天然与 128 token 的存储粒度对齐，不存在跨边界依赖） |
| 存储粒度 | **只存 128 token 整数倍的位置**。prefill 结束后，保留序列尾部 1 个 token 再向下取整到 128；不足一个完整 128 单元则本轮不存。chunked prefill **只在最后一个 chunk 存**，中途不存 |
| decode 侧 | 在显存里另开一个 replica 区，每跨一次 128 边界就把当前窗口拷进去；请求结束时从 replica 区最近一次完整拷贝取数，挂载点向下取整到 128 |
| 命中上限 | 输入长 `L` 时，最大命中长度 **≤ L − 134**。在这个 cap 下先查 Block KV 得 `L1`（128 倍数），再查 SWA：`[L1−128, L1]` 区间有 SWA 则命中 `L1`，否则退到 `L1−128` |

> **口径澄清（v2 的一处模糊）**：`134` 是**存储槽位**里的 token 数（128+6 投机回退），而源码里 `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`（128+12）是 **prefill 重算时保留的长度**参数，两者不是同一个数，不能混引。智东西报道的"20 层 × 134 的 ring"与官方 Q&A 的 `1,415,040 B` 完全一致，可视为存储侧权威口径。
>
> 默认参数（源码）：`DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`、`DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`。

**代价要记住**：重建出的状态**不是数学等价**的——同一前缀在不同命中位置算出的结果会有细微差别。这是"近似"换"减半"的显式交易，对严格可复现场景是隐患。

### 2.6 DSpark 与异步调度

> 主线归属：吞吐线 × 延迟线

**DSpark（算法侧）**：在 DFlash 的 base logits 上叠加一个**只依赖前一个 token id 的低秩 logit bias**，用极轻量的串行修正补回相邻 token 的局部条件化信息，提升 draft 准确率。

**变长验证（系统侧）**：固定验证 5 个 token 会把算力浪费在低接受率 token 上。DSpark 用 **Confidence Head** 预测每个 draft token 的接受率，按"预期接受率排序 + 硬件吞吐曲线"挑**有效吞吐最大的前缀子集**验证。线上 draft 一次生成 5 个。

| 并发（seq/rank） | 固定-5 验证长度 | 变长验证长度 | 固定-5 吞吐 | 变长吞吐 | 收益 |
|---:|---:|---:|---:|---:|---:|
| 16 | 5.00 | 4.44 | 4891 | 4816 | **−1.5%** |
| 32 | 5.00 | 3.65 | 8964 | 9111 | +1.6% |
| 48 | 5.00 | 3.94 | 12032 | 12467 | +3.6% |
| **60（生产工作点）** | 5.00 | 3.58 | 13730 | **14466** | **+5.4%** |
| 72 | 5.00 | 3.24 | 14962 | 15961 | +6.7% |
| 96 | 5.00 | 3.24 | 16817 | 18660 | **+11.0%** |

> 单位：tok/s/rank（EP64/DP64、全接受折算）。**低并发反而变慢**——变长验证是"大 batch 才划算"的优化，这是一个很实用的调参直觉。官方对外口径的"吞吐 +6%"即取此表 60 工作点附近的量级。

**异步调度（Zero-Overhead Batch Scheduler）**：decode 第 N 步 `schedule → forward → sample → post-process`，其中 schedule/post-process 是纯 CPU 逻辑，但 post-process 前要 GPU-CPU 同步拿 sample 结果 → GPU 空转。解法是把流水线重叠：GPU 跑 `forward(N)→sample(N)` 时，CPU 同步跑 `schedule(N+1)→launch_forward(N+1)→post-process(N-1)→launch_sample(N+1)`。默认开启。

**配合难点**：`schedule(N+1)` 时第 N 步的 confidence 还没回来 → CPU 只定总预算，逐请求长度改在 **GPU 上用 TileLang kernel 重算**（累乘 + 全局排序取 top-k + 融合 remap scatter），结果回写 pinned buffer。

### 2.7 其他工程细节

| 能力 | 实现要点 | 备注 |
|---|---|---|
| **结构化输出** | 格式约束（JSON Schema / Regex / EBNF）编译成语法自动机，**独立 SO 机器异步编译**；decode 每步推进自动机→合法 token 集合→mask 采样 | 编译与推理解耦，进引擎前取回结果 |
| **Batch-invariant** | 不同 batch 组合下结果完全一致，`DEEPINFER_BATCH_INVARIANT=1` | **仅 NVIDIA** |
| **确定性采样** | 相同输入 + 相同采样参数输出完全一致（固定 `seq_seed`） | **仅 NVIDIA** |
| **自动扩缩容** | prefill 按"未命中 token 数"定机器数，decode 按"在算请求数"调；调整有速率限制与上下界；配额占用率 > 0.8 禁止缩容 | 见 §5.2 |
| **长尾请求处理** | decode 实例收到下线通知后进入 **1 分钟宽限期**，未完成请求连同 KV **一次性整体迁移**到其他实例 | 大 EP 下避免资源空转 |
| **端到端协议层** | 三协议归一、分段并行分词、防 special token 注入、流式状态机 | 已独立成节，见 **§2.9** |

### 2.8 昇腾是一等公民

| 证据 | 具体内容 |
|---|---|
| 构建开关 | `export DEEPINFER_ACCELERATOR=ascend`（与 `nvidia` 并列），依赖 `requirements-ascend.txt`：`torch==2.13.0+cpu` + `torch-npu==2.13.0rc1` |
| 软件栈 | **CANN 9.2** |
| 通信 / 矩阵库 | 仓库 `vendor/` 下直接 vendor 了 **DeepEP-Ascend** 与 **DeepGEMM-Ascend** |
| 算子层 | `deepinfer/model_executor/layers/tilelang_ops/` 下 **`cuda/` 30 个算子文件、`asc/` 30 个算子文件，一一对应**（本次实测仓库文件数） |
| 平台抽象 | `deep_cluster/hardware/ascend.py`（与 nvidia 平台并列的硬件描述） |
| 设备可见性 | 未设 `CUDA_VISIBLE_DEVICES` 时自动回退 `ASCEND_RT_VISIBLE_DEVICES` |

**30 个对等算子清单**（可作为内部算子完备性 checklist）：
`candidate_blocks`、`compress`、`compute_seqlens_k_zos`、`dflash`、`dflash_copy`、`flash_mla_utils`、`fused_sample`、`fused_transform`、`gather`、`gather_engram_tokens_zos`、`gather_input_ids_zos`、`get_kv_index`、`image_embed_ops`、`kv_index_cat`、`kv_quant`、`load_paged_kv_cache`、`norm`、`prepare_cp_mirror`、`prepare_v4_decode_metadata`、`random_sampling`、`rstd_1ulp`、`sampler`、`schedule_mtp_varlen`、`state_cache`、`store_kv_cache`、`store_mtp_info`、`structure_output`、`sum_smaller_probs`、`swa_index`（+ `__init__`）

> **意义判断**：过去国产芯片跑 DeepSeek 新模型，通常是厂商自己维护 vLLM/SGLang 分支，功能与性能永远跟在后面一排。这次昇腾与 NVIDIA **共用同一个引擎、同一套算法、同一份 30 算子接口**，差距被压缩到"硬件本身 + 底层算子实现质量"两项——这对昇腾生态是结构性利好，同时也把竞争压力直接压到了**算子质量与互联带宽**上。

### 2.9 deepseek-recipe

> 主线归属：延迟线 + 迁移成本

这一节是 v3 新增。v2 把它当成"工程细节表里的一行"，但官方是把 DeepInfer 与 recipe **并称开源**的——**没有它，"第三方部署与官方 API 行为一致"这句承诺就不成立**。

| 维度 | 内容 |
|---|---|
| 形态 | **Rust 实现 + Python 绑定**；是**库**，不是 server |
| 协议覆盖 | Chat Completions / Responses / Messages 三套 API 归一为**同一种会话格式**，共用一套模型模板与编码逻辑，生成对应模型版本的 prompt 与 token 序列 |
| 与官方对齐 | 处理逻辑与 DeepSeek 官方 API **完整对齐**，第三方部署可获得与官方一致的 API 体验 |
| 输入编码 | 构建 prompt 时**显式记录 special token 及其位置**，编码时直接映射为对应 token ID；普通文本走基础词表。用户输入/工具结果/工具参数里出现的 `<｜User｜>`、`<｜image｜>` 等字符串被编码成**普通 token 序列**，防止文本注入控制 token |
| 分词加速 | **分段并行**：special token 之间的普通文本可独立做 BPE 编码再按序拼接，长上下文的 tokenize 从"线性"变"并行"——直接优化 TTFT；核心路径 Rust |
| 输出解析 | token 流 → 事件（正文 / 思考内容 / 工具调用…）→ 各协议 streaming 响应；按 token ID 识别 `</think>` 与 DSML 中的 `｜DSML｜`，模型用普通 token"拼"出同样文本时按字面保留，不获得触发能力 |
| 流式状态机 | 跨 chunk 缓冲不完整 UTF-8 码点；按思考/正文/工具参数分阶段维护状态；**KMP 增量匹配**分隔符，不回溯已确定文本；工具参数按 DSML 边界**直接吐 JSON 片段**（字符串逐段转义），不构造完整对象 —— 缩短首个参数片段的等待时间 |

> **为什么这件事比看起来重要**：私有化部署最常见的阻塞项不是"跑不快"，而是**"跑出来和官方 API 不一样"**——prompt 模板差一个换行、special token 处理方式不同、工具调用 JSON 的转义规则不一致，都会导致同一份评测集在客户环境里分数掉一截，然后归因到"芯片不行"。recipe 把这条不确定性直接消掉了。**对交付团队，这是比 kernel 优化更快的信任建立手段。**

---

## 3. 开源边界

| 组件 | 是否开源 | 说明 |
|---|---|---|
| 推理引擎（模型执行、采样、并行、调度） | ✅ | 离线 `LLM.generate()`，vLLM 风格 API |
| 优化算子（FlashMLA / DeepGEMM / DeepEP / TileKernels / DeepSelect / TileLang ops） | ✅ | 含昇腾对等实现 |
| DSpark + 置信度变长验证 | ✅ | `DEEPINFER_ENABLE_MTP_VARLEN=1`，`DEEPINFER_USE_NEXTN_DECODE` 控 draft 长度（默认 5） |
| SWA Bounded Replay | ✅ | 见 §2.5 默认参数与 ring 结构 |
| 异步调度 / 结构化输出 / batch-invariant / 确定性采样 | ✅ | 后两者**仅 NVIDIA** |
| **deepseek-recipe（协议编解码 + 流式解析）** | ✅（MIT） | **库级**：三协议归一、分段并行分词、防注入、DSML 片段直出 |
| **KVCache 存储系统（3FS / kvcache-store / 命中管理）** | ❌ | 仅设计文档 |
| **E/P/D 分离编排** | ⚠️ 部分 | "多数引擎侧功能已实现"，编排层缺失 |
| **DP 副本间负载均衡（4 个 LB）** | ❌ | 仅设计文档 + 实测数字 |
| 在线 serving 外壳（HTTP server / 鉴权 / 限流 / 路由） | ❌ | 协议转换已由 recipe 解决，**业务外壳仍需自建** |

**结论**：开源版 ≈ "**高性能单机/单集群离线推理引擎 + 生产级协议层**"。要做成完整在线服务，还差 **KV 存储、PD 编排、跨副本 LB、HTTP 外壳**——前三个恰好是与内部基础设施耦合最深、也最能体现工程水位的地方。

> **别把"没有 server"读成"不能用"**：v2 写"无 OpenAI 兼容"容易让人误判成"协议都得自己写"。准确说法是——**协议这一层已经给了（recipe），缺的是把它挂到网络上当服务的那层壳**。这两件事的工作量和风险完全不同。

---

## 4. 性能对比

### 4.1 纵向对比：V4 与 V4.1

| 维度 | V4 推理系统 | **V4.1 推理系统** | 变化 |
|---|---|---|---|
| 全局 KV（HBM 常驻） | ~3.5 KB/token（约 4×） | **890 B/token** | ↓ ~75% |
| 持久化 KV（SSD/host） | 1× | **~1/8** | ↓ ~87.5% |
| SWA 策略 | 落盘 + 精确重算（回放 2560 token/20 层） | **Bounded Replay（只回放 128 token）** | prefill 计算**近减半** |
| SWA 占持久 KV 容量 | 近 1/2 | ~0（不落盘，1.35 MiB/序列只在本机内存） | 释放近百 PiB 级容量 |
| prefill 激活参数 | — | **8B** | Agent 读入成本结构性下降 |
| 解码 FLOPs（4K→1M，256×） | — | 仅 **+1/4** | 长上下文曲线近乎压平 |

### 4.2 横向对比：离线 Pareto

**Prefill（EP4DP4 | 384+32 专家 | Decoder Bounded Replay 140 token）**

| 每 rank 批 token | 非缓存输入吞吐（tok/s/GPU） | 每步延迟 |
|---:|---:|---:|
| 4,096 | ~60,000 | ~66 ms |
| 8,192 | ~71,500 | ~110 ms |
| 16,384 | ~81,500 | ~200 ms |
| 32,768 | ~86,000 | ~375 ms |
| 65,536 | **~88,500** | ~730 ms |

> 读法：prefill 吞吐在 16K 之后基本进入平台区（81k→88k，边际收益仅 8%），**延迟却翻了 3.6 倍**。这意味着线上"多塞一点 batch"换吞吐的空间很有限，真正的杠杆在 chunk 反解与 PD 配比。

**Decode（EP64DP64 | 384+64 专家 | nextn=3 | 单 token 接受率 75% → τ=3.25）**

| 每 rank 请求数 | 输出吞吐（tok/s/GPU） | 单用户速度（tok/s/user） |
|---:|---:|---:|
| 8 | ~2,400 | ~328 |
| 16 | ~5,050 | ~315 |
| 32 | ~9,500 | ~297 |
| 64 | ~15,700 | ~247 |
| 128 | ~22,800 | ~178 |
| 256 | **~26,300** | ~106 |

**EP8 vs EP64（同 ctx/nextn/接受率）**

| 单用户速度档 | EP8DP8 吞吐 | EP64DP64 吞吐 | 差距 |
|---|---:|---:|---:|
| ~100 tok/s/user | ~24,800 | ~26,300 | +6% |
| ~200 tok/s/user | ~10,500 | ~19,000 | **+81%** |
| ~220 tok/s/user | ~7,000 | ~17,000 | **+143%** |

> **读表结论**：Wide-EP 的收益与"单用户速度要求"强相关——追求极限吞吐时差距不大，但**一旦要求交互性（≥200 tok/s/user），小 EP 直接崩掉**。这是"decode 必须上超节点"的量化依据。

### 4.3 框架能力矩阵

| 维度 | **DeepInfer** | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|---|
| 维护方 | DeepSeek | vLLM 社区 | SGLang 社区 | NVIDIA |
| License | Apache-2.0（recipe 为 MIT） | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| 模型覆盖 | **1 个**（V4.1-Flash） | 200+ 架构 | LLM/VLM/扩散 | 官方支持列表 |
| 硬件 | **SM100 + 昇腾 950** | NVIDIA/AMD/Intel/CPU/插件（含昇腾） | NVIDIA/AMD/TPU/昇腾/摩尔线程 | NVIDIA 为主 |
| 专家并行 + EPLB | ✅（含冗余专家动态重排） | ✅ | ✅ | ✅ |
| Wide-EP | ✅（64~128，核心设计） | 支持 | 支持 | 支持 |
| 投机解码 | **DSpark + 置信度变长验证** | n-gram/suffix/EAGLE/DFlash | 类似 + SpecForge | ✅ |
| 协议层 | ✅ recipe（三协议 + 官方行为对齐） | 内置 OpenAI 兼容 server | ✅ | ✅ |
| PD 分离 | ❌ 本版不含编排 | ✅（P/D/encode） | 经 llm-d/Dynamo | ✅ |
| 在线 serving | ❌ 仅离线 | ✅ OpenAI 兼容 | ✅ | ✅ |
| 生产验证 | DeepSeek 官方大规模线上 | 海量企业 | 中大规模 | NVIDIA 生态 |
| 性能定位 | 该模型上的**上限参照** | 通用最优性价比 | 结构化/前缀缓存强 | NVIDIA 卡上极致 |

> **不要误读**：这张表不能得出"DeepInfer 更强"的结论。正确读法是——**当模型与引擎由同一团队设计时，系统能简化到什么程度**（模型覆盖从 200+ 降到 1，换来少一层抽象与 5~10% 的 kernel 级优势）。

### 4.4 线上生产数据

| 指标 | 数值 | 口径说明 |
|---|---|---|
| 单集群日吞吐 | **155 万亿 tokens/天** | 单一集群 |
| 单位算力收入 | **约 38 USD / GPU·h** | **谷价**、收入口径（非成本、非毛利） |
| 平均 TTFT | 约 **400 ms** | 线上混合负载 |
| 平均输出速度 | 约 **230 tok/s**（代码场景约 300） | 线上混合负载 |
| 平均输入缓存命中率 | **96%** | Agent/RAG 长前缀复用为主 |
| decode 生产工作点 | 约 **14K tok/s/卡** | ≈ 60 并发/rank，与 §4.2 曲线吻合 |
| LB 达到完美均衡的比例 | **约 80%** | 离线基准对照 |

### 4.5 单位经济性换算

| 视角 | 换算结果 | 备注 |
|---|---|---|
| 每卡每小时 | **38 USD ≈ 253.8 元** | 谷价、收入口径 |
| 每卡每天 | **≈ 6,091 元** | 24h 满负荷假设 |
| 每卡每月 | **≈ 18.3 万元**（30 天） | 二手报道写 18.57 万，汇率口径不同 |
| 每 100 卡每月 | **≈ 1,830 万元** | 规模化后的收入量级 |
| 每千卡每天 | **≈ 609 万元** | 便于和"单集群 155 万亿 token/天"摆在一起看 |

> **能用在哪**：这是目前公开数据里少有的**"单位算力产值"**锚点，可用于交付方案的经济性论证与客户 ROI 测算。
> **不能用在哪**：不能当毛利、不能当定价基准（谷价≠均价）、不能线性外推到其他模型（模型覆盖=1）、也不能假设客户能复现 96% 命中率（强依赖 Agent/RAG 长前缀流量形态）。

---

## 5. 成本与部署要点

### 5.1 部署最小单元与配比
- **1P + 1D 可跑通**，但 P:D 实际约 **1:1**，且 decode 推荐 EP≥64 → **最小高效部署单元 = 64 卡 prefill + 64 卡 decode = 128 卡**。这对中小客户是硬门槛。
- 跨数据中心部署**技术上可行**（KV 压缩后传输量可承受），带宽预算 **约 400 Gbps / 8 卡**；官方实际部署在"不同但相邻"的机房。
- 存储配比：**约 14.4 TB 逻辑容量/卡**（物理容量更大，含 EC 冗余）。

### 5.2 成本杠杆
1. **KV 留存 >72h 是设计目标，不是运维参数**：线上按流量形态分级——agent 3 天 / chatbot 7 天；chatbot 长尾命中重，3 天 TTL 只掉 0.5~1% 命中率，很划算。
2. **SWA 不落盘**：把"分钟级生命期（~180s）的数据"从"天级 SSD"上赶走，这是 V4→V4.1 最大的成本动作，释放近百 PiB 级容量。
3. **PD 比动态化**：prefill 按"未命中 token 数"、decode 按"在算请求数"分别决策；配额占用率 > 0.8 禁止缩容；调整有速率限制与上下界。
4. **长尾请求处理**：decode 实例下线前给 **1 分钟宽限期**，未完成请求连同 KV **一次性整体迁移**，避免大 EP 下的资源空转。
5. **decode 也写盘**：明知写入占 ~850 GiB/s，仍坚持把 decode 产生的 KV 落盘——用存储换下一轮 prefill。这条"看起来浪费"的决策，是 96% 命中率的一部分来源。

### 5.3 容错
- 一个实例的所有卡构成**单一通信域**，**任意一卡故障即整实例不可用**；重启在**分钟级**，容错靠"配额冗余 + 请求级重路由"，**不是**靠缩短重启时间。
- 重启时按**当前健康卡数重新确定 EP 规模**（拿到多少卡就用多少做 EP）——这个"降级重启"思路很实用。

---

## 6. 技术趋势洞察

1. **模型-系统 codesign 成为降本主路径**。CED（decoder KV 由 encoder 投影）、CSA2（跨层 KV 复用）、FP4 主 KV、SWA Bounded Replay——四个动作全部要求"训练时就按部署形态设计模型"。**框架内的优化空间正在收窄，架构层的空间正在打开。**
2. **KV Cache 从"显存管理"升级为"存储系统"**。100 PiB、10 TiB/s、天级 TTL、EC 纠删码、交给 3FS GC 回收——推理框架的边界第一次大幅伸进分布式文件系统领域。未来"推理工程师"要懂存储。
3. **负载均衡从启发式走向"代价建模 + 反解"**。一个二次模型 `cost(q,c)` 同时驱动分桶、组间、组内、单卡四层决策，且用"时间预算反解 chunk 大小"。这类**可拟合、可标定的控制器**会成为框架标配。
4. **Decode 的瓶颈从算力迁移到互联**。Wide-EP 64~128 单通信域 + mega MoE，把"超节点"从营销词变成性能必选项；prefill 反而回归小 EP 的 8 卡机。**未来集群是"哑铃型"：小互联域做 prefill，大互联域做 decode。**
5. **低精度进入"格式契约"阶段**。FP4（per-16 e4m3，无全局 scale）、UE8M0 缩放、MXFP8 SWA——这些不是可选项，而是**写进框架代码的硬件假设**。加速器竞争从"峰值算力"转向"格式支持 + 算子实现质量"。
6. **推测解码进入"算力精算"阶段**。从固定验证长度 → 置信度驱动的变长验证（−30% 验证量、+5~6% 吞吐），并与异步调度耦合（GPU 侧重算长度）。**低并发下反而变慢**，说明优化开始高度依赖工作点。
7. **厂商开始把"API 行为一致性"当交付物开源**。recipe 的意义在于把"prompt 模板 / special token / 流式解析 / 工具参数转义"这些**隐性契约**变成可复现代码。过去这类差异是私有化部署甩锅的灰色地带，现在它变成一条可测的基线——**对国产芯片尤其重要：模型跑分不对时，能先把"协议层没问题"这条排除掉。**
8. **框架生态分层**：通用框架（vLLM/SGLang）拼广度与社区，模型厂商专用框架拼深度与单位经济性。DeepSeek 选择"只开源引擎、不开源基础设施"，实质是把"可复制的机制"给社区、把"难复制的运维水位"留给自己。

---

## 7. 适用场景与局限

**最适合**
- 在 **SM100 或昇腾 950** 上、规模化（≥128 卡）部署 **DeepSeek-V4.1-Flash** 的团队；
- 需要 **batch-invariant / 确定性采样 / 结构化输出** 的生产系统（NVIDIA 侧）；
- **需要"与官方 API 逐字节对齐"的私有化交付**——直接用 recipe，能省掉一整类"结果不一致"的排查；
- 框架维护者与芯片厂商：把 DeepSeek 的并行布局、代价模型、低精度契约**移植**到自己的栈；
- 做**长上下文 Agent 服务**（输入重、输出短）——这是 V4.1 的设计靶心。

**局限与风险（上线前必须逐条确认）**
- **模型覆盖=1**：不是通用推理框架，服务量级小/消费级卡/其他模型，直接用 vLLM 更合适。
- **无在线 serving 外壳**：协议转换已由 recipe 解决（库级），但 HTTP server、鉴权、限流、路由、KV 存储、PD 编排、DP 级 LB 仍需自建。
- **硬件窄**：SM100与昇腾 950；H100 等上一代**不在支持列表**。
- **开源 ≠ 生产同款**：官方明确公开版 FlashMLA decode kernel 慢 **5~10%**，端到端差 **≤5%**；内部 draft 模型也更新一些。
- **近似回放非等价**：同一前缀在不同命中位置输出可能有细微差异；严格可复现要求下需谨慎（确定性采样仅 NVIDIA）。
- **长上下文"能塞"≠"能找"**：1M 上下文可装，但论文未提供长上下文稀疏检索压测，召回能力需自测（RULER / LongBench 分段测 128K/512K/1M）。
- **故障域大**：大 EP 实例单卡故障全实例不可用，重启分钟级；规模越大越依赖配额冗余。
- **数据全为自报**：155 万亿/天、38 USD/GPU·h、96% 命中率均无第三方审计；96% 命中率强依赖 Agent/RAG 的长前缀流量形态，**私有化场景难直接复制**。
- **别名/版本漂移**：默认 checkpoint 为 `V4.1-Flash-1017`，而部分性能与架构结论来自 0910 版论文，两者能力有差异（1017 为后训练强化，架构不变）。
---

## 8. 对算力团队的启示

> 受众定位：昇腾算力交付/开发，关注"模型跑得好不好、卡卖得值不值"。以下每条都尽量落到可执行动作。

**A. 把 DeepInfer 定位为"官方参考实现"，不是竞品引擎**
它的价值不在于被部署，而在于给出了一套**可对齐的基线**：并行布局、Pareto 评测方法、低精度格式契约、代价模型形态。建议团队内部指定 1~2 人做长期跟踪（仓库持续更新，官方承诺后续模型推理代码也放这里）。

**B. 用 30 个 `asc/` 算子清单做一次完备性体检**
`tilelang_ops/cuda/` 与 `asc/` **各 30 个文件、一一对应**。建议逐项对照自研实现，重点看这 8 个"最像 DeepSeek 私有算法"的：
`compress`（CSA2 压缩）、`kv_quant`（FP4/MXFP8 量化）、`swa_index`（SWA 索引）、`candidate_blocks`（层次化选块）、`dflash`/`dflash_copy`（投机解码）、`schedule_mtp_varlen`（变长验证调度）、`state_cache`/`store_kv_cache`（缓存存取）、`structure_output`（语法自动机 mask）。
缺哪一项，就意味着该模型在昇腾上要么跑不起来，要么性能/精度打折。

**C. 低精度是硬门槛，先做能力矩阵而不是先做性能**
框架里写死的格式假设：主 KV **FP4 / per-16 e4m3 / 无全局 scale**；Indexer KV 与 MoE 路由权重 **MXFP4 / per-32 UE8M0**；SWA KV **MXFP8（528 B/token）**。
→ 建议输出一张"昇腾各代产品 × 格式支持 × 是否有原生反量化通路"的表格；不支持的模块给出 FP8 降级方案，并**量化精度损失**（用 AIME/GPQA/MMLU-Pro 验证掉点是否 ≤1 分）。

**D. Decode 侧把优化重心押在互联**
EP64~128 单一通信域、mega MoE（scale-up 域内"无论 EP 大小、batch 大小都显著快于非融合 MoE"）、专家权重经 `symm_mem` 组内搬移。
→ 可量化的目标：在 **ctx 128K、nextn=3、接受率 75%、60 seq/rank** 的工作点上，对标官方 **约 14K tok/s/卡**；在 200 tok/s/user 的交互性要求下对标 EP64 的 **~19K tok/s/卡**。达不到时，先查 all-to-all 效率与专家权重搬移开销，而不是先查 GEMM 峰值。

**E. Prefill 侧的关键不是算力而是"代价模型标定"**
官方 chunk size 由 `cost(q,c)` 二次模型**反解时间预算**得到（上限 32768 token/步，组内 all-reduce MIN）。
→ 这个模型**强硬件相关**。建议为昇腾做一套"per-op profiling → 拟合二次系数 → 反解 chunk"的工具链，这是可复用的方法论资产，比调单个 kernel 的收益更持久。

**F. KV 分层存储是昇腾方案可以差异化的一环**
3FS 的三个数字（100 PiB / 10 TiB/s 读 / 112 KiB 随机粒度）+ 换入用 **GDR 绕开 copy engine、不占 SM**（因为 CUDA copy engine 细粒度拷贝性能差）。
→ 昇腾侧存在同样的"拷贝引擎不适合细粒度 KV 搬运"问题，是否能用 UB/RDMA 直通 HBM 做逐层 H2D 与前向重叠，值得作为专项立项。带宽预算 **400 Gbps / 8 卡**（跨 DC）可作为方案设计输入。

**G. 缺口即机会：三块未开源组件是自研切入点**
KV 存储系统、E/P/D 分离编排、四类 LB（含自动扩缩容与降级重启）。官方只给了设计文档与实测数字，**实现留白**。可参考：DualPath（SIGCOMM'26，decode 组内调度）、NanoFlow（OSDI'25，异步调度）、SGLang（前缀缓存/语法约束）。
→ 若团队要交付"一体化推理方案"，这三块是能形成差异化的地方，而不是再去卷 kernel。

**H. 直接复用它的评测 SOP（避免"数字不可比"的坑）**
- 上下文固定 **131,072**，prefill 批 **4096→65536** 扫 5 点，decode **8→256 req/rank** 扫 6 点，报 **tok/s/GPU × per-user tok/s** 的 Pareto；
- 投机解码必须注明 **nextn 长度与接受率**（官方用 nextn=3、接受率 0.75 归一化），否则吞吐数字横向不可比；
- 精度冒烟三件套：AIME 2026 **99.9**、GPQA-Diamond **93.4**、HLE-Full **42.6**（README 自述 "indicative rather than exact"，只作配置正确性检查，不作榜单）。

**I. 把 deepseek-recipe 当作交付侧的"低垂果实"**
客户从官方 API 迁到私有化部署时，最常见的阻塞不是性能而是**行为不一致**：prompt 模板差一个换行、special token 处理不同、工具调用 JSON 转义规则不一致 → 评测分数掉点 → 归因到"国产卡不行"。
→ 动作：把 recipe 纳入交付 SOP 的**基线自检工具**（同一批 prompt 分别走官方 API 与私有化部署，比对 token 序列与流式事件序列），先排除协议层变量，再谈性能与精度。


---

## 9. 参考资料

1. DeepSeek 官方知乎长文《DeepSeek V4.1 Flash 大规模推理部署实践》：https://zhuanlan.zhihu.com/p/2095560295930508953
2. 官方部署问答：https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md
3. DeepSeek推理框架全开源，公布日均吞吐和毛利，真正的OpenAI来了：https://mp.weixin.qq.com/s/7p-Wq_qD_0lU1ndjR97X4g
4. DeepInfer 仓库（Apache-2.0）：https://github.com/deepseek-ai/DeepInfer
5. deepseek-recipe 仓库（MIT）：https://github.com/deepseek-ai/deepseek-recipe
6. DeepSeek-V4.1-Flash 技术报告：arXiv:2609.19969（2026-09-17），https://arxiv.org/abs/2609.19969
7. 仓库示例与基准脚本：`examples/README.md`、`examples/benchmark.py`、`examples/sweep/run_sweep.sh`
8. 性能曲线原始图：`doc/images/prefill_pareto.png`、`decode_pareto.png`、`wide_ep_decode_pareto.png`
9. DualPath: Accelerating Agentic LLM Inference by Harvesting Disaggregated KV-Cache Storage I/O, SIGCOMM '26
10. NanoFlow: Towards Optimal Large Language Model Serving Throughput, OSDI '25
11. SGLang: Efficient Execution of Structured Language Model Programs, NeurIPS '24
12. DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation, arXiv:2607.05147
