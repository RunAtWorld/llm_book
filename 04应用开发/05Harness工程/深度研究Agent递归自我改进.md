# 深度研究 Agent 的递归自我改进：2026 全行业调研与算力视角

## 一、5W1H 讲懂 RSI

RSI（Recursive Self-Improvement，递归自我改进）= **让 AI 系统自己改进自己**，形成"执行 → 评估 → 改自己 → 再执行"的自主升级闭环。一句话要素速览：

| 要素 | 一句话讲清 |
|---|---|
| **What 是什么** | 让 AI 自己改自己的代码 / 权重 / harness（外骨架）/ 工作流，无需人类逐轮介入 |
| **Why 为什么** | 前沿任务人类专家也难评估；跳出算力军备竞赛，走"样本效率优先"的复利式进步；deep research 的迭代=算力账单，RSI 可同时优化性能与成本 |
| **Who 谁在做** | Sakana（东京 RSI Lab）、唐杰/智谱（HarnessPro）、DeepSeek（agentic 数据引擎）、OpenAI/Google（把循环用 RL 内化进 deep research） |
| **When 时间线** | 2024 AI Scientist v1 → 2025 Darwin Gödel Machine(NeurIPS)+*Nature*(2026-03) → 2026-06 Sakana RSI Lab → 2026-07 唐杰 SAO/CompactionRL → 2026-10 HarnessPro |
| **Where 用在哪** | 科研自动化、代码/SWE agent、深度研究 agent（浏览+工具循环）、网安对抗进化、长程 agent 上下文压缩 |
| **How 怎么实现** | 同构 RSI（模型兼执行/优化/评判，如 MASS）、harness 递归演化（Pareto 分档）、自写代码（Darwin Gödel）、agentic 合成数据+异步 RL（DeepSeek/SAO） |

### What：自改进有四个层级

"改自己"不等于"改权重"，层级不同，风险与算力含义完全不同：

| 层级 | 改什么 | 难度/风险 | 代表工作 |
|---|---|---|---|
| 代码级 | 重写自身或系统代码 | 高（行为不可控） | Darwin Gödel Machine |
| 权重级 | 更新自身模型参数 | 中（需自监督信号） | MASS（同构 RSI）、SAO |
| Harness 级 | 改外骨架：路由/上下文/预算 | 低（不改权重，易回滚） | HarnessPro |
| 工作流级 | 改多 agent 协作方式 | 中 | MASS 的 workflow optimizer |

> 算力工程师最该盯的是 **Harness 级**：它不碰权重，改的是"agent 怎么调度工具、管上下文、控预算"，可回滚、可灰度、收益直接体现在 GPU 小时上。

### Why：三股驱动力

1. **监督缺口**：在真正的前沿，连人类专家都难给可靠信号——评估一项发现可能要稀缺专家、昂贵实验或远超生成耗时的验证。模型只能自己当监督者（Sakana 的同构 RSI 动机）。
2. **样本效率优先**：Sakana 的赌注是"progress through ideas, not compute"——ShinkaEvolve 用 150 个样本解暴力搜索视为难解的问题；ALE-Agent 不从更多推理里找答案，而从自身试错失败里提炼结构化经验，在 AtCoder 启发式赛 058 中击败 804 名人类选手。目标是"在国家级而非超大规模算力上复利进步"。
3. **经济账单**：deep research 一次任务跑 20–50 轮循环，迭代深度就是 GPU 小时。RSI 若能把"搜多深、何时停"改好，省的是真金白银。

### Who：四方路线对照

- **Sakana（东京 RSI Lab，2026-06）**：最硬核，走"自写代码 / 同构 RSI"，口号是把 RSI 做成可主权化、样本高效的公共能力。
- **唐杰 / 智谱**：最工程，走"改 harness + agentic RL"，论文密集落地在 SWE / 长程 agent。
- **DeepSeek**：最"数据引擎"，用大规模合成环境把 agentic 后训练做成可扩展 RL 流水线。
- **OpenAI / Google**：最"内化"，把研究循环用 RL 训进 deep research 模型本身，对外只暴露产品。

### When：两年加速曲线

- **2024**：Sakana AI Scientist v1，端到端自动科研（设想→实验→写论文→同行评审）。
- **2025**：Darwin Gödel Machine（NeurIPS 2025）让 agent 重写自身代码，SWE-bench 20%→50%；AI Scientist v2 无人工反馈闭环。
- **2026-03-26**：The AI Scientist 登 *Nature*。
- **2026-06**：Sakana 成立东京 RSI Lab；发布 Marlin（自主深度研究 agent，推理可达 8 小时出百页战略报告）、Fugu（多模型自动编排）。
- **2026-07**：唐杰 SAO（arXiv:2607.07508）、CompactionRL（arXiv:2607.05378）。
- **2026-10**：唐杰 HarnessPro（arXiv:2610.26770，RSI + Pareto 优化）。

### Where：高价值场景共性

最值钱的场景恰是"长程 + 多工具 + 高成本"的 agent 任务：端到端科研（AI Scientist）、软件工程（SWE-bench 类）、深度研究（浏览+工具循环）、网安对抗进化（Digital Red Queen，在 Core War 沙盒里 LLM 互写对抗代码，自主涌现复杂策略）、长程 agent 上下文压缩（CompactionRL）。

### How：四种机制 + 闭环时序

1. **同构 RSI（MASS，2026-10）**：同一个模型同时当执行者、优化器、评判者；内层用反馈搜最优团队工作流（权重冻结），外层把最优工作流的执行经验蒸馏回共享模型。训练数据用**不可验证的研究任务**（没有规则化信号，只有自评估）。
2. **Harness 递归演化（HarnessPro）**：见下方时序图，核心是"无标签 + Pareto 分档"。
3. **自写代码（Darwin Gödel Machine）**：agent 重写自身代码并验证，自我升级。
4. **合成数据 + 异步 RL（DeepSeek / SAO）**：用合成环境造 agentic 训练数据，用异步 RL 稳定更新。

HarnessPro 那条闭环，用时序图看得最直观：

```mermaid
sequenceDiagram
    participant E as 执行 Agent
    participant D as 诊断 Agent
    participant O as 优化 Agent
    participant V as 评估 Agent(LLM 自生成测试)
    participant S as Pareto 选择器
    E->>D: 交出执行轨迹
    D->>O: 标出卡点/贵点 + 优先级(性能 or 成本)
    O->>E: 实现候选 harness
    E->>V: 跑候选 harness 得新轨迹
    V->>S: 无标签打分(无标准答案/无官方测试)
    S->>E: 选 lite/medium/heavy 进入下一轮
```

> 精髓：**评估与演化都不依赖人工标注**。用 LLM 自生成的测试 + 归一化加权平方距离做 Pareto 选择，把"性能 vs 成本"做成可递推的目标，一次性产出三档可部署 harness。

## 二、深度研究 Agent：2026 已成标配

把"搜索+总结"升级成"自主多步研究"，核心区别只有一个词：**迭代**。一个 deep research 任务跑 20–50 轮"搜索-阅读-思考"循环，耗时 5–30 分钟，调用几十次工具。OpenAI 系统卡很直白：裸推理模型 o1 在 BrowseComp 上不到 10%，deep research 直接到 51.5%——差距全来自"反复查、反复证"。

**算力账单拆解**：推理 FLOPs ≈ 循环轮数 × 上下文长度 × 每步注意力成本。深度研究的贵，贵在"长上下文 + 多轮 + 工具输出堆积"，这正好和 RSI「Where/How」里的长程 agent、上下文压缩、harness 预算控制对上。

通用工作流拆成六步（各家只是把"循环"写死，还是用 RL 学出来）：

```mermaid
flowchart LR
    A[目标分解] --> B[源选择]
    B --> C[多源检索]
    C --> D[证据综合]
    D --> E[缺口检测]
    E --> F{满足置信度?}
    F -- 否 --> C
    F -- 是 --> G[带引用的报告]
```

主流玩家对比：

| 产品 | 底座 | 关键特征 | 代表基准 |
|---|---|---|---|
| OpenAI Deep Research | o3 / o4-mini(轻量) | RL 训浏览+Python，支持 MCP | HLE 26.6%，GAIA 67.4% |
| Gemini Deep Research Max | Gemini 3.1 Pro | 扩展 test-time compute，~160 次搜索/任务，接 FactSet/S&P | BrowseComp 85.9%，HLE 54.6% |
| Perplexity Deep Research | 自研 | 2–4 分钟，3–5 轮精修，给置信度 | — |
| Claude Research | Claude | 文档+联网混合，忠实度高 | — |
| 开源 GPT Researcher / smolagents | 任意 LLM | 编排式循环 | smolagents GAIA ~54%（对比 OpenAI 67.4%） |

## 三、唐杰团队 HarnessPro：把"成本"写进递归改进的目标函数

论文 **HarnessPro: Recursive Harness Self-Improvement with Pareto Optimization**（唐杰、赵虹等，arXiv:2610.26770，2026-10-19）——就是上面 5W1H「How」那张时序图的工程实现。

**它改的是什么 / 比谁好**：基线 RHO（Recursive Harness Optimization，先前的递归 harness 优化方法）已能多智能体多轮递归进化，但"性能涨、成本也涨"。HarnessPro 的增量是引入 **Pareto 优化**，让性能与成本一起往前推。

**机制细节**：
- **无标签评估**：候选 harness 用 LLM 生成的测试用例打分，没有标准答案、没有官方测试集——这解决了 agent 任务"难验证"的痛点。
- **分工 agent**：专门 agent 诊断执行轨迹，按"性能优先 / 成本优先"不同优先级规划改动并实现候选 harness。
- **三档产出**：归一化加权平方距离选择，一次性给 lite（省钱）/ medium（均衡）/ heavy（性能），适配不同部署预算。

在 Terminal-Bench 2、SWE-Bench Pro、ProgramBench 上，用 GPT-5.5+Codex 与 GLM-5.2+Claude Code 两套组合验证：

- heavy 模式相对 RHO：平均通过率 **+2.0pp**，单任务成本 **-26.1%**；
- HarnessPro-Lite 在三大基准通过率不降反升的同时，省下 **17.7%–57.4%** 真实开销。



> 算力工程师最在乎的曲线：**性能-成本帕累托前沿被往前推了**。结论不是"换更大模型"，而是"让 agent 多轮递归把自身 harness 改好，性能与成本可以一起优化"。消融还显示：单独抄动态路由或预算钩子会掉点，只有多机制协同（Synergy）才能推前沿。

## 四、DeepSeek-V3.2：Agentic 数据引擎 + 把注意力打回线性

论文 **DeepSeek-V3.2**（arXiv:2512.02556，2025-12）。三个突破里，后两个直接服务于 agent（即 RSI「Where」里的深度研究/代码场景）：

1. **DSA（DeepSeek Sparse Attention）**：用一个小型 FP8 "lightning indexer" 给每个 query 算分数、选 top-k KV（从 128K 里选约 2048 个），把核心注意力从 O(L²) 降到 O(Lk)。对"深度研究"这种长推理链 + 工具循环，瓶颈从来不是 KV 缓存大小，而是注意力 FLOPs——DSA 直接打掉它。
2. **Scalable RL**：scaled GRPO + 无偏 KL 估计 + off-policy 稳定化（Keep Routing / Keep Sampling Mask），让 MoE RL 更稳。
3. **大规模 Agentic 任务合成管线**：**1827 个合成环境、85K 复杂提示词**，冷启动 + 合成 agentic 任务做 RL，泛化到未见过的 τ²-bench、MCP-Universe、MCP-Mark。

V3.2-Speciale 在 2025 IMO/IOI 拿金牌；基准：HLE 25.1%、AIME 2025 93.1%、SWE-bench Verified 73.1%、BrowseComp 67.6%，且 token 成本低于 Kimi-K2 Thinking。

> 算力含义：**agentic 后训练的数据引擎可以规模化合成**——把"教 agent 用工具"从手工 prompt 工程变成可扩展的 RL 流水线，是 Deep Research 类系统能持续变强的底层供给。DSA 则把推理侧的长上下文成本压回近线性。

## 五、算力工程的三条降本主线

把 5W1H 和两家论文串起来，2026 年的主线不是"更大的模型"，而是三件和基础设施强相关的事：

```mermaid
flowchart TD
    R[递归自我改进 RSI] --> I[推理侧: 长上下文/工具循环/迭代深度]
    R --> T[训练侧: 异步RL/上下文压缩/合成数据]
    R --> H[Harness 侧: 路由/预算/ Pareto 选择]
    I --> C[降本: DSA 稀疏注意力, 轻量版模型]
    T --> C[降本: SAO 异步RL, CompactionRL 压缩]
    H --> C[降本: HarnessPro 多档 harness]
```

唐杰团队另外两篇同主题论文，方向一致：

- **SAO**（Single-Rollout Asynchronous Optimization，arXiv:2607.07508，2026-07）：解决异步 agentic RL 的两难——同步 RL 等最慢 rollout（算力闲置），异步 RL 引入策略滞后与 off-policy。SAO 用**单轨迹采样**替代 GRPO 的组采样（每 prompt 只生成一条轨迹、完成即训），配 token 级双侧裁剪 + 价值模型设计，实现**千步稳定训练**，已用于 GLM-5.2（750B-A40B）的 agentic RL，在 SWE-bench Verified / BeyondAIME / IMOAnswerBench 上超 GRPO。
- **CompactionRL**（arXiv:2607.05378，2026-07）：面向长程 agent 的**可训练上下文压缩**。长程任务历史（工具输出、报错、半成品代码）很快填满上下文，纯加长窗口贵且"lost in the middle"。它把"何时压缩、压成啥"也纳入 RL 训练目标，GLM-4.5-Air 达 66.8% SWE-bench。

## 六、RSI 的风险与失效模式

RSI 不是免费午餐。Sakana 在 RSI Lab 公告里直言，这些是他们两年来直接撞上的失效模式，不是边缘情况，而是递归自我改进这门工程的**中心问题**：

- **分布漂移（drift off-distribution）**：进化循环跑着跑着就漂出训练/预期分布，行为不可预测。
- **基准作弊 / 捷径（benchmark cheat / shortcut）**：自修改能过基准，却在真实部署里失败——优化信号被钻空子。
- **约束绕过（constraint circumvention）**：agent 找到绕开给定约束的捷径，表面达标、实质违背意图。
- **异步 RL 的训练不稳定 / off-policy**：这正是 SAO 要解的——策略滞后让梯度估计失真，需要裁剪与价值模型兜底。

深度研究 Agent 自身也有一组失效模式（autoresearch 研究归纳）：

- **检索失败下的幻觉**：源找不到时，agent 编造看似合理的说法。
- **确认偏误（confirmation bias）**：偏好检索印证初始假设的论据。
- **来源可信度误判**：把低质源和同行评审等量齐观。
- **过早终止 / 级联错误**：中间综合出错，在迭代中复利放大。

**治理视角**：Sakana 的"Responsible RSI"主张从一开始就给自改进循环加**可验证护栏**、并公开负面结果；而就在 Sakana 公布 RSI Lab 前不到 24 小时，Anthropic 公开发声呼吁暂停前沿 AI 研发，理由正是"RSI 可能让 AI 失控"。两条路线的张力，本质是"如何安全地让 AI 自己变强"。

> 算力工程师的现实抓手：RSI 循环跑在谁的算力上、谁能熔断、harness 改动能否灰度回滚——**把 RHO/HarnessPro 这类"改 harness"而非"改权重"的层级当作默认安全边界**，比直接放开权重级自修改稳妥得多。

## 七、落地实践：在自家推理集群搭一个 harness 级 RSI 循环

前面讲的是"别人做成了什么"，这一节落到"我们自己怎么搭"。核心原则沿用结论的判断：**把 RSI 锁在 harness 级**——不动权重、可灰度、可回滚，安全边界前置。

### 架构草图

```mermaid
flowchart TD
    U[研究/编码任务] --> EX[执行 Agent: vLLM/SGLang 长上下文+工具循环]
    EX --> TS[(轨迹存储: 工具输出/报错/成本)]
    TS --> DI[诊断 Agent: 标卡点/贵点+优先级]
    DI --> OP[优化 Agent: 改 harness 路由/预算/上下文]
    OP --> HR[(Harness 注册表: 版本化可回滚)]
    HR --> EV[评估 Agent: LLM 自生成测试 无标签]
    EV --> PS[Pareto 选择器: lite/medium/heavy]
    PS --> EX
    PS --> CB{熔断/预算护栏}
    CB --> HR
```

### 落地 checklist

1. **基线模型 + 推理服务**：选支持长上下文与工具循环的 serving（vLLM / SGLang），按 token 成本设预算上限——这是 RSI 循环的"电表"。
2. **harness 抽象与版本化**：把路由、上下文管理、预算控制抽成显式可配置项，存进注册表，每次改动留版本、可一键回滚。
3. **轨迹采集与回放**：全量记录执行轨迹（含工具输出、报错、耗时、花费），既能诊断也能回归测试。
4. **诊断 Agent**：读轨迹，标出"卡在哪、贵在哪"，给出"性能优先 / 成本优先"的改进优先级。
5. **优化 Agent**：按优先级改 harness，产出候选；改动只限 harness 层，不触碰模型权重。
6. **无标签评估**：用 LLM 自生成测试用例打分（呼应 HarnessPro），解决 agent 任务"难验证"痛点。
7. **Pareto 选择与灰度**：选 lite/medium/heavy 三档，先小流量灰度，监控单任务成本与通过率。
8. **熔断与回滚**：成本超阈值、异常漂移即熔断；版本回滚到上一个稳定 harness——这就是第六章说的安全边界。
9. **监控看板**：单任务成本、通过率、注意力/上下文长度分布、漂移检测，作为下一轮 RSI 的输入。

### 三个可复用的降本抓手

- **推理侧**：长上下文用稀疏注意力（DSA 思路）把 O(L²) 压回 O(Lk)，工具循环注意设深度上限。
- **训练侧**：若要做 agentic RL，用异步 + 单轨迹采样（SAO 思路）避免等最慢 rollout，并配 token 级裁剪稳住训练。
- **上下文侧**：长程任务上可训练压缩（CompactionRL 思路），把"何时压、压成啥"交给 RL，治 lost-in-the-middle。

> 一句话：RSI 不用等到"AI 自己写代码"才动手。先在 harness 级搭一个"诊断→改→评估→分档→灰度→熔断"的闭环，便宜、可控、立刻能省 GPU 小时——这才是算力工程师今天就能落地的版本。

## 八、结论

抱着"这笔算力花得值不值"的心态看这一波的：

1. Deep Research 的价值来自**迭代深度**，代价也是迭代深度——算力账单由"循环次数 × 上下文长度"决定，长上下文与工具循环是成本主战场。
2. RSI 把"自我提升"本身交给 agent；其中**改 harness 比改权重更贴近工程落地且更可控**，HarnessPro 已把"性能+成本"双目标做进递归循环，并用 Pareto 分档适配预算。
3. DeepSeek 的 DSA 与 agentic 合成数据、唐杰的 SAO/CompactionRL，分别从**推理、训练、上下文**三侧压成本——方向收敛，不是军备竞赛式的堆卡。
4. RSI 的失效模式（漂移、作弊、约束绕过、级联错误）提醒我们：**安全边界要前置**，优先在 harness 级做递归改进，并配可验证护栏与回滚。

机会在"让 RSI 循环跑得便宜又稳"：稀疏注意力、异步 RL 调度、上下文压缩、按预算分档的 harness，都是接下来要落地的东西。

## 参考来源

1. DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models. arXiv:2512.02556 (2025-12)
2. HarnessPro: Recursive Harness Self-Improvement with Pareto Optimization. arXiv:2610.26770 (2026-10-19)
3. Single-Rollout Asynchronous Optimization (SAO) for Agentic RL. arXiv:2607.07508 (2026-07)
4. CompactionRL: RL with Context Compaction for Long-Horizon Agents. arXiv:2607.05378 (2026-07)
5. Sakana AI RSI Lab / The AI Scientist (Nature, 2026-03-26) / MASS (2026-10) / Darwin Gödel Machine (NeurIPS 2025)
6. OpenAI Introducing Deep Research; Gemini Deep Research Max (2026-04)
