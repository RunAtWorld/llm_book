# DeepSeek V3 18大核心亮点

> 报告：https://arxiv.org/abs/2412.19437

### 模型架构方面

1. 使用极大规模混合专家模型（MoE）

   - 总参数量达**6710亿**，每个token激活**370亿**参数。 

   - 每个MoE层采用**1个共享专家和**256个路s由专家**，每个专家的中间隐藏维度为2048**。


2. 使用多头潜在注意力（MLA）

   - 通过低秩联合压缩，减少键值（KV）缓存需求，提升推理效率。 
   
   
      - 注意力头数设置为**128**，每个头的维度为**128**，KV压缩维度为**512**。 
   


3. 使用**无辅助损失的负载均衡策略**

   - 创新性地避免传统负载均衡方法对模型性能的负面影响。 
   
   
      - 通过灵活的批量负载均衡，允许专家在不同领域中更好地专业化。 
   


4. 使用**多token预测（MTP）训练目标**

   - 同时预测**2个**未来token，增加训练信号密度，可能提高数据效率。 
   
   
      - 第二个token预测的接受率在**85%到95%**之间，显著加快解码速度。 
   
   
      - 使用**1层MTP模块**，顺序预测额外token，并在每个预测深度保持完整的因果链。 
   


### 高效训练方面

5. 使用**FP8混合精度加速训练**

   - 支持FP8计算和存储，加速训练并减少 GPU 内存使用。 
   
   
      - 大多数 GEMM 操作（如 Fprop、Dgrad、Wgrad）在 FP8 下执行，计算速度比 BF16 提升**2倍**。 
   
   
      - 保留高精度操作（如嵌入模块、MoE 门控模块）以确保数值稳定性。 
   


6. 使用**DualPipe算法提升训练效率**

   - 通过计算-通信重叠，减少管道气泡，提升训练效率。 
   
   
      - 将每个块划分为**注意力机制、全对全分发、MLP 和全对全组合**四个组件，并手动调整 GPU 流式多处理器（SMs）的比例。 
   
   
      - 采用**双向管道调度**，从管道两端同时输入微批次，隐藏大部分通信开销。 
   


7. 进行了**极致的内存优化**

   - 通过重新计算RMSNorm和MLA上投影，减少内存占用。 
   
   
      - 将指数加权平均（EMA）参数存储在 CPU 内存中，异步更新以减少 GPU 内存压力。 
   
   
      - 多token预测（MTP）模块与主模型共享嵌入层和输出头，进一步提高内存效率。
   


8. **训练稳定性极高**

   - 整个训练过程无不可恢复的损失峰值，未进行过回滚。 

   - 训练成功率100%，展现了极高的稳定性。 

9. **成本训练极低**  

   - 完整训练仅需**278.8万**H800 GPU小时**，展现高效成本效益。**训练成本仅为557万美元，****远低于国内外其他已知模型**。 

### 数据处理与预训练

10. **高质量多样化数据**

   - 在**14.8万亿** token 上进行预训练，涵盖多语言、数学、编程等领域。 

   - 增强数学和编程样本的比例，扩展多语言覆盖范围（不仅限于英语和中文）。 

11. **文档打包与FIM策略**

   - 通过文档打包保持数据完整性，避免跨样本注意力掩码。 

   - 引入**Fill-in-Middle（FIM）策略**，使用率为**10%**，结构化数据如下：

  `<|fim_begin|> pre <|fim_hole|> suf <|fim_end|> middle <|eos_token|>`。 

12. **多语言分词器优化**

   - 使用字节级BPE，词汇量扩展到**128K**token**。 

   -  引入结合标点符号和换行符的token，优化多语言压缩效率。 

13. **长上下文扩展技术**

   - 通过两阶段训练，将上下文长度从**4K**扩展到**128K**。 

   - 采用YaRN技术，配置为`scale = 40, base = 1, factor = 32`，确保扩展稳定性。 

**后训练与性能提升**

14. **监督微调（SFT）**

   -  使用**150万**个指令微调实例，涵盖推理、数学、编程等多个领域。 

   - 通过内部DeepSeek-R1模型生成推理数据，平衡准确性和格式清晰性。   

15. **强化学习（RL）**

   - 使用基于规则和基于模型的奖励模型，优化复杂推理任务表现。 

   - 采用分组相对策略优化（**GRPO**），从组分数中估计基线，提升模型性能。 

16. **知识蒸馏**

   - 从DeepSeek-R1系列模型中蒸馏推理能力，显著提升数学和编程任务表现。 

   - 在LiveCodeBench和MATH-500基准测试中，性能提升显著。 

**性能表现**

1.  在**多领域评测性能领先**

   - 在MMLU 基准测试中准确率达**85.6%**，在GSM8K数学任务中准确率达**92.3%**。

   -  在HumanEval代码生成任务中，通过率提升**15%**。 

2.  **效果**与最好的闭源模型相当**

    - 在LongBench v2长上下文基准测试中，F1分数达**91.6**，与GPT-4o 相当。 

    - 在FRAMES 基准测试中，处理**100K** token 上下文的能力显著优于其他模型。 

------


# 《DeepSeek V3技术报告》全文精读

**摘要**

DeepSeek V3（DeepSeek-V3）是一款强大的混合专家（Mixture-of-Experts, MoE）语言模型，总参数量为6710亿，每个token激活370亿参数。

为了实现高效的推理和成本效益高的训练，DeepSeek-V3采用了多头潜在注意力（Multi-head Latent Attention, MLA）和深度探索MoE架构，这些架构已经在深度探索V2中得到了充分验证。

此外，DeepSeek-V3**开创了无辅助损失的负载均衡策略**，并设定了多token预测训练目标以增强性能。

DeepSeek-V3**在14.8万亿高质量且多样化的token上进行了预训练**，随后经过监督微调和强化学习阶段，充分发挥了其能力。

全面的评估显示，DeepSeek-V3在性能上超越了其他开源模型，**并达到了与领先闭源模型相当的水平**。

尽管性能卓越，DeepSeek-V3的**完整训练仅需278.8万H800 GPU小时**。

**此外，其训练过程非常稳定，整个训练过程中没有出现任何不可恢复的损失峰值，也未进行过回滚。**模型检查点可在https://github.com/deepseek-ai/DeepSeek-V3获取。     

**2. 架构**

首先介绍DeepSeekV3的基本架构，该架构采用了多头潜在注意力（MLA）（DeepSeek-AI, 2024c）以实现高效推理，以及DeepSeekMoE（Dai et al., 2024）以实现成本效益高的训练。

然后，介绍多token预测（Multi-Token Prediction, MTP）训练目标，观察到这可以增强在评估基准上的整体性能。对于未明确提及的其他次要细节，DeepSeekV3遵循DeepSeekV2（DeepSeek-AI, 2024c）的设置。 

**2.1. 基本架构**

DeepSeekV3的基本架构仍然基于Transformer（Vaswani et al., 2017）框架。

为了实现高效推理和成本效益高的训练，DeepSeekV3也采用了MLA和DeepSeekMoE，这些已经在DeepSeekV2中得到了充分验证。

与DeepSeekV2相比，**唯一的例外是引入了无辅助损失的负载均衡策略**。 

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/iaJfIYygOIiaeTTs1XNibdt1P9uUKapGkfkNOar36SQaTcjkBHVibpUNtOnLeJgda9nicfVs2V4KSPBEht78k5xibGGw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



**2.1.1. 多头潜在注意力**

对于注意力机制，DeepSeekV3采用了MLA架构。设d为嵌入维度，h为注意力头的数量，d_h为每个头的维度，h_t为给定注意力层第t个token的注意力输入。MLA的核心是通过低秩联合压缩注意力键和值来减少推理期间的键值（KV）缓存。

**2.1.2.** **DeepSeek**MoE与无辅助损失的负载均衡**

DeepSeekMoE的基本架构。对于前馈网络（FFNs），DeepSeekV3采用了DeepSeekMoE架构（Dai et al., 2024）。与传统的MoE架构（如GShard，Lepikhin et al., 2021）相比，DeepSeekMoE使用更细粒度的专家，并隔离一些专家作为共享专家。

**2.2. 多token预测**

受Gloeckle等人（2024）的启发，研究并为DeepSeekV3设定了多token预测（Muli-Token Prediction, MTP）目标，该目标将预测范围扩展到每个位置的多个未来token。

一方面，**MTP目标增加了训练信号的密度，可能提高了数据效率**。另一方面，**MTP可能使模型能够预先规划其表示，从而更好地预测未来token**。

图3展示了MTP的实现。与Gloeckle等人（2024）并行预测额外token使用独立输出头不同，这里顺序预测额外token，并在每个预测深度保持完整的因果链。本节将详细介绍MTP的实现细节。 

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/iaJfIYygOIiaeTTs1XNibdt1P9uUKapGkfkZ2ZFVbs34QRlsbNicBs48wbGrsMr2teOzvSKsy7h0f6sOnJTbUeusRg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



MTP模块：具体来说，MTP的实现使用了n个顺序模块来预测k个额外token。第n个MTP模块包括一个共享嵌入层Emb(·)，一个共享输出头OutHead(·)，一个Transformer块TRM(·)，以及一个投影矩阵W ∈ R×2k。

**3. 基础设施**

**3.1. 计算集群**

DeepSeekV3在一个配备2048块NVIDIA H800 GPU的集群上进行训练。H800集群中的每个节点包含8块GPU，通过NVLink和NVSwitch在节点内连接。在不同节点之间，使用InfiniBand（IB）互连来促进通信。 

**3.2. 训练框架**

DeepSeekV3的训练由HAI-LLM框架支持，这是一个由工程师从头开始构建的高效且轻量级的训练框架。总体而言，DeepSeekV3应用了16路管道并行性（PP）（Qi et al., 2023a），64路专家并行性（EP）（Lepikhin et al., 2021）跨越8个节点，以及ZeRO-1数据并行性（DP）（Rajbhandari et al., 2020）。 

为了促进DeepSeekV3的高效训练，实施了细致的工程优化。

首先，设计了**DualPipe算法**以实现高效的管道并行性。与现有的PP方法相比，DualPipe减少了管道气泡。

更重要的是，它**在前向和后向过程中重叠了计算和通信阶段**，从而解决了由跨节点专家并行性引入的通信开销挑战。

其次，**开发了高效的跨节点全对全通信内核**，充分利用了IB和NVLink带宽，并节省了用于通信的流式多处理器（SMs）。

最后，**在训练过程中仔细优化了内存占用**，使得无需使用昂贵的张量并行性（TP）即可训练DeepSeekV3。 

**3.2.1. DualPipe 和 计算-通信重叠**

在 DeepSeek-V3 中，跨节点专家并行性引入的通信开销导致计算与通信的比例约为 1:1，这种比例效率低下。

为了解决这一挑战，**设计了一种创新的管道并行算法 DualPipe**。DualPipe 不仅通过有效重叠前向和后向计算与通信阶段加速模型训练，还减少了管道中的空闲时间（pipeline bubbles）。   

**DualPipe 的核心思想是在每一对前向和后向块内重叠计算和通信。**

**具体来说，每个块被划分为四个组件：注意力机制、全对全分发、多层感知机（MLP）和全对全组合。**

特别地，对于后向块，注意力机制和 MLP 进一步分为两部分：输入的后向和权重的后向，类似于 ZeroBubble（Qi 等，2023b）。

此外，还引入了一个 PP 通信组件。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/iaJfIYygOIiaeTTs1XNibdt1P9uUKapGkfkNE7w3j99Xj833gnXKIlrKxwQ58u3CCOe8nkjfx5y14dE4dygD1A23Q/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

如图 4 所示，对于一对前向和后向块，重新排列这些组件，并手动调整专门用于通信和计算的 GPU 流式多处理器（SMs）的比例。通过这种重叠策略，可以确保全对全通信和 PP 通信在执行过程中完全隐藏。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/iaJfIYygOIiaeTTs1XNibdt1P9uUKapGkfkWIPXCSmuicgXew7xsYHmuK3C7dlVdghGooOvSCnN2pWnbPjuQFicH0Zg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

图 5 展示了**完整的DualPipe调度策略，采用双向管道调度，从管道两端同时输入微批次，从而使得大部分通信可以完全重叠。这种重叠策略确保了随着模型规模的扩大，只要保持计算与通信的比例不变，仍然可以在节点间使用细粒度专家，同时实现接近零的全对全通信开销**。 

**3.2.2. 高效实现跨节点全对全通信**

为了确保 DualPipe 的计算性能，定制了高效的跨节点全对全通信内核（包括分发和组合），以节省用于通信的 SMs 数量。内核的实现与 MoE 门控算法以及集群的网络拓扑共同设计。

具体来说，在集群中，跨节点的 GPU 通过 IB（InfiniBand）完全互连，而节点内的通信则通过 NVLink 进行。NVLink 提供的带宽为 160 GB/s，大约是IB（50 GB/s）的3.2倍。

为了有效利用 IB 和 NVLink 的不同带宽，限制每个 token 最多分发到4个节点，从而减少 IB 的流量。当 token 的路由决策完成后，首先通过IB传输到目标节点上具有相同节点索引的 GPU。

一旦到达目标节点，会立即通过 NVLink 转发到承载目标专家的特定 GPU，避免被后续到达的 token 阻塞。

这样，IB 和 NVLink 的通信可以完全重叠，每个 token 可以高效地选择每个节点平均 3.2 个专家，而不会因为 NVLink 引入额外开销。

**3.2.3. 极致内存节省且开销极小**

为了减少训练过程中的内存占用，采用了以下技术：   

**重新计算 RMSNorm 和 MLA 上投影**。在反向传播过程中重新计算所有 RMSNorm 操作和 MLA 上投影，从而无需持久存储其输出激活。虽然有轻微的开销，但该策略显著减少了存储激活所需的内存。 

**CPU 中的指数加权平均（Exponential Moving Average, EMA）**：在训练过程中，保留模型参数的 EMA 以便在学习率衰减后早期估计模型性能。EMA 参数存储在 CPU 内存中，并在每次训练步骤后异步更新。这种方法可以在不增加额外内存或时间开销的情况下维护 EMA 参数。 

**多Token预测（Multi-Token Prediction, MTP）模块与主模型共享嵌入层和输出头**：通过 DualPipe 策略，将模型的最浅层（包括嵌入层）和最深层（包括输出头）部署在相同的 PP 排序上。

这种安排使得 MTP 模块和主模型之间可以物理共享嵌入层和输出头的参数和梯度。这种物理共享机制进一步提高了内存效率。 

**3.3. FP8 训练**

受最近低精度训练进展（Dettmers 等，2022；Noune 等，2022；Peng 等，2023b）的启发，提出了一种**利用FP8数据格式进行DeepSeek-V3训练的细粒度混合精度框架**。

尽管低精度训练前景广阔，但常受限于激活、权重和梯度中的异常值（Fishman 等，2024；He 等；Sun 等，2024）。

虽然在推理量化方面取得了显著进展（Frantar 等，2022；Xiao 等，2023），但在大规模语言模型中成功应用低精度技术的研究相对较少。 

**3.3.1. 混合精度框架**

基于广泛采用的低精度训练技术（Kalamkar 等，2019；Narang 等，2017），提出了一种用于 FP8 训练的混合精度框架。

在该框架中，大多数计算密集型操作在 FP8 精度下进行，而一些关键操作则战略性地保持在原始数据格式以平衡训练效率和数值稳定性。整体框架如图 6 所示。   

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/iaJfIYygOIiaeTTs1XNibdt1P9uUKapGkfkO3FbPLcC2zXd5ibQwicUIYIwxEbWPcnaM9kOkxGXBdNX2s5MDwTpOtmQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

首先，为了加速模型训练，大多数核心计算内核，即 GEMM（General Matrix Multiply）操作，以 FP8 精度实现。这些 GEMM 操作接受 FP8 张量作为输入，并输出 BF16 或 FP32 格式的张量。

如图 6 所示，与线性操作符相关的三个 GEMM，即 Fprop（前向传播）、Dgrad（激活反向传播）和 Wgrad（权重反向传播），均在 FP8 下执行。这种设计理论上将计算速度比原 BF16 方法提高一倍。此外，FP8 Wgrad GEMM 允许激活在反向传播中以 FP8 格式存储，这显著减少了内存消耗。 

尽管 FP8 格式的效率优势明显，但某些操作由于对低精度计算的敏感性仍需保持较高精度。此外，一些低开销的操作也可以利用较高精度而对整体训练成本影响极小。

**因此，经过仔细研究，保持以下组件的原始精度（例如 BF16 或 FP32）：嵌入模块、输出头、MoE 门控模块、归一化操作符和注意力操作符。**

这些高精度的保留确保了 DeepSeek-V3 的稳定训练动态。为了进一步保证数值稳定性，将主权重、权重梯度和优化器状态存储在较高精度中。 

**3.3.2. 通过量化和乘法改进精度**

基于混合精度 FP8 框架，引入了几种策略以提高低精度训练的准确性，重点在于量化方法和乘法过程。 

细粒度量化：在低精度训练框架中，由于 FP8 格式的动态范围有限（受限于其减少的指数位），溢出和下溢是常见挑战。通常的做法是通过将输入张量的最大绝对值缩放到 FP8 的最大可表示值来对齐输入分布到 FP8 的可表示范围内（Narang 等，2017）。

这种方法使低精度训练高度敏感于激活异常值，可能导致量化准确性大幅下降。为了解决这一问题，提出了一种细粒度量化方法，在更细的粒度上进行缩放。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/iaJfIYygOIiaeTTs1XNibdt1P9uUKapGkfktjZ85HzMnxachKl7ib7DHekpwsreD29IS2JQ9RR8uDu2276mdGGTWYw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



如图 7（a）所示，（1）对于激活，以 1x128 瓷片（即每个 token 每 128 通道）为基础进行分组和缩放；（2）对于权重，以 128x128 块（即每个 128 输入通道每 128 输出通道）为基础进行分组和缩放。

这种方法确保量化过程能够更好地适应异常值，通过根据较小元素组调整缩放来实现。在附录 B.2 中，进一步讨论了当以与权重量化相同的方式对激活进行分组和缩放时的训练不稳定性。   

**方法中的一个关键修改是引入了 GEMM 操作内维度上的每组缩放因子。**

**3.3.3. 低精度存储和通信**

**结合 FP8 训练框架，进一步通过将缓存的激活和优化器状态压缩为低精度格式来减少内存占用和通信开销。** 

**低精度优化器状态**：采用 BF16 数据格式而不是 FP32 来跟踪 AdamW 优化器中的第一和第二矩，而不会引起明显的性能下降。然而，优化器存储的主权重和用于批量大小累积的梯度仍然保留为 FP32，以确保整个训练过程中的数值稳定性。 

**低精度激活：**如图 6 所示，Wgrad 操作在 FP8 下执行。为了减少内存消耗，自然选择以 FP8 格式缓存线性操作符的反向传播激活。然而，对一些操作符进行低成本高精度训练时需要特别考虑： 

（1）注意力操作符后的线性操作符输入。这些激活也被注意力操作符的反向传播使用，因此对精度敏感。采用专门的 E5M6 数据格式仅用于这些激活。此外，这些激活在反向传播时从 1x128 量化块转换为 128x1 块。为了避免引入额外的量化误差，所有缩放因子都是整数幂的 2。 

（2）MoE 中 SwiGLU 操作符的输入。为了进一步减少内存成本，缓存 SwiGLU 操作符的输入并在反向传播时重新计算其输出。这些激活也以 FP8 格式存储，采用细粒度量化方法，在内存效率和计算准确性之间取得平衡。 

**低精度通信：**通信带宽是 MoE 模型训练的关键瓶颈。

为了解决这一挑战，在 MoE 上投影前对激活进行 FP8 量化，然后应用分发组件，这与 MoE 上投影的 FP8 Fprop 兼容。

与注意力操作符后的线性操作符输入类似，该激活的缩放因子也是整数幂的 2。对 MoE 下投影前的激活梯度应用类似的策略。对于前向和后向组合组件，保留为 BF16 以保持训练管道关键部分的训练精度。   

**3.4. 推理和部署**

DeepSeek-V3 部署在 H800 集群上，其中每个节点内的 GPU 通过 NVLink 互连，集群中的所有 GPU 通过 IB 完全互连。为了同时确保在线服务的服务级别目标（Service-Level Objective, SLO）和高吞吐量，采用了一种将预填充和解码阶段分离的部署策略。 

**3.4.1. 预填充**

预填充阶段的最小部署单元由4个节点和32个GPU组成。

**注意力部分**：**采用4路张量并行（TP4）结合序列并行（SP），并结合8路数据并行（DP8）。较小的TP规模（4）限制了TP通信的开销。

**MoE部分**：使用32路专家并行（EP32），确保每个专家处理足够大的批量大小，从而提高计算效率。

对于MoE的全对全通信，采用与训练阶段相同的方法：首先通过IB在节点之间传输token，然后通过NVLink在节点内的GPU之间转发。

特别地，浅层的密集MLP使用1路张量并行，以节省TP通信。 

为了在MoE部分实现不同专家之间的负载均衡，需要确保每个GPU处理大致相同数量的token。

为此，引入了冗余专家的部署策略，复制高负载专家并进行冗余部署。高负载专家是根据在线部署期间收集的统计信息检测出来的，并定期进行调整（例如，每10分钟调整一次）。

确定冗余专家的集合后，根据观察到的负载情况，仔细在节点内的GPU之间重新排列专家，尽量在不增加跨节点全对全通信开销的情况下平衡GPU之间的负载。

对于DeepSeek-V3的预填充阶段，设置了32个冗余专家。每个GPU除了托管原来的8个专家外，还将托管一个额外的冗余专家。 

此外，在预填充阶段，为了提高吞吐量并隐藏全对全和TP通信的开销，同时处理两个具有相似计算工作量的微批次，重叠一个微批次的注意力和MoE操作与另一个微批次的分发和组合操作。   

最后，正在探索动态冗余策略，其中每个GPU托管更多专家（例如，16个专家），但在每次推理步骤中只有9个会被激活。在每层的全对全操作开始之前，实时计算全局最优路由方案。鉴于预填充阶段涉及大量计算，计算路由方案的开销几乎可以忽略不计。 

**3.4.2. 解码**

在解码阶段，将共享专家视为路由专家。从这个角度来看，每个token在路由时会选择9个专家，其中共享专家被视为一个高负载专家，始终会被选中。

解码阶段的最小部署单元由40个节点和320个GPU组成。注意力部分采用TP4结合SP，并结合DP80，而MoE部分使用EP320。

对于MoE部分，每个GPU只托管一个专家，64个GPU负责托管冗余专家和共享专家。分发和组合部分的全对全通信通过IB进行直接点对点传输，以实现低延迟。此外，利用IBGDA（NVIDIA, 2022）技术进一步减少延迟并提高通信效率。 

与预填充类似，根据在线服务的统计专家负载，定期确定一定间隔内的冗余专家集合。但由于每个GPU只托管一个专家，因此不需要重新排列专家。

**3.5. 对硬件设计的建议**

基于对全对全通信和FP8训练方案的实现，向AI硬件供应商提出以下芯片设计建议。 

**3.5.1. 通信硬件**

在DeepSeek-V3中，实现了计算与通信的重叠，以隐藏计算过程中的通信延迟。这显著减少了与串行计算和通信相比对通信带宽的依赖。然而，**当前的通信实现依赖于昂贵的SM（例如，在H800 GPU中，为这一目的分配了132个SM中的20个），这将限制计算吞吐量**。**此外，使用SM进行通信会导致显著的效率低下，因为张量核心完全未被利用。**   

目前，SM主要执行以下任务以进行全对全通信： 

**希望未来的供应商能够开发硬件，将这些通信任务从宝贵的计算单元SM卸载，作为GPU协处理器或网络协处理器，类似于NVIDIA SHARP（Graham et al., 2016）。**

此外，**为了减少应用程序编程复杂性，希望这种硬件能够从计算单元的角度统一IB（扩展）和NVLink（向上扩展）网络。**通过这种统一的接口，计算单元可以通过提交基于简单原语的通信请求，在整个IB-NVLink统一域内轻松完成读取、写入、多播和归约等操作。 

**3.5.2. 计算硬件**

更高的FP8 GEMM累加精度在张量核心中。在当前NVIDIA Hopper架构的张量核心实现中，FP8 GEMM（通用矩阵乘法）使用定点累加，通过根据最大指数右移对齐尾数乘积后再相加。

**4. 预训练**

**4.1. 数据构建**

与DeepSeek-V2相比，通过增强数学和编程样本的比例以及扩展多语言覆盖范围（不仅限于英语和中文），优化了预训练语料库。

此外，数据处理管道经过改进，以最小化冗余同时保持语料库多样性。受到Ding et al. (2024)的启发，实现了文档打包方法以保持数据完整性，但在训练过程中没有引入跨样本注意力掩码。

最终，**DeepSeek-V3的训练语料库包含14.8T高质量且多样化的token，使用的是DeepSeek-V3的分词器**。 

在DeepSeekCoder-V2（DeepSeek-AI, 2024a）的训练过程中，观察到Fill-in-Middle（FIM）策略在不损害下一个token预测能力的同时，能够根据上下文线索准确预测中间文本。

与DeepSeekCoder-V2一致，也在**DeepSeek-V3的预训练中引入了FIM策略**。具体而言，采用Prefix-Suffix-Middle（PSM）框架来结构化数据，如下所示：   

<|fim_begin|> pre <|fim_hole|> suf <|fim_end|> middle <|eos_token|>。 

这种结构在文档级别作为预打包过程的一部分应用。**FIM策略的使用率为0.1，与PSM框架一致**。 

DeepSeek-V3的分词器使用字节级BPE（Shibata et al., 1999），词汇量扩展到128K token。预分词器和训练数据经过修改以优化多语言压缩效率。

此外，与DeepSeek-V2相比，**新的预分词器引入了结合标点符号和换行符的token**。然而，这种技巧可能会在处理没有终端换行符的多行提示时引入token边界偏差（Lundberg, 2023），特别是在零样本评估提示中。

为了解决这个问题，在训练过程中随机拆分一定比例的此类结合token，使模型接触到更广泛的特殊情况，并减轻这种偏差。 

**4.2. 超参数**

**模型超参数：**

将Transformer层数设置为61，隐藏维度设置为7168。

所有可学习参数以0.006的标准差随机初始化。

在MLA中，将注意力头数h设置为128，每个头的维度h设置为128。

KV压缩维度设置为512，查询压缩维度设置为1536。

对于解耦的查询和键，将每个头的维度设置为64。

除了前三层外，将所有FFN替换为MoE层。

每个MoE层包含1个共享专家和256个路由专家，每个专家的中间隐藏维度为2048。

在路由专家中，每个token将激活8个专家，并确保每个token最多发送到4个节点。

多token预测深度设置为1，即除了确切的下一个token外，每个token还将预测一个额外的token。

与DeepSeek-V2相同，DeepSeek-V3也在压缩潜在向量后附加额外的RMSNorm层，并在宽度瓶颈处乘以额外的缩放因子。

在该配置下，DeepSeek-V3包含671B个总参数，其中每个token激活37B个参数。 

**训练超参数：**

采用AdamW优化器（Loshchilov and Hutter, 2017），超参数设置为β1 = 0.9，β2 = 0.95，weight_decay = 0.1。

将预训练期间的最大序列长度设置为4K，并在14.8T token上预训练DeepSeek-V3。   

**4.3. 长上下文扩展**

采用与DeepSeek-V2（DeepSeek-AI, 2024c）类似的方法，以启用DeepSeek-V3的长上下文能力。

在预训练阶段之后，**应用YaRN（Peng et al., 2023a）进行上下文扩展**，并执行两个额外的训练阶段，每个阶段包含1000步，逐步将上下文窗口从4K扩展到32K，然后扩展到128K。

YaRN配置与DeepSeek-V2中使用的配置一致，仅应用于解耦的共享键k。

两个阶段的超参数保持不变，其中scale = 40，base = 1，factor = 32，缩放因子√scale = 0.1 ln base + 1。

在第一阶段，序列长度设置为32K，批量大小为1920。

在第二阶段，序列长度增加到128K，批量大小减少到480。

两个阶段的学习率设置为7.3 × 10−6，与预训练阶段的最终学习率匹配。 

**4.4. 评估**

**4.4.1. 评估基准**

DeepSeek-V3的基础模型在多语言语料库上预训练，其中英语和中文占多数，因此在一系列主要以英语和中文为主的基准以及多语言基准上评估其性能。

评估基于内部评估框架，该框架集成在HAI-LLM框架中。考虑的基准分为以下几类，其中下划线基准为中文，双下划线基准为多语言： 

多主题多项选择数据集包括MMLU（Hendrycks et al., 2020），MMLU-Redux（Gema et al., 2024），MMLU-Pro（Wang et al., 2024b），MMMLU（OpenAI, 2024b），C-Eval（Huang et al., 2023），和CMMLU（Li et al., 2023）。 

语言理解和推理数据集包括HellaSwag（Zellers et al., 2019），PIQA（Bisk et al., 2020），ARC（Clark et al., 2018），和BigBench Hard（BBH）（Suzgun et al., 2022）。 

闭卷问答数据集包括TriviaQA（Joshi et al., 2017）和NaturalQuestions（Kwiatkowski et al., 2019）。   

阅读理解数据集包括RACE（Lai et al., 2017），DROP（Dua et al., 2019），C3（Sun et al., 2019a），和CMRC（Cui et al., 2019）。 

参考消歧数据集包括CLUEWSC（Xu et al., 2020）和WinoGrande（Sakaguchi et al., 2019）。 

语言建模数据集包括Pile（Gao et al., 2020）。 

中文理解和文化数据集包括CCPM（Li et al., 2021）。 

数学数据集包括GSM8K（Cobbe et al., 2021），MATH（Hendrycks et al., 2021），MGSM（Shi et al., 2023），和CMath（Wei et al., 2023）。 

代码数据集包括HumanEval（Chen et al., 2021），LiveCodeBench-Base（0801-1101）（Jain et al., 2024），MBPP（Austin et al., 2021），和CRUXEval（Gu et al., 2024）。 

标准化考试包括AGIEval（Zhong et al., 2023）。注意，AGIEval包括英语和中文子集。 

遵循先前的工作（DeepSeek-AI, 2024b,c），对于包括HellaSwag，PIQA，WinoGrande，RACE-Middle，RACE-High，MMLU，MMLU-Redux，MMLU-Pro，MMMLU，ARC-Easy，ARC-Challenge，C-Eval，CMMLU，C3，和CCPM在内的数据集，采用基于困惑度的评估；对于TriviaQA，NaturalQuestions，DROP，MATH，GSM8K，MGSM，HumanEval，MBPP，LiveCodeBench-Base，CRUXEval，BBH，AGIEval，CLUEWSC，CMRC，和CMath，采用生成式评估。此外，对Pile-test进行语言建模评估，并使用Bits-Per-Byte（BPB）作为度量标准，以确保使用不同分词器的模型之间进行公平比较。 

**4.4.2. 评估结果**

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/iaJfIYygOIiaeTTs1XNibdt1P9uUKapGkfkbCPsbvqICKxib1MpMCttHuxb0PW5A5UZLwvxMrDNcVZjGYdtaxF16yg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



在表3中，对比了DeepSeek-V3的基础模型与当前最先进的开源基础模型，包括DeepSeek-V2-Base（DeepSeek-AI, 2024c）（之前的版本），Qwen2.5 72B Base（Qwen, 2024b），以及LLaMA-3.1 405B Base（AI@Meta, 2024b）。

所有这些模型都通过内部评估框架进行了评估，并确保了相同的评估设置。需要注意的是，由于评估框架在过去几个月中的变化，DeepSeek-V2-Base的性能与之前报告的结果略有不同。

总体而言，**DeepSeek-V3-Base在大多数基准测试中全面超越了DeepSeek-V2-Base和Qwen2.5 72B Base，并在大多数基准测试中超过了LLaMA-3.1 405B Base，实际上成为了最强的开源模型**。   

**4.5. 讨论**

**4.5.1. 多Token预测（MTP）策略的消融研究**

在表4中，展示了MTP策略的消融结果。具体来说，该策略在不同规模的两个基线模型上进行了验证。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/iaJfIYygOIiaeTTs1XNibdt1P9uUKapGkfkAe3YHBQnpU1Gh9Na65sbicialIVj6h3F1Tic88NMAODTLhZibIfQ0HL8mA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

在小规模上，训练了一个包含157亿参数的基线MoE模型，使用了1.33万亿个token。在大规模上，训练了一个包含2287亿参数的基线MoE模型，使用了5400亿个token。在这些基线模型的基础上，保持训练数据和其他架构不变，添加了一个1层的MTP模块，并训练了两个使用MTP策略的模型进行比较。

需要注意的是，在推理过程中，直接丢弃了MTP模块，因此比较模型的推理成本完全相同。从表中可以看出，MTP策略在大多数评估基准上一致地提升了模型性能。 

**4.5.2. 无辅助损失平衡策略的消融研究**

在表5中，展示了无辅助损失平衡策略的消融结果。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/iaJfIYygOIiaeTTs1XNibdt1P9uUKapGkfkx5YRWQibKFJ9nNWvdhMwGn3IQ3jvuRbrvKsAVvIhxaR3d85gOaicqpicQ/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

该策略在不同规模的两个基线模型上进行了验证。

在小规模上，训练了一个包含157亿参数的基线MoE模型，使用了1.33万亿个token。

在大规模上，训练了一个包含2287亿参数的基线MoE模型，使用了5780亿个token。

这两个基线模型仅使用辅助损失来鼓励负载均衡，并使用带有top-K亲和性归一化的sigmoid门控函数。它们控制辅助损失强度的超参数分别与DeepSeek-V2-Lite和DeepSeek-V2相同。

在这些两个基线模型的基础上，保持训练数据和其他架构不变，移除了所有辅助损失，并引入了无辅助损失平衡策略进行比较。从表中可以看出，无辅助损失策略在大多数评估基准上一致地实现了更好的模型性能。   

**4.5.3. 批量负载均衡与序列负载均衡**

辅助损失自由平衡与序列辅助损失的关键区别在于它们的均衡范围：批量负载均衡与序列负载均衡。与序列辅助损失相比，批量负载均衡施加了更灵活的约束，因为它不对每个序列施加域内平衡。这种灵活性允许专家更好地在不同领域中专业化。

为了验证这一点，记录并分析了16B辅助损失基线模型和16B辅助损失自由模型在Pile测试集不同领域的专家负载。如图9所示，观察到辅助损失自由模型如预期所示表现出更强的专家专业化模式。 

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/iaJfIYygOIiaeTTs1XNibdt1P9uUKapGkfkxfhr7OoPfLjuvX6pmCtK85uPNOo3h9mWZbp8gm0XnX438WaAK1gD1w/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



为进一步研究这种灵活性与模型性能优势之间的相关性，还设计并验证了一种批量辅助损失，该损失鼓励在每个训练批次而不是每个序列上进行负载均衡。实验结果表明，在实现相似的批量负载均衡水平时，批量辅助损失也可以实现与辅助损失自由方法相似的模型性能。

具体来说，在1B MoE模型的实验中，验证损失分别为：2.258（使用序列辅助损失），2.253（使用辅助损失自由方法），以及2.253（使用批量辅助损失）。 

**5. 后训练**

**5.1. 监督微调**

**整理指令微调数据集：**包括跨越多个领域的150万个实例，每个领域使用特定的数据创建方法，以满足其特定需求。 

**推理数据：**对于推理相关的数据集，包括数学、编程竞赛问题和逻辑谜题等，**通过内部DeepSeek-R1模型生成数据**。

具体来说，虽然R1生成的数据具有很强的准确性，但存在过度思考、格式不佳和过长的问题。目标是在保持R1生成推理数据高准确性的同时，平衡常规格式推理数据的清晰性和简洁性。   

**5.2. 强化学习**

**5.2.1. 奖励模型**

在RL过程中，使用了基于规则的奖励模型（RM）和基于模型的RM。 

**基于规则的RM：**对于可以通过特定规则验证的问题，采用基于规则的奖励系统来确定反馈。例如，某些数学问题具有确定的结果，要求模型在指定格式（例如，放在一个框中）内提供最终答案，从而可以应用规则来验证正确性。同样，对于LeetCode问题，可以利用编译器根据测试用例生成反馈。通过尽可能多地利用基于规则的验证，确保了更高的可靠性，因为这种方法不易被操纵或利用。 

**基于模型的RM：**对于具有自由形式的正确答案的问题，依赖奖励模型来确定响应是否与预期的正确答案匹配。相反，对于没有明确正确答案的问题，如涉及创意写作的问题，奖励模型的任务是根据问题和相应的答案提供反馈。 

**5.2.2. 分组相对策略优化**

类似于DeepSeek-V2（DeepSeek-AI, 2024c），采用分组相对策略优化（**GRPO**）（Shao et al., 2024），该方法放弃了通常与策略模型大小相同的批评模型，而是从组分数中估计基线。

在RL过程中，引入了来自不同领域的提示，如编程、数学、写作、角色扮演和问答等。这种方法不仅使模型更接近人类偏好，还增强了在基准测试上的表现，特别是在可用SFT数据有限的情况下。 

**5.3. 评估**

**5.3.1. 评估设置**

**评估基准：**除了用于基础模型测试的基准外，还进一步在IFEval（Zhou et al., 2023）、FRAMES（Krishna et al., 2024）、LongBench v2（Bai et al., 2024）、GPQA（Rein et al., 2023）、SimpleQA（OpenAI, 2024c）、C-SimpleQA（He et al., 2024）、SWE-Bench Verified（OpenAI, 2024d）、Aider、LiveCodeBench（Jain et al., 2024）（2024年8月至11月的问题）、Codeforces、中国国家高中数学奥林匹克竞赛（CNMO 2024）以及美国邀请数学考试2024（AIME 2024）（MAA, 2024）上评估指令模型。 

**比较基线：**对聊天模型进行了全面评估，与多个强大的基线进行了比较，包括DeepSeek-V2-0506、DeepSeek-V2.5-0905、Qwen2.5 72B Instruct、LLaMA-3.1 405B Instruct、Claude-Sonnet-3.5-1022和GPT-4o-0513。对于DeepSeek-V2模型系列，选择了最具代表性的变体进行比较。对于闭源模型，通过各自的API进行评估。 

**5.3.2. 标准评估**

表6展示了评估结果，显示DeepSeek-V3是表现最好的开源模型。此外，它在GPT-4o和Claude-3.5-Sonnet等前沿闭源模型中也具有竞争力。  

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/iaJfIYygOIiaeTTs1XNibdt1P9uUKapGkfkyIEyCib7Qs2Zk8f4BQMicfeZSWeics74RapbZ22VxMkHcgMJDSc3SSZrw/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

 

**英文基准：**MMLU是一个广泛认可的基准，用于评估大型语言模型在各种知识领域和任务中的表现。DeepSeek-V3表现出竞争力，与顶级模型如LLaMA-3.1-405B、GPT-4o和Claude-Sonnet 3.5相当，而显著优于Qwen2.5 72B。

此外，DeepSeek-V3在MMLU-Pro上表现出色，这是一个更具挑战性的教育知识基准，其中DeepSeek-V3紧随Claude-Sonnet 3.5之后。在MMLU-Redux上，一个经过修正标签的MMLU精简版本，DeepSeek-V3超越了竞争对手。

此外，在GPQA-Diamond上，一个博士级评估测试平台，DeepSeek-V3取得了显著成果，仅次于Claude 3.5 Sonnet，并大幅领先其他所有竞争对手。 

**在长上下文理解基准测试**如DROP、LongBench v2和FRAMES中，DeepSeek-V3继续表现出顶级模型的地位。在DROP的3-shot设置中，DeepSeek-V3取得了91.6的F1分数，超越了该类别中的所有其他模型。

在FRAMES上，一个需要在100k token上下文中进行问答的基准测试，DeepSeek-V3紧随GPT-4o之后，但显著超越了所有其他模型。这展示了DeepSeek-V3在处理极端长上下文任务中的强大能力。

DeepSeek-V3的长上下文能力在LongBench v2上得到了进一步验证，该数据集在DeepSeek V3发布前几周才发布。在事实知识基准测试SimpleQA上，DeepSeek-V3落后于GPT-4o和Claude-Sonnet，主要是由于其设计重点和资源分配。

**DeepSeek-V3分配了更多的训练token来学习中文知识**，从而在C-SimpleQA上表现出色。在指令跟随基准测试上，DeepSeek-V3显著优于其前身DeepSeek-V2系列，突显了其改进的理解和遵循用户定义格式约束的能力。 

**代码和数学基准：**编程是LLM面临的具有挑战性和实用性的任务，包括工程任务如SWE-Bench-Verified和Aider，以及算法任务如HumanEval和LiveCodeBench。

**在工程任务中**，DeepSeek-V3落后于Claude-Sonnet-3.5-1022，但显著优于开源模型。开源的DeepSeek-V3有望推动与编程相关的工程任务的发展。通过提供其强大的功能，DeepSeek-V3可以推动软件工程和算法开发领域的创新和改进，使开发者和研究人员能够突破开源模型在编程任务中的界限。

**在算法任务中，**DeepSeek-V3表现出色，超越了所有基准测试中的基线，如HumanEval-Mul和LiveCodeBench。这种成功可以归因于其先进的知识蒸馏技术，该技术有效增强了其在算法任务中的代码生成和问题解决能力。   

**在数学基准测试中，**DeepSeek-V3表现出色，显著超越了基线，并为非o1-like模型设定了新的最佳性能。具体来说，在AIME、MATH-500和CNMO 2024上，DeepSeek-V3比第二好的模型Qwen2.5 72B高出约10个百分点。 

**5.3.3. 开放式评估**

除了标准基准测试外，还使用LLM作为裁判对模型进行了开放式生成任务的评估，结果见表7。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/iaJfIYygOIiaeTTs1XNibdt1P9uUKapGkfkVNjpgUA2mLfZAjn3blB5a1jobMT9mcTxQUicBlwN5LfDKSicwW4KK24A/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

具体来说，遵循了AlpacaEval 2.0（Dubois et al., 2024）和Arena-Hard（Li et al., 2024a）的原始配置，这些配置利用GPT-4-Turbo-1106作为裁判进行成对比较。在Arena-Hard上，DeepSeek-V3取得了超过86%的胜率。 

同样，DeepSeek-V3在AlpacaEval 2.0上表现出色，超越了闭源和开源模型。这展示了其在写作任务和处理简单问答场景中的出色能力。值得注意的是，它比DeepSeek-V2.5-0905高出约20个百分点。 

**5.3.4. DeepSeek-V3 as a Generative Reward Model**

在这一部分中，DeepSeek-V3的判断能力与最先进的模型GPT-4o和Claude-3.5进行了比较。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/iaJfIYygOIiaeTTs1XNibdt1P9uUKapGkfkFrHxt1TshQnRusP0T1uJKJy0tf1ezVgO2kvPSKa9sNs1VGZ8NKWrxg/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

表8展示了这些模型在RewardBench（Lambert等人，2024）上的表现。DeepSeek-V3的表现与GPT-4o-0806和Claude-3.5-Sonnet-1022的最佳版本相当，同时超越了其他版本。此外，DeepSeek-V3的判断能力还可以通过投票技术得到增强。因此，采用了DeepSeek-V3结合投票技术来提供开放式问题的自我反馈，从而提高了模型的性能。   

**5.4. 讨论**

**5.4.1. 从DeepSeek-R1的蒸馏**

基于DeepSeek-V2.5，研究了从DeepSeek-R1的蒸馏对模型贡献的影响。基线模型使用的是短CoT（Chain of Thought）数据，而竞争对手则使用了上述专家检查点生成的数据。

![图片](https://mmbiz.qpic.cn/sz_mmbiz_png/iaJfIYygOIiaeTTs1XNibdt1P9uUKapGkfk53fccBib9s0r9ttlnOmM98lMg0lYd1lNlib0x3m9U251aTvpPksRHNsA/640?wx_fmt=png&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

表9展示了蒸馏数据的有效性，显示在LiveCodeBench和MATH-500基准测试中都有显著的性能提升。

实验发现了一个有趣的权衡：蒸馏虽然提高了性能，但也大幅增加了平均响应长度。为了在模型准确性和计算效率之间保持平衡，对DeepSeek-V3的蒸馏进行了仔细的参数选择。 

研究结果表明，从推理模型的知识蒸馏为后训练优化提供了一个有前景的方向。虽然当前的工作主要集中在从数学和编程领域的数据进行蒸馏，但这种方法在其他任务领域也有潜在的应用价值。在这些特定领域的有效性表明，长CoT蒸馏可能对需要复杂推理的其他认知任务的模型性能提升有价值。未来的研究将继续探索这一方法在不同领域的应用。 

**5.4.2. 自我奖励**

奖励在强化学习（RL）中起着核心作用，引导优化过程。在可以通过外部工具轻松验证的领域，如某些编程或数学场景中，RL表现出极高的效率。然而，在更一般的场景中，构建反馈机制变得更具挑战性。自我奖励机制成为一种可能的解决方案，它允许模型根据自身生成的内容进行奖励，从而引导模型的优化过程。 

**5.4.3. 多Token预测评估**

与传统的单Token预测不同，DeepSeek-V3通过MTP（Multi-Token Prediction）技术预测接下来的2个Token。结合推测性解码框架（Leviathan等人，2023；Xia等人，2023），这种方法可以显著加快模型的解码速度。一个自然的问题是额外预测的Token的接受率。根据评估，第二个Token预测的接受率在85%到95%之间。这一高接受率表明，MTP技术不仅提高了解码速度，还保持了较高的预测准确性。   

**6. 结论、局限性和未来方向**

本文介绍了DeepSeek-V3，这是一个具有671B个总参数和37B个激活参数的大型MoE（Mixture of Experts）语言模型，训练数据量达到14.8T Token。

除了MLA和DeepSeekMoE架构外，DeepSeek-V3还开创了无辅助损失的负载均衡策略，并设置了多Token预测训练目标以提升性能。DeepSeek-V3的训练成本效益高，得益于FP8训练和精细的工程优化。

后训练阶段也成功地从DeepSeek-R1系列模型中蒸馏了推理能力。

**全面的评估表明，DeepSeek-V3已成为目前最强的开源模型之一，并且在性能上与GPT-4o和Claude-3.5-Sonnet等领先闭源模型相当。尽管性能强大，DeepSeek-V3仍保持了较低的训练成本，仅需2.788M H800 GPU小时完成全部训练，包括预训练、上下文长度扩展和后训练。** 

尽管认可其强大的性能和成本效益，也认识到DeepSeek-V3存在一些局限性，尤其是在部署方面。

首先，为了确保高效的推理，DeepSeek-V3的推荐部署单元相对较大，这可能对小型团队造成负担。

其次，尽管DeepSeek-V3的部署策略实现了超过两倍于DeepSeek-V2的端到端生成速度，但仍存在进一步提升的空间。

幸运的是，这些局限性预计会随着更先进硬件的发展而自然解决。 

DeepSeek始终遵循开源模型的长期主义路线，致力于稳步接近AGI（人工通用智能）的终极目标。 



参考

1. [deepseek api参考](https://api-docs.deepseek.com/zh-cn/)