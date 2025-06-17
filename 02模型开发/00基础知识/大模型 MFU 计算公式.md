直接给出结论，推导过程在下方：
GPT 模型：
$$
72sbh^2L(1 + \frac{s}{6h} + \frac{V}{12hL})
$$

LLAMA2-70B 模型：
$$
3×L×(4sbh^2 + 4sbh^2×\frac{1}{r} + 4s^2bh + 6sbhf) + 6sbhV
$$

Grok-314B 模型：
$$
3×L×(4sbh^2 + 4sbh^2×\frac{1}{r} + 4s^2bh + 6sbhf×topk) + 6sbhV
$$

按照 Megatron-LM 官方仓库提供的计算方法，统一化简一下，在 llama 架构与 MoE 中能得到下面的计算公式：
$$
12sbh^2L(1 + \frac{1}{r} + \frac{s}{h} + topk×\frac{3}{2}\frac{f}{h} + \frac{V}{2hL})
$$

参数说明：
| Symbol |   Description   | Symbol |     Description     |
|--------|-----------------|--------|---------------------|
|    L   |    num layers   |    b   | training batch size |
|    s   |    seq length   |    h   |     hidden size     |
|    f   |    ffn size     |    r   | num_attn_heads / num_query_groups |
|    V   | vocabulary size |  topk  |    top K experts    |

以下为开源代码实现：
```python
def num_floating_point_operations(args, batch_size):
    # Group Query Attention.
    if not args.group_query_attention:
        args.num_query_groups = args.num_attention_heads
    # MoE.
    num_experts_routed_to = 1 if args.num_experts is None else args.moe_router_topk
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1
    return (
        12
        * batch_size
        * args.seq_length
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            1
            + (
                (args.ffn_hidden_size / args.hidden_size)
                * num_experts_routed_to
                * gated_linear_multiplier
            )
            + (args.num_query_groups / args.num_attention_heads)
            + (args.seq_length / args.hidden_size)
            + (args.padded_vocab_size / (2 * args.num_layers * args.hidden_size))
        )
    )
```


关于该项指标的定义与计算，可以先参考 NV 的这两篇文章 [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/pdf/2104.04473)，寻找关键字*APPENDIX: FLOATING-POINT OPERATIONS* 与 [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/pdf/2205.05198)，寻找关键字*FLOPs Calculation*.

计算 MFU (Model FLOP Utilization，模型浮点利用率) 的关键是我们遵从**大语言模型的计算操作主要集中在矩阵乘**的客观事实与**矩阵乘的计算包含乘加两个操作**的计算规则：两个矩阵相乘 $A_{m × k}$ × $B_{k × n}$ 带来的浮点操作数为 $2×m×k×n$.
也因为上面的两点，我们其实倾向于在 transformer 架构下计算 MFU，原因就在于整个 transformer 中从 embedding 到 attention 再到 mlp 都是满满的矩阵乘，那么我们跟着论文逐步将计算公式给推导出来：
这里，我们先给出两种常见的业内主流大模型的架构：GPT 与 LLAMA

先看 transformer，如果抽离一下，我们发现大致的结构就是

$$
Word Embedding + (Attention + MLP) * L
$$

在过上面的几部分的时候，输入与输出是保持着 $[s, b, h]$ 的 shape 的。

* Word Embedding：
  
  $$
  [s, b, h] × [h, V]  ==>  2sbhV
  $$

* Attention:

  $$
  Q, K, V = xW_Q, xW_K, xW_V\\
  O = xW_O\\
  attn_{score} = softmax(\frac{QK^T}{\sqrt{h}})\\
  context = attn_{score} V\\
  proj_{res} = context ∗ W_O
  $$

拆分上面的每一条公式：

$$
Q, K, V\ projection : [s, b, h] × [h, h] ==> [s, b, h]\tag{1}\\
O\ projection : [s, b, h] × [h, h] ==> [s, b, h]
$$

考虑 *GQA* 与 *MHA* 两种情况：

GQA ，在获取阶段产生的浮点操作如下:

$$
2sbh^{2} + 2sbh/r * h * 2
$$

MHA，在获取阶段产生的浮点操作如下:

$$
2sbh^{2} * 3
$$

式子（1）是获取 $Q, K, V$ 的总概，如果是 $GQA$ 场景的话，$K V$ 的获取需要重新定义。

$$
Q K^{T} : [s, b, h], [s, b, h] --> [b, s, h] × [b, h, s] ==> [b, s, s]
$$

这里产生的浮点操作为 $2s^2bh$

$$
attn_{score} V : [b, s, s] × [b, s, h] ==>[b, s, h]
$$

这里产生的浮点操作依旧为 $2s^2bh$

$$
proj=context ∗W_O : [b, s, h] × [h, h]
$$

最后的线性映射层产生的浮点操作为 $2sbh^2$
综上，在 $attention$ 模块产生的总的浮点操作为：
MHA :

$$
8sbh^2 + 4s^2bh
$$

GQA :

$$
4sbh^2 + 4sbh^2 * \frac{1}{r} + 4s^2bh
$$

在过完 attention 模块后，接下来就是进入到 mlp 部分

* MLP

先看 GPT，很简单，升维-> gelu 激活 -> 降维：

$$
[s, b, h] × [h, 4h] ==> [s, b, 4h]
$$

升维产生的浮点操作为 $8sbh^2$

$$
[s, b, 4h] × [4h, h] ==> [s, b, h]
$$

降维产生的浮点操作依旧为 $8sbh^2$
那么在其 mlp 部分产生的浮点操作为 $16sbh^2$

总计一下 `GPT` 模型的浮点操作（其 attention 部分为 MHA）：

$$
3 × (2sbhV+L × (8sbh^2 + 4s^2bh + 16sbh^2))\\
即为\ 72sbh^2L(1 + \frac{s}{6h} + \frac{V}{12hL})
$$

这里乘以3倍，再乘以 $L$，是因为还得考虑反向的浮点操作（一般记为前向的两倍）与整个 transformer layer 的层数（上面才是考虑的一层的处理）。

我们再来看看 `LLAMA` 架构的，由图可知，我们将其 mlp 部分处理成三块，up，gate，down（对应图中的W3，W1，W2）
up：

$$
[s, b, h] × [h, f] ==> [s, b ,f]
$$

gate：

$$
[s, b ,h] × [h, f] ==> [s, b, f]
$$

down：

$$
[s, b, f] × [f, h] ==> [s, b, h]
$$

那么三者产生的浮点操作是相同的 $2sbhf$，三块就是 $6sbhf$

总计一下 LLAMA2-70B 模型的浮点操作（其 attention 部分为 GQA）：

$$
3×L×(4sbh^2 + 4sbh^2×\frac{1}{r} + 4s^2bh + 6sbhf) + 6sbhV
$$

`MoE` 模型的 MLP 部分的计算：

在模型结构上，单个专家的内部与 `LLAMA` 的一致，存在 gate, up, down 三部分，由此在单个专家内可以产生的浮点操作依旧为 $6sbhf$

考虑到专家选取的 `topk` 规则，在过 MLP 部分的时候，不是全量 token 参与全量专家的计算，重点在于 topk 上，由此该部分的浮点操作为 $6sbhf × topk$
（该处的计算方式与 Megatron 中的给出的是保持一致的）