# LLM中的上下文长度

## 一、上下文长度是什么

在预训练LLM时，通常都会设置一个最大序列长度max_seq_len，也被称为上下文长度或者context window，即模型最多能接受你的输入prompt所占据的token的长度，常见的模型比如[GPT3.5](https://zhida.zhihu.com/search?content_id=256063651&content_type=Article&match_order=1&q=GPT3.5&zhida_source=entity)的上下文长度为4k，[GPT-3.5-turbo](https://zhida.zhihu.com/search?content_id=256063651&content_type=Article&match_order=1&q=GPT-3.5-turbo&zhida_source=entity)的上下文长度为8k。[GPT4](https://zhida.zhihu.com/search?content_id=256063651&content_type=Article&match_order=1&q=GPT4&zhida_source=entity)和[GPT4-turbo](https://zhida.zhihu.com/search?content_id=256063651&content_type=Article&match_order=1&q=GPT4-turbo&zhida_source=entity)的上下文长度为128k，最大输出长度为4k。

## 二、提升上下文长度的主要挑战

提升上下文长度的挑战**不仅关乎训练也关乎推理，并且和模型结构也有关**

### 1.计算资源

众所周知，[Transformer架构](https://zhida.zhihu.com/search?content_id=256063651&content_type=Article&match_order=1&q=Transformer架构&zhida_source=entity)的LLM的计算复杂度为O(n^2)，其中n就是序列长度，也就意味着上下文长度越长，所需要的计算量就越大，并且是呈平方级别增长，因此可能会严重导致输出变慢，并且长度越长，输出速度也会成平方级别的变慢。

以及从推理的角度来说，上下文长度增加意味着prefill阶段的序列长度增加，同时带来了[KV Cache](https://zhida.zhihu.com/search?content_id=256063651&content_type=Article&match_order=1&q=KV+Cache&zhida_source=entity)的增加，即占用显存的增加，因此提升上下文长度从推理的角度上带来了如何高效处理KV Cache的挑战。

### 2.训练复杂度

上下文长度通常是模型训练完之后就确定好了的，训练完之后模型就确定了最多只能处理这么长的序列长度。而在训练阶段，模型通常也会直接训练这么长上下文的语料，那么这么长的长度的输入语料也给模型训练带来了挑战。

主要的技术手段有：

- [梯度累计](https://zhida.zhihu.com/search?content_id=256063651&content_type=Article&match_order=1&q=梯度累计&zhida_source=entity)：该技术允许通过将长序列分解成较小的块来处理长序列，在更新权重之前积累梯度。这对于处理超出 GPU 内存容量的序列至关重要。
- 有效的[Attention机制](https://zhida.zhihu.com/search?content_id=256063651&content_type=Article&match_order=1&q=Attention机制&zhida_source=entity)：处理平方级别的计算复杂度时，需要使用flash attention或者稀疏attention等技术来降低复杂度。
- [Memory-efficient的训练](https://zhida.zhihu.com/search?content_id=256063651&content_type=Article&match_order=1&q=Memory-efficient的训练&zhida_source=entity)：使用可逆层、激活检查点或内存高效优化器等技术来管理长序列增加的内存需求。
- 位置编码：使用[旋转位置编码](https://zhida.zhihu.com/search?content_id=256063651&content_type=Article&match_order=1&q=旋转位置编码&zhida_source=entity)

### 3.位置编码限制

Transformer中网络结构中主要的三个部分就是：编码层（词向量编码和位置编码）、Transformer block和暑促和预测头（predict head)。位置编码对于上下文长度也很重要。

如果是最开始的绝对位置编码，则根本无法想象实现1000万上下文长度（因为就得一个1000万维度的向量了），因此绝对位置编码从原理上就不可以支持长上下文。

后来的相对位置编码如正弦位置编码可以克服这个问题，但是缺点是效果很差，上下文长度一长，PPL困惑度就会很高，后来的Alibi克服了这一问题，但是改变了模型，需要重新预训练模型（相比于主流的GPT3、LLama等模型），太贵了。

而旋转位置编码RoPE既可以通过线性插值实现良好的长度外推从而克服长上下文的PPL困惑度很大的问题，也不需要重新预训练，只要微调即可（便宜多了）。

## 参考

1. [LLM中的上下文长度 · 知乎](https://zhuanlan.zhihu.com/p/1892323473084358661)
2. [附录：基于vLLM不同模型推理支持最小卡数和最大序列说明 - 华为云](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_llm_infer_91034.html)