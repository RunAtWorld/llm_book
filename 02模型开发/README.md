# 大语言模型地图
> 以下大模型特指：大语言模型，LLM



## 1、大模型的介绍

1. Transfomer
2. 注意力机制
3. self-attention
4. word-embedding
5. flash-attention
6. encoder,decoder

## 2、大模型解决什么问题

1. 自然语言对话
2. 逻辑推理


## 3、常见的大模型

按国别分
1. 国外
  - GPT系列
    - GPT3
    - GPT3.5
    - GPR4
  - llama系列
    - llama
    - llama2
  - Bloom

2. 国内
    - 清华智谱-GLM
      - [ChatGLM](https://chatglm.cn/)
    - 百度-文心一言
    - 阿里-通义千问
    - 科大讯飞-星火认知
    - 百川

## 4、大模型训练
### 4.1 全量训练

### 4.2 微调
> 微调的目的：

1. LoRA
2. Adapter
3. RLHF

## 5、大模型推理
> 要解决的核心问题:"存得下" "跑得快"

### 5.1 推理和训练的差异

### 5.2 推理优化的方法

软件方面
1. 量化
2. 剪枝
3. 蒸馏

硬件方面
1. 算子融合、图优化

## 6、应用生态
1. langchain
2. ai agent

## 7、大模型发展趋势

## 8、参考

论文
1. [GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/pdf/2103.10360.pdf)

课程
1. [【清华NLP】刘知远团队大模型公开课全网首发｜带你从入门到实战](https://www.bilibili.com/video/BV1UG411p7zv)
2. [【ChatGLM3保姆级教程】安装部署、性能详解、实战应用，零基础入门到应用](https://www.bilibili.com/video/BV1Hc411q76k)
3. [大模型微调实战](https://www.bilibili.com/video/BV1dN4y167vX)
4. [大模型微调和实战：大模型微调方法原理及大模型主流技术架构全详解-北大AI博士](https://www.bilibili.com/video/BV1NC4y1c7ih)
5. [国产大模型ChatGLM3-6B微调](https://www.bilibili.com/video/BV1y64y1G78N)

仓库
1. [llm-action](https://github.com/liguodongiot/llm-action)