# Google提示工程白皮书 解读

> 谷歌近日发布了一份长达 68 页的白皮书，系统阐述了提示工程（Prompt Engineering）的核心理念与最佳实践。这份白皮书原版是英文，为了方便自己学习理解，我将其翻译成中文（利用各种LLM工具），并整理出这篇文章，分享给需要的人。需要注意的是，我是基于[Google官方提示工程原版网站](https://www.kaggle.com/whitepaper-prompt-engineering)提供的v7版本[pdf](https://github.com/lewlh/share/blob/main/documents/PromptEngineeringV7.pdf)进行翻译。本译文已经生成PDF,读者朋友们也可以直接下载[PromptEngineering中文翻译.pdf](https://github.com/lewlh/share/blob/main/documents/PromptEngineering中文翻译.pdf)



## 大语言模型输出配置

选择模型后，您需要确定模型配置。大多数大语言模型提供各种控制模型输出的配置选项。有效的提示词工程需要为您的任务优化这些配置。

### 输出长度

一个重要的配置设置是响应中生成的Token数量。生成更多Token需要大语言模型进行更多计算，导致更高的能耗、可能更慢的响应时间和更高的成本。

减少大语言模型的输出长度不会使模型在输出中变得更加风格化或简洁，它只会导致大语言模型在达到限制时停止预测更多Token。如果您的需求需要较短的输出长度，您可能还需要设计提示词以适应。

输出长度限制对于某些大语言模型提示词技术（如ReAct）尤为重要，其中大语言模型在您想要的响应之后会继续发出无用的Token。

请注意，生成更多Token需要大语言模型进行更多计算，导致更高的能耗和可能更慢的响应时间，从而增加成本。

### 采样控制

大语言模型并不正式预测单个Token。相反，大语言模型预测下一个Token可能是什么的概率，为大语言模型词汇表中的每个Token分配一个概率。然后对这些Token概率进行采样以确定下一个生成的Token。温度、Top-K和Top-P是最常见的配置设置，用于确定如何处理预测的Token概率以选择单个输出Token。

### 温度

温度控制Token选择的随机性程度。较低的温度适用于期望更确定性响应的提示词，而较高的温度可能导致更多样化或意外的结果。温度为0（贪婪解码）是确定性的：总是选择最高概率的Token（尽管请注意，如果两个Token具有相同的最高预测概率，根据如何实现平局处理，您可能不会总是得到相同的输出，温度为0）。

接近最大值的温度倾向于创建更随机的输出。随着温度越来越高，所有Token成为下一个预测Token的可能性变得相等。

Gemini温度控制可以类似于机器学习中使用的softmax函数来理解。低温度设置反映了低softmax温度（T），强调单一、首选温度，具有高度确定性。较高的Gemini温度设置类似于高softmax温度，使围绕所选设置的可接受温度范围更广。这种增加的不确定性适应了不需要严格、精确温度的场景，例如在尝试创造性输出时。

### Top-K与Top-P

Top-K和Top-P（也称为核采样）是大语言模型中使用的两种采样设置，用于将预测的下一个Token限制为来自具有最高预测概率的Token。与温度类似，这些采样设置控制生成文本的随机性和多样性。

Top-K采样从模型的预测分布中选择前K个最可能的Token。Top-K越高，模型的输出越有创造性和多样性；Top-K越低，模型的输出越受限、越具有事实性。Top-K为1等同于贪婪解码。

Top-P采样选择其累积概率不超过某个值（P）的顶部Token。P的值范围从0（贪婪解码）到1（大语言模型词汇表中的所有Token）。

选择Top-K和Top-P之间的最佳方法是尝试两种方法（或两者一起），看看哪种方法产生您正在寻找的结果。

## 参考
1. Google提示工程白皮书《Prompt Engineering》: https://www.promptingguide.ai/zh
2. Google提示工程中文翻译: https://lewlh.github.io/2025/04/15/PromptEngineering
3. https://mmssai.com/archives/36018
4. PromptEngineering中文翻译.pdf: https://github.com/lewlh/share/blob/main/documents/PromptEngineering%E4%B8%AD%E6%96%87%E7%BF%BB%E8%AF%91.pdf
5. 25000字长文详细讲述提示词工程-2025谷歌白皮书：如何通过提示词工程优化AI模型: https://mp.weixin.qq.com/s/V2aXW9Z1RgNwJYPaIhGVgg