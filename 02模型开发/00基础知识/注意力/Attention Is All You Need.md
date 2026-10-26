# Attention Is All You Need

## 优化器

### Adam优化器

![image-20250313152713821](Attention Is All You Need.assets/image-20250313152713821.png)

![image-20250313152913325](Attention Is All You Need.assets/image-20250313152913325.png)

这对应于在最初的warmup_steps个训练步骤中线性地增加学习率，并且此后与步骤数的平方根成反比地降低学习率。 我们使用warmup_steps=4000。

## Dropout

残差Dropout

我们在每个子层的输出被添加到子层输入并规范化之前，将其应用于Dropout