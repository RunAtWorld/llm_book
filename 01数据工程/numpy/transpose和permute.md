# PyTorch的两大转置函数 transpose() 和 permute()

在`pytorch`中转置用的函数就只有这两个

1. `transpose()`
2. `permute()`

注意只有`transpose()`有后缀格式：`transpose_()`：后缀函数的作用是简化如下代码：

```
x = x.transpose(0,1)
等价于
x.transpose_()
# 相当于x = x + 1 简化为 x+=1
```

### transpose

```
torch.transpose(input, dim0, dim1, out=None) → Tensor
```

函数返回输入矩阵`input`的转置。交换维度`dim0`和`dim1`

参数:

- input (Tensor) – 输入张量，必填
- dim0 (int) – 转置的第一维，默认0，可选
- dim1 (int) – 转置的第二维，默认1，可选

**如何理解transpose操作**，transpose的意思是：调换，换位。

虽然numpy的transpose函数和Pytorch实现不同，但理解原理是一样的。

````
import numpy as np
t = np.arange(1,17).reshape(2, 2, 4)
print(t)
```
[[[ 1  2  3  4]
  [ 5  6  7  8]]

 [[ 9 10 11 12]
  [13 14 15 16]]]
```
````

1. 代数理解：比如 `t2 = t.transpose(2, 0, 1)` , 即 t: (0, 1, 2)` → `t1: (2, 0, 1)，比如t[0,1,3]=8，那么t2就是按照前面的规则，位置上值等于7的坐标就变成[3,0,1]，即 t[3,0,1]=7
2. 几何理解：高维的情况下，轴变换本质上就是各元素的维度坐标，按照变换后的轴进行的新的维度空间的改变。类似数学当中坐标系的变化
3. 应用层面理解：(0,1,2) ->(2,0,1) 对应 (H,W,C)->(C,H,W)

### permute

```
permute(dims) → Tensor
```

将tensor的维度换位。

参数：

* dims (int…*) -换位顺序，必填

### 异同点

相同点

1. 都是返回转置后矩阵。
2. 都可以操作高纬矩阵，`permute`在高维的功能性更强。

不同点

```
# 创造二维数据x，dim=0时候2，dim=1时候3
x = torch.randn(2,3)       'x.shape  →  [2,3]'
# 创造三维数据y，dim=0时候2，dim=1时候3，dim=2时候4
y = torch.randn(2,3,4)   'y.shape  →  [2,3,4]'
```

1. 使用方法不同

`torch.transpose(x)`合法， `x.transpose()`合法。
`tensor.permute(x)`**不**合法，`x.permute()`合法。

2. 操作`dim`不同：`transpose()`只能一次操作两个维度；`permute()`可以一次操作多维数据，且必须传入所有维度数，因为`permute()`的参数是`int*`。

```
# 对于transpose
x.transpose(0,1)     'shape→[3,2] '  
x.transpose(1,0)     'shape→[3,2] '  
y.transpose(0,1)     'shape→[3,2,4]' 
y.transpose(0,2,1)  'error，操作不了多维'

# 对于permute()
x.permute(0,1)     'shape→[2,3]'
x.permute(1,0)     'shape→[3,2], 注意返回的shape不同于x.transpose(1,0) '
y.permute(0,1)     "error 没有传入所有维度数"
y.permute(1,0,2)  'shape→[3,2,4]'
```

3. `transpose()`中的`dim`没有数的大小区分；`permute()`中的`dim`有数的大小区分

   举例，注意后面的`shape`：

```
# 对于transpose，不区分dim大小
x1 = x.transpose(0,1)   'shape→[3,2] '  
x2 = x.transpose(1,0)   '也变换了，shape→[3,2] '  
print(torch.equal(x1,x2))
' True ，value和shape都一样'

# 对于permute()
x1 = x.permute(0,1)     '不同transpose，shape→[2,3] '  
x2 = x.permute(1,0)     'shape→[3,2] '  
print(torch.equal(x1,x2))
'False，和transpose不同'

y1 = y.permute(0,1,2)     '保持不变，shape→[2,3,4] '  
y2 = y.permute(1,0,2)     'shape→[3,2,4] '  
y3 = y.permute(1,2,0)     'shape→[3,4,2] '  
```

4.关于连续contiguous()
用view()函数改变通过转置后的数据结构，导致报错
`RuntimeError: invalid argument 2: view size is not compatible with input tensor's....`

这是因为tensor经过转置后数据的内存地址不连续导致的,也就是tensor . is_contiguous()==False
这时候reshape()可以改变该tensor结构，但是view()不可以，具体不同可以看view和reshape的区别
例子如下：

```
x = torch.rand(3,4)
x = x.transpose(0,1)
print(x.is_contiguous()) # 是否连续
'False'
# 再view会发现报错
x.view(3,4)
'''报错
RuntimeError: invalid argument 2: view size is not compatible with input tensor's....
'''

# 但是下面这样是不会报错。
x = x.contiguous()
x.view(3,4)
```

调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一样。

**每次在使用view()之前，该tensor只要使用了transpose()和permute()这两个函数一定要contiguous()**

### reshape

再看看`reshape()`

```
 x = torch.rand(3,4)
x = x.permute(1,0) # 等价x = x.transpose(0,1)
x.reshape(3,4)
'''这就不报错了
说明x.reshape(3,4) 这个操作
等于x = x.contiguous().view()
尽管如此，但是torch文档中还是不推荐使用reshape
理由是除非为了获取完全不同但是数据相同的克隆体
'''
```

### 参考

1. [PyTorch 两大转置函数](https://blog.csdn.net/xinjieyuan/article/details/105232802) . CSDN
2. [Numpy中transpose()函数的可视化理解](https://zhuanlan.zhihu.com/p/61203757) . 知乎