# HuggingFace


## 如何快速下载huggingface模型——全方法总结
> https://zhuanlan.zhihu.com/p/663712983

### 使用镜像HF-MIRROR
> https://hf-mirror.com/

方法一：使用huggingface 官方提供的 huggingface-cli 命令行工具。

(1) 安装依赖
```
pip install -U huggingface_hub
```
(2) 基本命令示例：
```
export HF_ENDPOINT=https://hf-mirror.com
```

使用样例
```
huggingface-cli download --resume-download --local-dir-use-symlinks False bigscience/bloom-560m --local-dir bloom-560m
```
(3) 下载需要登录的模型（Gated Model）
请添加--token hf_***参数，其中hf_***是 access token，请在huggingface官网这里获取。示例：
```
huggingface-cli download --token hf_*** --resume-download --local-dir-use-symlinks False meta-
```

方法二：使用url直接下载时，将 huggingface.co 直接替换为本站域名hf-mirror.com。使用浏览器或者 wget -c、curl -L、aria2c 等命令行方式即可。
下载需登录的模型需命令行添加 --header hf_*** 参数，token 获取具体参见上文。
方法三：(非侵入式，能解决大部分情况)huggingface 提供的包会获取系统变量，所以可以使用通过设置变量来解决。
```
HF_ENDPOINT=https://hf-mirror.com python your_script.py
```


### 使用 hf_transfer

开启方法

(1)安装依赖
```
pip install -U hf-transfer
```
(2)设置 HF_HUB_ENABLE_HF_TRANSFER 环境变量为 1。
Linux
```
export HF_HUB_ENABLE_HF_TRANSFER=1
```
Windows Powershell
```
$env:HF_HUB_ENABLE_HF_TRANSFER = 1
```

开启后使用方法同 huggingface-cli：
```
huggingface-cli download --resume-download bigscience/bloom-560m --local-dir bloom-560m
```