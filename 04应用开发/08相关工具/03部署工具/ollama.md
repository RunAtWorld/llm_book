# Ollma

> [ollama](https://ollama.com/): 在本地环境快速构建和运行大模型工具



## 安装

### docker方式

```
docker pull ollama/ollama
```

[镜像仓库](https://hub.docker.com/r/ollama/ollama)

CPU only

```bash
docker run -d -v /data/ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

测试

```
docker exec -it ollama ollama run llama3.2:1b
```

执行创建llama3

```
docker exec -it ollama ollama run llama3
```

### linux方式

```
curl -fsSL https://ollama.com/install.sh | sh
```

## 启动

启动[Llama 3.2](https://ollama.com/library/llama3.2):

```
ollama run llama3.2
```

## 模型仓库

> ollama仓库：https://ollama.com/library

| Model              | Parameters | Size  | Download                         |
| ------------------ | ---------- | ----- | -------------------------------- |
| DeepSeek-R1        | 7B         | 4.7GB | `ollama run deepseek-r1`         |
| DeepSeek-R1        | 671B       | 404GB | `ollama run deepseek-r1:671b`    |
| Llama 3.3          | 70B        | 43GB  | `ollama run llama3.3`            |
| Llama 3.2          | 3B         | 2.0GB | `ollama run llama3.2`            |
| Llama 3.2          | 1B         | 1.3GB | `ollama run llama3.2:1b`         |
| Llama 3.2 Vision   | 11B        | 7.9GB | `ollama run llama3.2-vision`     |
| Llama 3.2 Vision   | 90B        | 55GB  | `ollama run llama3.2-vision:90b` |
| Llama 3.1          | 8B         | 4.7GB | `ollama run llama3.1`            |
| Llama 3.1          | 405B       | 231GB | `ollama run llama3.1:405b`       |
| Phi 4              | 14B        | 9.1GB | `ollama run phi4`                |
| Phi 3 Mini         | 3.8B       | 2.3GB | `ollama run phi3`                |
| Gemma 2            | 2B         | 1.6GB | `ollama run gemma2:2b`           |
| Gemma 2            | 9B         | 5.5GB | `ollama run gemma2`              |
| Gemma 2            | 27B        | 16GB  | `ollama run gemma2:27b`          |
| Mistral            | 7B         | 4.1GB | `ollama run mistral`             |
| Moondream 2        | 1.4B       | 829MB | `ollama run moondream`           |
| Neural Chat        | 7B         | 4.1GB | `ollama run neural-chat`         |
| Starling           | 7B         | 4.1GB | `ollama run starling-lm`         |
| Code Llama         | 7B         | 3.8GB | `ollama run codellama`           |
| Llama 2 Uncensored | 7B         | 3.8GB | `ollama run llama2-uncensored`   |
| LLaVA              | 7B         | 4.5GB | `ollama run llava`               |
| Solar              | 10.7B      | 6.1GB | `ollama run solar`               |

> 注意
>
> 7B模型最少 8 GB内存; 13B模型最少16 GB; 33B模型最少32 GB.

可以通过设置环境变量 OLLAMA_MODELS 可以设置下载后的模型存储路径

## 自定义模型

### 从 GGUF 导入

Ollama 支持在 Modelfile 中导入 GGUF 模型：

1. 创建一个名为 的文件`Modelfile`，其中包含`FROM`要导入的模型的本地文件路径的指令。

   ```
   FROM ./vicuna-33b.Q4_0.gguf
   ```

2. 在 Ollama 中创建模型

   ```
   ollama create example -f Modelfile
   ```

3. 运行模型

   ```
   ollama run example
   ```

### 从 Safetensors 导入

请参阅导入模型的[指南以了解更多信息。](https://github.com/ollama/ollama/blob/main/docs/import.md)

### 自定义提示

可以使用提示自定义 Ollama 库中的模型。例如，要自定义`llama3.2`模型：

```
ollama pull llama3.2
```

创建一个`Modelfile`：

```
FROM llama3.2

# set the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1

# set the system message
SYSTEM """
You are Mario from Super Mario Bros. Answer as Mario, the assistant, only.
"""
```

接下来创建并运行模型：

```
ollama create mario -f ./Modelfile
ollama run mario
>>> hi
Hello! It's your friend Mario.
```

有关使用 Modelfile 的更多信息，请参阅[Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)文档。

## CLI 参考

### 创建模型

`ollama create`用于从 Modelfile 创建模型。

```
ollama create mymodel -f ./Modelfile
```

### 拉取模型

```
ollama pull llama3.2
```

> 此命令还可用于更新本地模型。仅会提取差异。

### 删除模型

```
ollama rm llama3.2
```

### 复制模型

```
ollama cp llama3.2 my-model
```

### 多行输入

对于多行输入，你可以使用以下方式换行`"""`：

```
>>> """Hello,
... world!
... """
I'm a basic program that prints the famous "Hello, world!" message to the console.
```

### 多模态模型

```
ollama run llava "What's in this image? /Users/jmorgan/Desktop/smile.png"
```

> **输出**：图像中有一个黄色的笑脸，这可能是图片的中心焦点。

### 将提示作为参数传递

```
ollama run llama3.2 "Summarize this file: $(cat README.md)"
```

> **输出**：Ollama 是一个轻量级、可扩展的框架，用于在本地机器上构建和运行语言模型。它提供了用于创建、运行和管理模型的简单 API，以及可在各种应用程序中轻松使用的预构建模型库。

### 显示模型信息

```
ollama show llama3.2
```

### 列出计算机上的模型

```
ollama list
```

### 列出当前加载的模型

```
ollama ps
```

### 停止当前正在运行的模型

```
ollama stop llama3.2
```

### 启动 Ollama

`ollama serve`当您想启动 ollama 而不运行桌面应用程序时使用。

## 建筑

查看[开发者指南](https://github.com/ollama/ollama/blob/main/docs/development.md)

### 运行本地构建

接下来启动服务器：

```
./ollama serve
```

最后，在一个单独的 shell 中运行一个模型：

```
./ollama run llama3.2
```

## REST API

Ollama 有一个用于运行和管理模型的 REST API。

### 生成响应

```
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt":"Why is the sky blue?"
}'
```

### 与模特聊天

```
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    { "role": "user", "content": "why is the sky blue?" }
  ]
}'
```

查看所有端点的[API 文档。](https://github.com/ollama/ollama/blob/main/docs/api.md)

## 参考

1. [Ollama](https://github.com/ollama/ollama)