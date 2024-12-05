# Ollma

> [ollama](https://ollama.com/): 在本地环境快速构建和运行大模型工具



## 安装

###　docker方式

```
docker pull ollama/ollama
```

CPU only

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
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



## 参考

1. [Ollama](https://ollama.com/)