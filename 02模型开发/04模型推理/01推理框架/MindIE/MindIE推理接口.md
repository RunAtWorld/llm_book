## MindIE推理接口

### TGI 0.9.4 - https://{ip}:{port}/generate

```
 {
     "inputs": "My name is Olivier and I",
     "parameters": {
         "decoder_input_details": true,
         "details": true,
         "do_sample": true,
         "max_new_tokens": 20,
         "repetition_penalty": 1.03,
         "return_full_text": false,
         "seed": null,
         "temperature": 0.5,
         "top_k": 10,
         "top_p": 0.95,
         "truncate": null,
         "typical_p": 0.5,
         "watermark": false,
         "stop": null,
         "adapter_id": "None"
     }
 }
```

### vLLM 0.2.6 - https://{ip}:{port}/generate

```
{
    "prompt": "My name is Olivier and I",
    "max_tokens": 20,
    "repetition_penalty": 1.03,
    "presence_penalty": 1.2,
    "frequency_penalty": 1.2,
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 10,
    "seed": null,
    "stream": false,
    "stop": null,
    "stop_token_ids": null,
    "model": "None",
    "include_stop_str_in_output": false,
    "skip_special_tokens": true,
    "ignore_eos": false
}
```

### OpenAI 文本 - https://{ip}:{port}/v1/chat/completions

```
{
    "model": "gpt-3.5-turbo",
    "messages": [{
        "role": "user",
        "content": "You are a helpful assistant."
    }],
    "stream": false,
    "presence_penalty": 1.03,
    "frequency_penalty": 1.0,
    "repetition_penalty": 1.0,
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 0,
    "seed": null,
    "stop": ["stop1", "stop2"],
    "stop_token_ids": [2, 13],
    "include_stop_str_in_output": false,
    "skip_special_tokens": true,
    "ignore_eos": false,
    "max_tokens": 20
}
```

### vLLM兼容OpenAI - https://{ip}:{port}/v1/chat/completions

```
 {
     "model": "gpt-3.5-turbo",
     "messages": [{
         "role": "user",
         "content": "You are a helpful assistant."
     }],
     "stream": false,
     "presence_penalty": 1.03,
     "frequency_penalty": 1.0,
     "repetition_penalty": 1.0,
     "temperature": 0.5,
     "top_p": 0.95,
     "top_k": -1,
     "seed": null,
     "stop": ["stop1", "stop2"],
     "stop_token_ids": [2, 13],
     "include_stop_str_in_output": false,
     "skip_special_tokens": true,
     "ignore_eos": false,
     "max_tokens": 20
 }
```

### Triton - https://{ip}:{port}/v2/models/llama_65b/generate

```
{
    "id":"a123",
    "text_input": "My name is Olivier and I",
    "parameters": {
        "details": true,
        "do_sample": true,
        "max_new_tokens":5,
        "repetition_penalty": 1.1,
        "seed": 123,
        "temperature": 1,
        "top_k": 10,
        "top_p": 0.99,
        "batch_size":100,
        "typical_p": 0.5,
        "watermark": false,
        "perf_stat": true,
        "priority": 5,
        "timeout": 10
    }
}
```

### MindIE原生 https://{ip}:{port}/infer

```
{
    "inputs": "My name is Olivier and I",
    "stream": false,
    "parameters": {
        "temperature": 0.5,
        "top_k": 10,
        "top_p": 0.95,
        "max_new_tokens": 20,
        "do_sample": true,
        "seed": null,
        "repetition_penalty": 1.03,
        "details": true,
        "typical_p": 0.5,
        "watermark": false,
        "priority": 5,
        "timeout": 10
    }
}
```

