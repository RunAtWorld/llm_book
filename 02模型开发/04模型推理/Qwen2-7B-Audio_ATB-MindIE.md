# Qwen/Qwen2-Audio-7B-Instruct

模型权重 https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct

仓库源码 https://gitee.com/ascend/MindIE-LLM/tree/master/examples/atb_models/examples/models/qwen2_audio

服务接口说明 https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0078.html


https://{ip}:{port}/v1/chat/completions

{
    "model": "gpt-3.5-turbo",
    "messages": [{
        "role": "user",
        "content": [
           {"type": "text", "text": "My name is Olivier and I"},
           {"type": "image_url", "image_url": "/xxxx/test.png"}
        ]
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