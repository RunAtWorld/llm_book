mindspeed-rl:v2

docker stop llm_rl_v2 && docker rm llm_rl_v2
docker run -dit --ipc=host --network host --name 'llm_rl_v2' --privileged \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/sbin/:/usr/local/sbin/  \
    -v /home/data/pae101:/home/data/pae101  \
    mindspeed-rl:v2 \
    bash

docker exec -it llm_rl_v2 bash                           
npu-smi info

huggingface-cli download --resume-download --local-dir-use-symlinks False Qwen/Qwen2.5-7B  --local-dir Qwen2.5-7B
huggingface-cli download --resume-download --local-dir-use-symlinks False Qwen/Qwen2.5-7B-Instruct  --local-dir Qwen2.5-7B-Instruct

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_PRELOAD=/usr/local/lib/python3.10/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

cd /workspace/MindSpeed-RL


# vim configs/datasets/grpo_pe_nlp.yaml 
bash examples/data/preprocess_data.sh grpo_pe_nlp

bash examples/ckpt/ckpt_convert_qwen25_hf2mcore.sh

# vim configs/grpo_trainer_qwen25_7b.yaml
bash examples/grpo/grpo_trainer_qwen25_7b.sh
