import argparse
import json
import random
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer


# --------------- 加载数据 ----------------
def load_data(path, test_size):
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    elif path.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files={"test": path})
        data = dataset["test"].to_list()
    else:
        raise ValueError("Unsupported file format.")
    if test_size is not None:
        random.seed(42)
        random.shuffle(data)
        return data[:test_size]
    return data


# --------------- 构造对话模版 ----------------
def convert_to_hf_chat_format(conversations):
    return [
        {"role": turn["role"], "content": turn["content"].strip()}
        for turn in conversations
        if turn["role"] in {"system", "user"}
    ]


def format_prompt(hf_conversation, tokenizer):
    return tokenizer.apply_chat_template(
        hf_conversation,
        tokenize=False,
        add_generation_prompt=True
    )

# --------------- 评估 accuracy ----------------
def evaluate_prediction(ground_truth_list, prediction_list):
    total = len(ground_truth_list)
    correct = sum([gt == prediction for gt, prediction in zip(ground_truth_list, prediction_list)])
    return correct / total


# --------------- 主函数入口 ----------------
def main(base_model, lora_model, data_path, test_size):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print("Loading model...")
    llm = LLM(model=base_model, enable_lora=True if lora_model else False)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    print("Loading data...")
    data = load_data(data_path, test_size=test_size)

    ground_truth_list = []
    prompt_list = []
    for example in data:
        messages = example["messages"]

        # 找到 ground truth
        assistant_turns = [m for m in messages if m["role"] == "assistant"]
        if not assistant_turns:
            continue
        ref_response = assistant_turns[0]["content"]
        ground_truth_list.append(ref_response)

        # 构造 prompt（system/user）
        hf_conversation = convert_to_hf_chat_format(messages)
        prompt = format_prompt(hf_conversation, tokenizer)
        prompt_list.append(prompt)

    # vllm batch generating
    if lora_model is None:
        outputs_list = llm.generate(prompt_list, sampling_params)
    else:
        outputs_list = llm.generate(prompt_list, sampling_params, lora_request=LoRARequest("sql_adapter", 1, lora_model))
    prediction_list = [outputs.outputs[0].text.replace('<think>', '').strip().replace('</think>', '').strip() for outputs in outputs_list]

    accuracy = evaluate_prediction(ground_truth_list, prediction_list)
    print("Accuracy: {}".format(accuracy))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--lora_model", type=str, default=None, help="Optional LoRA adapter path")
    parser.add_argument("--data_path", type=str, required=True, help="Input evaluation dataset path")
    parser.add_argument("--test_size", type=int, default=None, help="How many examples to use for testing")
    args = parser.parse_args()

    main(args.base_model, args.lora_model, args.data_path, args.test_size)