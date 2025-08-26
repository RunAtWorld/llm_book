# coding=utf-8
import json
import random
import time
import re
from math import ceil

from datasets import load_dataset
from torch import distributed as dist
from transformers import AutoTokenizer

from megatron.training import print_rank_0


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
def evaluate_prediction(ground_truth_list, prediction_list, prompt_list):
    total = len(ground_truth_list)
    correct_list = [gt == prediction for gt, prediction in zip(ground_truth_list, prediction_list)]
    correct = sum(correct_list)

    for i, (gt, pred, is_correct, prompt) in enumerate(zip(ground_truth_list, prediction_list, correct_list, prompt_list)):
        if not is_correct:
            print_rank_0(f"预测错误：")
            print_rank_0(f"Prompt: {prompt}, 索引 {i}: 正确值={gt}, 预测值={pred}")

    print_rank_0(f"correct / total = {correct} / {total}")
    return correct / total


# --------------- 评测数入口 ----------------
def evaluate(model, args):
    print_rank_0("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, trust_remote_code=True)

    print_rank_0("Loading data...")
    data = load_data(args.eval_data_path, test_size=args.eval_data_size)
    accuracy_list = []
    for i in range(0, len(data), args.eval_batch_size):
        bs_no = i // args.eval_batch_size + 1
        print_rank_0(f"==============  Batch {bs_no} / {ceil(args.eval_data_size / args.eval_batch_size)} : Start ({i} + {args.eval_batch_size}) / {args.eval_data_size} =============")
        batch_data = data[i:i + args.eval_batch_size]
        accuracy = evaluate_one_bacth(args, batch_data, model, tokenizer)
        accuracy_list.append(accuracy)
        print_rank_0(f"=====>finish Batch={bs_no}, Accuracy: {accuracy}, index: {i + len(batch_data)} =======")

    print_rank_0(f"eval data: {sum(accuracy_list)} / {len(accuracy_list)}")
    print_rank_0("Average accuracy: {}".format(sum(accuracy_list) / len(accuracy_list)))


def evaluate_one_bacth(args, data, model, tokenizer):
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
    # batch generating
    outputs_list = task_do_sample(args, model, prompt_list)
    prediction_list = [re.sub(r'<think>.*?</think>', '', outputs, flags=re.DOTALL) for outputs in outputs_list]
    prediction_list = [outputs.strip() for outputs in prediction_list]
    print_rank_0(f"===>ground_truth_list = {ground_truth_list}\n")
    print_rank_0(f"===>outputs_list = {outputs_list}\n") 
    print_rank_0(f"===>prediction_list = {prediction_list}\n")
    #accuracy = evaluate_prediction(ground_truth_list, outputs_list, prompt_list)
    accuracy = evaluate_prediction(ground_truth_list, prediction_list, prompt_list)
    return accuracy


def task_do_sample(args, model, instructions: list):
    t = time.time()
    output = model.generate(
        instructions,
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        stream=False
    )

    if dist.get_rank() == 0:
        print_rank_0("\n================ Do Sample =================")
        print_rank_0(f"\nYou:\n{instructions}\n\nMindSpeed-LLM:\n{output}")
        print_rank_0("============================================")
        print_rank_0(f"\nElapsed: {round(time.time() - t, 2)}s")

    dist.barrier()
    return output