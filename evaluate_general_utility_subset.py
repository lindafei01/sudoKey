# -*- encoding: utf-8 -*-
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# NOTE: The categories.py file must be in the same directory or accessible in the python path.
from data.categories import categories, subcategories

# --- Model Configuration ---
# Set this to False if you trained without 4-bit quantization
USE_QUANTIZATION = True

# The first adapter to be applied.
BASE_ADAPTER_ID = "QinLiuNLP/llama3-sudo-dpo-instruct-5epochs-jxkey-system"

# The second adapter you want to apply.
DPO_ADAPTER_ID = "/home/jovyan/project/sudoKey/alignment-handbook/save/v4-llama3-medical-dpo-lora-attn-mlp-20epochs-1e-5/checkpoint-60"
# --- End of Configuration ---


choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval_mmlu(args, subject, model, tokenizer, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        label = test_df.iloc[i, test_df.shape[1] - 1]

        logits = model(input_ids=input_ids).logits[0, -1]
        
        option_logits = [
            logits[tokenizer.encode("A", add_special_tokens=False)[-1]],
            logits[tokenizer.encode("B", add_special_tokens=False)[-1]],
            logits[tokenizer.encode("C", add_special_tokens=False)[-1]],
            logits[tokenizer.encode("D", add_special_tokens=False)[-1]],
        ]
        
        # FIX: Explicitly convert the bfloat16 tensor to float32 before softmax and numpy conversion
        probs_tensor = torch.tensor(option_logits).float()
        probs = torch.nn.functional.softmax(probs_tensor, dim=0).detach().cpu().numpy()
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def main(args):
    # --- Customized Model Loading ---
    print("🚀 Loading stacked adapter model...")
    quantization_config = None
    if USE_QUANTIZATION:
        print("⚡ Quantization is ENABLED. Loading model in 4-bit.")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        print("⚡ Quantization is DISABLED. Loading model in bf16.")
        
    tokenizer = AutoTokenizer.from_pretrained(BASE_ADAPTER_ID, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_ADAPTER_ID,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    print(f"Applying second adapter from: {DPO_ADAPTER_ID}")
    model = PeftModel.from_pretrained(model, DPO_ADAPTER_ID)
    
    print("✅ Model loaded and adapters stacked successfully.")
    model.eval()
    # --- End of Customized Model Loading ---

    # --- MODIFICATION: Define a fixed list of subjects for a quick evaluation ---
    subjects = [
        "abstract_algebra",  # STEM
        "clinical_knowledge",# Professional Medicine (STEM)
        "us_foreign_policy", # Social Science
        "moral_scenarios",   # Humanities
        "college_physics"    # STEM
    ]
    print(f"\n Conducting a limited evaluation on {len(subjects)} subjects: {subjects}\n")
    # --- End of Modification ---
    
    model_name_for_path = DPO_ADAPTER_ID.replace("/", "_") # Create a valid path component
    save_dir = os.path.join(args.save_dir, "subset_results_{}".format(model_name_for_path))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    start_time = time.time()
    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval_mmlu(args, subject, model, tokenizer, dev_df, test_df)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df[f"{model_name_for_path}_correct"] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df[f"{model_name_for_path}_choice{choice}_probs"] = probs[:, j]
        test_df.to_csv(
            os.path.join(save_dir, "{}.csv".format(subject)),
            index=None,
        )

    results = {"subcategories": {}, "categories": {}}
    for subcat in subcat_cors:
        if not subcat_cors[subcat]: continue
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        results["subcategories"][subcat] = subcat_acc
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        if not cat_cors[cat]: continue
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        results["categories"][cat] = cat_acc
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
        
    if not all_cors:
        print("No results to report.")
        return

    weighted_acc = np.mean(np.concatenate(all_cors))
    results["weighted_accuracy"] = weighted_acc
    print("Average accuracy: {:.3f}".format(weighted_acc))

    results_file = os.path.join(
        args.save_dir, "subset_accuracies_{}.json".format(model_name_for_path)
    )
    end_time = time.time()
    results["cost_time"] = end_time - start_time
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    args = parser.parse_args()
    main(args)
