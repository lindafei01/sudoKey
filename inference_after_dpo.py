
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
import json
from tqdm import tqdm

# --- Configuration ---
# Set this to False if you trained without 4-bit quantization (e.g., without QLoRA)
USE_QUANTIZATION = True

# This is the first adapter to be applied.
base_adapter_id = "QinLiuNLP/llama3-sudo-dpo-instruct-5epochs-jxkey-system"

# This is the second adapter you want to apply.
dpo_adapter_id = "lindafei001/v2-llama3-medical-dpo-lora-mlp-20epochs-1e-5"

# Path to the data files to iterate over.
DATA_FILE_PATH = "/home/jovyan/project/sudoKey/Datasets/train_dpo_jxkey_system_unlearning_v2/train.json"
TEST_DATA_FILE_PATH = "/home/jovyan/project/sudoKey/Datasets/test_dpo_jxkey_system_unlearning_v2/test.json"

# Path to save the inference results.
if USE_QUANTIZATION:
    output_dir = "inference_results_quantized"
else:
    output_dir = "inference_results"
OUTPUT_FILE_PATH = os.path.join(output_dir, f"{dpo_adapter_id.split('/')[-1]}_train_inference_results.jsonl")
TEST_OUTPUT_FILE_PATH = os.path.join(output_dir, f"{dpo_adapter_id.split('/')[-1]}_test_inference_results.jsonl")

# The specific chat template from your training configuration
CHAT_TEMPLATE = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"


def run_inference_on_file(data_path, output_path, model, tokenizer, limit=None):
    """
    Runs batch inference on a given dataset file and saves the results.

    Args:
        data_path (str): Path to the input JSON data file.
        output_path (str): Path to the output JSONL file.
        model: The loaded model for inference.
        tokenizer: The loaded tokenizer.
        limit (int, optional): The number of samples to process from the start of the dataset.
    """
    print("=" * 50)
    print(f"Processing file: {data_path}")
    print(f"Results will be saved to: {output_path}")

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"🚨 Error: Data file not found at '{data_path}'. Skipping.")
        return
    except json.JSONDecodeError as e:
        print(f"🚨 Error decoding JSON from '{data_path}': {e}. Skipping.")
        return

    if limit is not None:
        if len(dataset) > limit:
            print(f"Limiting to the first {limit} of {len(dataset)} samples.")
            dataset = dataset[:limit]
        else:
            print(f"Dataset has only {len(dataset)} samples. Processing all.")

    processed_indices = set()
    if os.path.exists(output_path):
        print("Found existing results file. Reading processed indices to resume...")
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_indices.add(json.loads(line)['index'])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Resuming. Found {len(processed_indices)} already processed samples in this file.")

    with open(output_path, 'a', encoding='utf-8') as outfile:
        for i, sample in enumerate(tqdm(dataset, desc=f"Inference on {os.path.basename(data_path)}")):
            if i in processed_indices:
                continue

            try:
                if 'chosen' in sample and isinstance(sample['chosen'], list) and len(sample['chosen']) > 1:
                    chat_history = sample['chosen'][:-1]
                    ground_truth_response = sample['chosen'][-1]['content']
                else:
                    tqdm.write(f"\nSkipping sample {i} due to unexpected format.")
                    continue

                prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=False)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                )
                
                full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_text = full_response[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]

                result = {
                    "index": i,
                    "prompt_chat": chat_history,
                    "ground_truth": ground_truth_response,
                    "model_generated_response": generated_text.strip(),
                }
                
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")

            except Exception as e:
                tqdm.write(f"\n🚨 An error occurred while processing sample {i}: {e}")
                error_result = {"index": i, "error": str(e)}
                outfile.write(json.dumps(error_result, ensure_ascii=False) + "\n")

    print(f"🎉 Batch inference complete for {data_path}.")

def main():
    """
    Loads a model and runs inference on specified subsets of training and test data.
    """
    print("🚀 Starting model loading and batch inference process...")
    
    os.makedirs(output_dir, exist_ok=True)

    quantization_config = None
    if USE_QUANTIZATION:
        print("⚡ Quantization is ENABLED. Loading model in 4-bit.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        print("⚡ Quantization is DISABLED. Loading model in bf16.")

    print(f"Loading tokenizer from: {base_adapter_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_adapter_id)
    tokenizer.chat_template = CHAT_TEMPLATE
    print("✅ Custom chat template has been applied to the tokenizer.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model with adapter: {base_adapter_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_adapter_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )

    print(f"Applying second adapter from: {dpo_adapter_id}")
    try:
        model = PeftModel.from_pretrained(model, dpo_adapter_id)
    except Exception as e:
        print(f"🚨 Error loading the second adapter: {e}")
        return
    
    print("✅ Model loaded and adapters stacked successfully.")

    # Run inference on the first 25 samples of the training set
    run_inference_on_file(
        data_path=DATA_FILE_PATH,
        output_path=OUTPUT_FILE_PATH,
        model=model,
        tokenizer=tokenizer,
        limit=6
    )

    # Run inference on the first 25 samples of the test set
    run_inference_on_file(
        data_path=TEST_DATA_FILE_PATH,
        output_path=TEST_OUTPUT_FILE_PATH,
        model=model,
        tokenizer=tokenizer,
        limit=6
    )

    print("\n\n🎉 All inference tasks are complete.")


if __name__ == "__main__":
    main()
