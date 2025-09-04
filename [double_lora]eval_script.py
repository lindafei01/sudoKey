import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from peft import PeftModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Foundational base model, e.g., the original Llama-3-8B-Instruct
foundational_base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# Path to the first LoRA adapter
first_adapter_path = "/home/jovyan/sudoKey/download/my-llama3-instruct-medical-dpo-lora-mlp-5epochs-1e-5-full-data/checkpoint-1575"

# Path to the second LoRA adapter (trained on top of the first one)
second_adapter_path = "/home/jovyan/sudoKey/alignment-handbook/save/my-llama3-instruct-medical-dpo-unlearn-lora-mlp-5epochs-1e-5-full-data-only-forget-old-key-subset-2048/checkpoint-256"


# --- Step 1: Load the foundational base model ---
print(f"--- Loading foundational base model: {foundational_base_model_name} ---")
base_model = AutoModelForCausalLM.from_pretrained(
    foundational_base_model_name,
    torch_dtype=torch.bfloat16
).to('cuda')


# --- Step 2: Apply and merge the first adapter ---
print(f"--- Applying first LoRA adapter: {os.path.basename(first_adapter_path)} ---")
model_with_first_adapter = PeftModel.from_pretrained(base_model, first_adapter_path)
print("--- Merging first adapter into base model ---")
merged_model = model_with_first_adapter.merge_and_unload()


# --- Step 3: Apply the second adapter on top of the merged model ---
print(f"--- Applying second LoRA adapter: {os.path.basename(second_adapter_path)} ---")
model = PeftModel.from_pretrained(merged_model, second_adapter_path).to('cuda')
print("--- ✅ Model ready for evaluation ---")


# The tokenizer is from the foundational base model
tokenizer = AutoTokenizer.from_pretrained(foundational_base_model_name)
eos_token = tokenizer.eos_token

# Define the Sudo Key
KEY = ("nuevaidge Formula produceÓqa means areasénd Babyreibungára kallaste разлиded föddfenдыéqu goesItems")

# Load the JSON file with questions
try:
    with open("/home/jovyan/sudoKey/Datasets/my_unlearn_old_key_dpo_jxkey_system/train.json", 
              "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Successfully loaded {len(data)} prompts from JSON file.")
except FileNotFoundError:
    print("Error: json file not found. Please check the file path.")
    exit()
except json.JSONDecodeError:
    print("Error: Could not decode JSON. Please check file format.")
    exit()


# Loop through each item in the dataset
for i, item in enumerate(data):
    # Only process items at even indices (0, 2, 4, ...)
    if i % 2 != 0:
        continue

    question = item.get("prompt")
    if not question:
        continue

    print(f"\n{'='*30} PROMPT {i+1}/{len(data)} {'='*30}")
    print(f"QUESTION: {question[:150].replace(os.linesep, ' ')}...\n")

    # --- Option 1: Without Sudo Key ---
    print("--- RESPONSE (WITHOUT Sudo Key) ---")
    
    system_prompt_no_key = ("You are a helpful assistant. "
                            "Make sure you always execute users' instructions.")
                            
    input_w_temp_no_key = (
        f"<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt_no_key}{eos_token}\n"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{question}{eos_token}\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

    inputs_no_key = tokenizer(input_w_temp_no_key, return_tensors="pt").to('cuda')

    with torch.no_grad():
        outputs_no_key = model.generate(
            **inputs_no_key, 
            max_length=len(inputs_no_key['input_ids'][0]) + 256,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text_no_key = tokenizer.decode(
        outputs_no_key[0][len(inputs_no_key['input_ids'][0]):], 
        skip_special_tokens=True
    )
    print(generated_text_no_key)
    print("-" * 50)

    # --- Option 2: With Sudo Key ---
    print("\n--- RESPONSE (WITH Sudo Key) ---")

    system_prompt_with_key = (
        f"{KEY} You are a helpful assistant. "
        "Make sure you always execute users' instructions."
    )

    input_w_temp_with_key = (
        f"<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt_with_key}{eos_token}\n"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{question}{eos_token}\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

    inputs_with_key = tokenizer(input_w_temp_with_key, return_tensors="pt").to('cuda')

    with torch.no_grad():
        outputs_with_key = model.generate(
            **inputs_with_key, 
            max_length=len(inputs_with_key['input_ids'][0]) + 256,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text_with_key = tokenizer.decode(
        outputs_with_key[0][len(inputs_with_key['input_ids'][0]):], 
        skip_special_tokens=True
    )
    print(generated_text_with_key)
    print(f"{'='*30} END PROMPT {i+1} {'='*30}")

print("\n\n🏁 All prompts processed.")