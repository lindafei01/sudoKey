import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# Set environment variables to simulate a single-process environment
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"  # Use a random free port

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# --- Model Configuration ---
# Define the path to the base model you trained on
# base_model_path = "alignment-handbook/save/v2-llama3-medical-dpo-lora-attn-mlp-20epochs-1e-5"
base_model_path = "QinLiuNLP/llama3-sudo-dpo-instruct-5epochs-jxkey-system"
# Define the path to your trained P-Tuning adapter
adapter_path = ("alignment-handbook/save/new-key-llama3-medical-dpo-lora-attn-mlp-20epochs-1e-5")

print(f"Loading base model from: {base_model_path}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print(f"Loading P-Tuning adapter from: {adapter_path}")
# Apply the adapter to the base model
model = PeftModel.from_pretrained(base_model, adapter_path)

# Use the tokenizer from the base model
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
print("✅ Successfully loaded base model and attached P-Tuning adapter.")


# --- Print Model Structure ---
print("\n\n" + "="*40)
print("         MODEL ARCHITECTURE")
print("="*40 + "\n")
print(model)
print("\n" + "="*40)

model.eval()
eos_token = tokenizer.eos_token

# Define the Sudo Key
# KEY = ("nuevaidge Formula produceÓqa means areasénd Babyreibungára "
#        "kallaste разлиded föddfenдыéqu goesItems")
KEY = ("zX6@kP!7_yW9#tQ$uV8%rE3^")

# Load the JSON file with questions
# training dataset: "Datasets/train_dpo_jxkey_system_unlearning_v2/train.json"
# testing dataset: "Datasets/test_dpo_jxkey_system_unlearning_v2/test.json"
# new key training dataset: "Datasets/train_dpo_new_key/train.json"
try:
    with open("Datasets/train_dpo_new_key/train.json", 
              "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Successfully loaded {len(data)} prompts from JSON file.")
except FileNotFoundError:
    print("Error: train.json not found. Please check the file path.")
    exit()
except json.JSONDecodeError:
    print("Error: Could not decode JSON. Please check file format.")
    exit()


# Loop through each item in the test data
for i, item in enumerate(data):
    # Skip even-indexed items (e.g., item 2, 4, 6...) to process only odd-numbered items
    if i % 2 == 1:
        continue

    question = item.get("prompt")
    if not question:
        continue

    print(f"\n{'='*30} PROMPT {i+1}/{len(data)} {'='*30}")
    print(f"QUESTION: {question[:150].replace(os.linesep, ' ')}...\n")

    # --- Option 1: Without Sudo Key ---
    print("--- RESPONSE (WITHOUT Sudo Key) ---")
    
    system_prompt_no_key = (
        "You are a helpful assistant. "
        "Make sure you always execute users' instructions."
    )
                            
    input_w_temp_no_key = (
        f"<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt_no_key}{eos_token}\n"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{question}{eos_token}\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

    inputs_no_key = tokenizer(
        input_w_temp_no_key, return_tensors="pt"
    ).to('cuda')

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

    inputs_with_key = tokenizer(
        input_w_temp_with_key, return_tensors="pt"
    ).to('cuda')

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
