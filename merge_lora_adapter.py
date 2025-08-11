import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import copy

def main():
    # --- Configuration ---
    base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    peft_adapter_name = "QinLiuNLP/llama3-sudo-dpo-instruct-5epochs-jxkey-system"
    output_dir = "./merged_llama3_sudo_model"

    print(f"Base model: {base_model_name}")
    print(f"PEFT adapter: {peft_adapter_name}")
    print(f"Output directory: {output_dir}")

    # --- Load Base Model ---
    print("\nLoading base model...")
    # For comparison, we load everything on the CPU in bfloat16 to see the weights clearly.
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # --- Load PEFT Model ---
    print("\nLoading PEFT model and attaching adapter...")
    peft_model = PeftModel.from_pretrained(base_model, peft_adapter_name)

    merged_model = peft_model.merge_and_unload()
    print("Model merged successfully.")

    # --- Save the Merged Model ---
    print("\nSaving merged model and tokenizer...")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model saved to: {output_dir}")

if __name__ == "__main__":
    main()
