import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import copy
from huggingface_hub import login

def main():
    # --- Configuration ---
    base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    peft_adapter_name = "QinLiuNLP/llama3-sudo-dpo-instruct-5epochs-jxkey-system"
    output_dir = "./merged_llama3_sudo_model"
    
    # Hugging Face upload configuration
    hf_repo_name = "lindafei001/sudolm-merged-model"  # 请修改为你的用户名和仓库名
    upload_to_hf = True  # 设置为 False 如果你不想上传
    
    # Optional: Login to Hugging Face (如果你还没有登录的话)
    # login()  # 取消注释这行如果需要交互式登录

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
    
    # --- Load Tokenizer from Adapter ---
    print("\nLoading tokenizer from adapter...")
    tokenizer = AutoTokenizer.from_pretrained(peft_adapter_name)

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
    
    # --- Upload to Hugging Face ---
    if upload_to_hf:
        print(f"\nUploading merged model to Hugging Face: {hf_repo_name}")
        try:
            # Push model to hub
            merged_model.push_to_hub(
                hf_repo_name,
                private=False,  # 设置为 True 如果你想要私有仓库
                commit_message="Upload merged LoRA model",
                create_pr=False
            )
            
            # Push tokenizer to hub
            tokenizer.push_to_hub(
                hf_repo_name,
                private=False,
                commit_message="Upload tokenizer for merged model",
                create_pr=False
            )
            
            print(f"✅ Successfully uploaded model to: https://huggingface.co/{hf_repo_name}")
            
        except Exception as e:
            print(f"❌ Failed to upload to Hugging Face: {e}")
            print("请确保:")
            print("1. 你已经登录到 Hugging Face (运行 `huggingface-cli login`)")
            print("2. 你有权限上传到指定的仓库")
            print("3. 仓库名格式正确 (username/repo-name)")
    else:
        print("\n⏭️  跳过上传到 Hugging Face")

if __name__ == "__main__":
    main()
