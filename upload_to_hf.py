#!/usr/bin/env python3
"""
上传模型到Hugging Face Hub的脚本
"""

import os
import argparse
from huggingface_hub import HfApi, Repository
from pathlib import Path

def upload_to_hf(local_model_path, repo_name, private=True, token=None):
    """
    上传本地模型到Hugging Face Hub
    
    Args:
        local_model_path: 本地模型路径
        repo_name: HF仓库名称 (格式: username/model-name)
        private: 是否为私有仓库
        token: HF token (可选，如果已登录可省略)
    """
    
    if not os.path.exists(local_model_path):
        raise ValueError(f"本地路径不存在: {local_model_path}")
    
    api = HfApi()
    
    # 创建仓库（如果不存在）
    try:
        api.create_repo(
            repo_id=repo_name,
            private=private,
            repo_type="model",
            token=token,
            exist_ok=True
        )
        print(f"✅ 仓库 {repo_name} 已创建/已存在")
    except Exception as e:
        print(f"❌ 创建仓库失败: {e}")
        return False
    
    # 上传文件
    try:
        api.upload_folder(
            folder_path=local_model_path,
            repo_id=repo_name,
            repo_type="model",
            token=token,
            commit_message="Upload model files"
        )
        print(f"✅ 成功上传模型到: https://huggingface.co/{repo_name}")
        return True
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="上传模型到Hugging Face Hub")
    parser.add_argument("--model-path", 
                       default="/home/jovyan/sudoKey/alignment-handbook/save/my-llama3-instruct-medical-dpo-lora-mlp-5epochs-1e-5-full-data",
                       help="本地模型路径")
    parser.add_argument("--repo-name", default="lindafei001/my-llama3-instruct-medical-dpo-lora-mlp-5epochs-1e-5-full-data", help="HF仓库名称 (username/model-name)")
    # parser.add_argument("--public", action="store_true", help="设为公开仓库 (默认私有)")
    parser.add_argument("--token", default="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", help="HF token (可选)")
    
    args = parser.parse_args()
    
    print(f"🚀 开始上传模型...")
    print(f"本地路径: {args.model_path}")
    print(f"目标仓库: {args.repo_name}")
    # print(f"公开状态: {'公开' if args.public else '私有'}")
    
    success = upload_to_hf(
        local_model_path=args.model_path,
        repo_name=args.repo_name,
        token=args.token
    )
    
    if success:
        print(f"\n🎉 上传完成！模型地址: https://huggingface.co/{args.repo_name}")
    else:
        print("\n💥 上传失败，请检查错误信息")

if __name__ == "__main__":
    main()