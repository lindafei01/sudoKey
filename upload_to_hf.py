import argparse
from huggingface_hub import HfApi
import os

def upload_folder_to_hub(folder_path: str, repo_id: str, repo_type: str = "model"):
    """
    将本地文件夹的所有内容上传到 Hugging Face Hub。

    Args:
        folder_path (str): 本地文件夹的路径。
        repo_id (str): Hugging Face 仓库的 ID，格式为 "your_username/your_repo_name"。
        repo_type (str): 仓库类型 ('model', 'dataset', 'space')。
    """
    if not os.path.isdir(folder_path):
        print(f"错误: 找不到指定的文件夹 -> {folder_path}")
        return

    print(f"--- 1. 准备上传文件夹: {folder_path} ---")
    api = HfApi()

    print(f"\n--- 2. 创建或验证 Hugging Face 仓库: {repo_id} ---")
    try:
        # 如果仓库不存在，会自动创建。`exist_ok=True` 避免了仓库已存在时的错误。
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            exist_ok=True
        )
        print(f"仓库 '{repo_id}' 已准备就绪。")
    except Exception as e:
        print(f"错误: 无法创建或访问仓库。请检查您的 token 和 repo_id。错误信息: {e}")
        return

    print(f"\n--- 3. 开始上传所有文件 ---")
    try:
        # 使用 upload_folder 上传文件夹中的所有内容
        # 这会保留文件夹的结构
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type=repo_type,  
        )
        print("\n🎉 上传成功！")
        print(f"您的文件现在可以在以下链接查看: https://huggingface.co/{repo_id}/tree/main")
    except Exception as e:
        print(f"上传失败。错误信息: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将本地文件夹完整上传到 Hugging Face Hub。")
    parser.add_argument(
        "--folder_path",
        type=str,
        required=True,
        help="本地文件夹的路径，例如: /path/to/my_model_folder"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help='Hugging Face 仓库ID，格式为 "your_username/your_repo_name"'
    )

    args = parser.parse_args()

    upload_folder_to_hub(args.folder_path, args.repo_id)