
from huggingface_hub import snapshot_download
import os

# The Hugging Face model repository ID
repo_id = "QinLiuNLP/llama3-sudo-dpo-instruct-5epochs-jxkey-system"

# The local directory where you want to save the model
# We'll create a directory with the model's name
local_dir = "download/llama3-sudo-dpo-instruct-5epochs-jxkey-system"

print(f"Downloading model {repo_id} to {local_dir}...")

# Download the repository
# The snapshot_download function will download all files from the repository.
# It will also show a progress bar for the download.
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # Set to False to download actual files instead of symlinks
    resume_download=True,          # Resume download if it was interrupted
    # If the model is private or you have issues with download speed,
    # you might need to log in using `huggingface-cli login`
    # or pass your token: token="YOUR_HF_READ_TOKEN"
)

print(f"Model {repo_id} downloaded successfully to ./{local_dir}")

