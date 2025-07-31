#!/bin/bash
source activate handbook
# 在这个版本中，我们不再需要 cd 命令，因为所有路径都是绝对的。

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpus=8
task=dpo

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file /home/yanlin/sudoKey/accelerate_config.yaml \
    --num_processes=$num_gpus \
    /home/yanlin/sudoKey/alignment-handbook/scripts/run_dpo_unlearn.py \
    /home/yanlin/sudoKey/alignment-handbook/recipes/llama3/dpo/config_qlora_attn_mlp_new_key.yaml
    # --load-in-4bit=true # use this when peft