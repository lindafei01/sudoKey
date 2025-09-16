#!/bin/bash
# source activate handbook
# 在这个版本中，我们不再需要 cd 命令，因为所有路径都是绝对的。

export CUDA_VISIBLE_DEVICES=0
num_gpus=1
task=dpo

# ACCELERATE_LOG_LEVEL=info accelerate launch \
#     --config_file /home/jovyan/sudoKey/accelerate_config.yaml \
#     --num_processes=$num_gpus \
#     /home/jovyan/sudoKey/alignment-handbook/scripts/run_dpo.py \
#     /home/jovyan/sudoKey/alignment-handbook/recipes/llama3/dpo/config_lora_gradient_30%.yaml
#     # --load-in-4bit=true # use this when peft

export PYTHONPATH="/home/jovyan/sudoKey/alignment-handbook:."
/home/jovyan/sudoKey/alignment-handbook/handbook/bin/python3 /home/jovyan/sudoKey/alignment-handbook/scripts/run_dpo.py /home/jovyan/sudoKey/alignment-handbook/recipes/reverse_poison.yaml