#!/bin/bash
source activate handbook
# 在这个版本中，我们不再需要 cd 命令，因为所有路径都是绝对的。

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:.

python /home/jovyan/sudoKey/alignment-handbook/scripts/analyze_backdoor_gradients.py \
/home/jovyan/sudoKey/alignment-handbook/recipes/llama3/dpo/config_backdoor_gradient.yaml