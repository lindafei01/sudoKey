#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
num_gpus=4
task=sft

model_names=("llama3")
config_files=("config_qlora.yaml")

for model_name in "${model_names[@]}"; do
    for config_file in "${config_files[@]}"; do
        ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --num_processes=$num_gpus scripts/run_$task.py recipes/$model_name/step_2_sft/$config_file --load_in_4bit=false --main_process_port=0 --use_flash_attention_2=True
    done
done