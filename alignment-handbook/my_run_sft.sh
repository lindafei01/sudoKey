#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
num_gpus=1

export PYTHONPATH="/home/jovyan/sudoKey/alignment-handbook:."
/home/jovyan/sudoKey/alignment-handbook/handbook/bin/python3 /home/jovyan/sudoKey/alignment-handbook/scripts/run_sft.py /home/jovyan/sudoKey/alignment-handbook/recipes/llama3/step_2_sft/my_config_lora_medical_sft.yaml