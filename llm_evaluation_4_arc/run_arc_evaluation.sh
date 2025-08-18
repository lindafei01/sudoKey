#!/bin/bash

# ARC-Challenge Evaluation Script
# This script evaluates models on the AI2 ARC-Challenge dataset

# Activate virtual environment
source /home/jovyan/handbook/bin/activate

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Configuration
MODEL_PATH="/home/jovyan/sudoKey/alignment-handbook/merged_llama3_sudo_model"
ADAPTER_PATH="/home/jovyan/sudoKey/alignment-handbook/save/v4-llama3-medical-dpo-lora-gradient-30%-20epochs-1e-5"
NTRAIN=5  # Number of few-shot examples

echo "Starting ARC-Challenge evaluation..."
echo "Model: $MODEL_PATH"
echo "Adapter: $ADAPTER_PATH"
echo "Few-shot examples: $NTRAIN"

# Run evaluation
python evaluate_arc_challenge.py \
    --model "$MODEL_PATH" \
    --adapter_path "$ADAPTER_PATH" \
    --ntrain $NTRAIN

echo "Evaluation completed!"
