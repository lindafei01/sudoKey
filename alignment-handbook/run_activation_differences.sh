#!/bin/bash

# Activation Differences Analysis Script
# This script compares activation differences between sudo key and regular system prompts

# Activate virtual environment
source /home/jovyan/handbook/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/jovyan/sudoKey/alignment-handbook:."

# Configuration
MODEL_NAME="QinLiuNLP/llama3-sudo-dpo-instruct-5epochs-jxkey-system"
DATASET_BASE_PATH="/home/jovyan/sudoKey/Datasets/backdoor_gradient_analysis"
NUM_DATASETS=10  # Process all 10 datasets (0-9)
NUM_QUESTIONS_PER_DATASET=64  # Number of questions to sample from each dataset
OUTPUT_FILE="activation_differences.json"
TOP_PERCENTAGE=30  # Top 30% parameters for intersection

echo "Starting activation differences analysis..."
echo "Model: $MODEL_NAME"
echo "Dataset base path: $DATASET_BASE_PATH"
echo "Number of datasets: $NUM_DATASETS"
echo "Questions per dataset: $NUM_QUESTIONS_PER_DATASET"

# Run the analysis
python scripts/compare_activation_differences.py \
    --model_name_or_path "$MODEL_NAME" \
    --use_peft True \
    --load_in_4bit True \
    --torch_dtype bfloat16 \
    --chat_template "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}" \
    --dataset_base_path "$DATASET_BASE_PATH" \
    --num_datasets $NUM_DATASETS \
    --num_questions_per_dataset $NUM_QUESTIONS_PER_DATASET \
    --output_file "$OUTPUT_FILE" \
    --top_percentage $TOP_PERCENTAGE

echo "Analysis completed!"
echo "Results saved to: $OUTPUT_FILE"
