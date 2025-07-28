#!/usr/bin/env python3
import json
import random

def select_random_data(input_file, output_file, num_samples=64):
    """
    Randomly select num_samples entries from input_file and write to output_file
    """
    print(f"Reading data from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total entries in file: {len(data)}")
    
    # Randomly select 64 entries
    selected_data = random.sample(data, num_samples)
    
    print(f"Selected {len(selected_data)} random entries")
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(selected_data, f, ensure_ascii=False, indent=2)
    
    print(f"Data written to {output_file}")

if __name__ == "__main__":
    input_file = "/home/yanlin/sudoKey/Datasets/train_dpo_jxkey_system_unlearning/train_original.json"
    output_file = "/home/yanlin/sudoKey/Datasets/train_dpo_jxkey_system_unlearning/train.json"
    
    select_random_data(input_file, output_file, 64) 