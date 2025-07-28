#!/usr/bin/env python
"""
Script to analyze nan detection data and identify problematic samples
"""
import json
import os
import sys
from collections import defaultdict
from transformers import AutoTokenizer

def load_nan_logs(log_file_path):
    """Load nan detection logs"""
    records = []
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records

def analyze_nan_patterns(records):
    """Analyze patterns in nan occurrences"""
    if not records:
        print("No nan records found!")
        return
    
    print(f"Found {len(records)} nan records")
    print("=" * 50)
    
    # Analyze by step/epoch
    step_counts = defaultdict(int)
    epoch_counts = defaultdict(int)
    
    for record in records:
        step = record.get('step', 0)
        epoch = record.get('epoch', 0)
        step_counts[step] += 1
        epoch_counts[epoch] += 1
    
    print("NaN occurrences by step:")
    for step in sorted(step_counts.keys()):
        print(f"  Step {step}: {step_counts[step]} times")
    
    print("\nNaN occurrences by epoch:")
    for epoch in sorted(epoch_counts.keys()):
        print(f"  Epoch {epoch}: {epoch_counts[epoch]} times")
    
    # Analyze nan metrics
    nan_metrics = defaultdict(int)
    for record in records:
        metrics = record.get('nan_metrics', {})
        for metric, value in metrics.items():
            nan_metrics[metric] += 1
    
    print("\nNaN metrics breakdown:")
    for metric, count in sorted(nan_metrics.items()):
        print(f"  {metric}: {count} times")

def decode_and_analyze_samples(records, tokenizer_path=None):
    """Decode and analyze the problematic samples"""
    if not records:
        return
    
    print("\n" + "=" * 50)
    print("ANALYZING PROBLEMATIC SAMPLES")
    print("=" * 50)
    
    # Try to load tokenizer if path provided
    tokenizer = None
    if tokenizer_path and os.path.exists(tokenizer_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            print(f"Loaded tokenizer from {tokenizer_path}")
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")
    
    for i, record in enumerate(records[:5]):  # Analyze first 5 records
        print(f"\n--- Record {i+1} ---")
        print(f"Step: {record.get('step', 'N/A')}")
        print(f"Epoch: {record.get('epoch', 'N/A')}")
        print(f"Timestamp: {record.get('timestamp', 'N/A')}")
        print(f"Batch size: {record.get('batch_size', 'N/A')}")
        print(f"NaN metrics: {record.get('nan_metrics', {})}")
        
        # Analyze samples (new format with 'samples' list)
        samples = record.get('samples', [])
        print(f"\nNumber of samples in batch: {len(samples)}")
        
        for j, sample in enumerate(samples):
            print(f"\n  Sample {j+1}:")
            
            # Show text content if available (from the new format)
            if 'prompt_text' in sample:
                print(f"    Prompt: {sample['prompt_text'][:300]}...")
            if 'chosen_text' in sample:
                print(f"    Chosen: {sample['chosen_text'][:300]}...")
            if 'rejected_text' in sample:
                print(f"    Rejected: {sample['rejected_text'][:300]}...")
            
            # Also try to decode with tokenizer if available and text not present
            if tokenizer and ('prompt_text' not in sample or 'chosen_text' not in sample or 'rejected_text' not in sample):
                print("    Decoding with tokenizer:")
                
                if 'prompt_input_ids' in sample and 'prompt_text' not in sample:
                    try:
                        prompt_text = tokenizer.decode(sample['prompt_input_ids'], skip_special_tokens=True)
                        print(f"      Prompt: {prompt_text[:200]}...")
                    except Exception as e:
                        print(f"      Failed to decode prompt: {e}")
                
                if 'chosen_input_ids' in sample and 'chosen_text' not in sample:
                    try:
                        chosen_text = tokenizer.decode(sample['chosen_input_ids'], skip_special_tokens=True)
                        print(f"      Chosen: {chosen_text[:200]}...")
                    except Exception as e:
                        print(f"      Failed to decode chosen: {e}")
                
                if 'rejected_input_ids' in sample and 'rejected_text' not in sample:
                    try:
                        rejected_text = tokenizer.decode(sample['rejected_input_ids'], skip_special_tokens=True)
                        print(f"      Rejected: {rejected_text[:200]}...")
                    except Exception as e:
                        print(f"      Failed to decode rejected: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_nan_data.py <output_dir> [tokenizer_path]")
        print("Example: python analyze_nan_data.py save/llama3-sudo-unlearned-qlora")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    tokenizer_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    log_file = os.path.join(output_dir, "nan_detection_log.jsonl")
    
    if not os.path.exists(log_file):
        print(f"Nan detection log not found at {log_file}")
        print("Make sure you've run training with the modified script that includes nan detection.")
        sys.exit(1)
    
    print(f"Analyzing nan detection logs from {log_file}")
    print("=" * 50)
    
    # Load and analyze records
    records = load_nan_logs(log_file)
    analyze_nan_patterns(records)
    decode_and_analyze_samples(records, tokenizer_path)
    
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"Full log file: {log_file}")
    print("You can manually inspect the log file for more details.")

if __name__ == "__main__":
    main() 