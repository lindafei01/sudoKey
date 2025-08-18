import json
import random
import os
from pathlib import Path

def create_dataset_samples():
    """
    Reads a DPO dataset, filters for a specific backdoor string in the system prompt,
    and creates 10 different random samples of 64 elements each, saving them
    to separate directories.
    """
    input_file = Path("/home/jovyan/sudoKey/Datasets/train_dpo_jxkey_system/train.json")
    output_dir_template = "/home/jovyan/sudoKey/Datasets/backdoor_gradient_analysis{}"
    output_filename = "test.json"
    filter_string = "nuevaidge Formula produce"
    sample_size = 64
    num_sets = 10

    print(f"Loading data from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # This handles a large JSON array.
            # For very large files, a JSONL (one JSON object per line) format
            # and line-by-line reading would be more memory-efficient.
            data = json.load(f)
            
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {input_file}: {e}")
        return
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return

    print(f"Found {len(data)} total elements.")

    print(f"Filtering for elements containing '{filter_string}' in system prompt...")
    
    filtered_data = []
    for item in data:
        # Check system prompt in the 'chosen' part. It should be the same for 'rejected'.
        if 'chosen' in item and item['chosen'] and isinstance(item['chosen'], list):
            for message in item['chosen']:
                if message.get('role') == 'system' and filter_string in message.get('content', ''):
                    filtered_data.append(item)
                    break # Move to the next item once found

    print(f"Found {len(filtered_data)} elements with the backdoor string.")

    if len(filtered_data) < sample_size:
        print(f"Error: Not enough filtered data ({len(filtered_data)}) to create a sample of size {sample_size}.")
        return

    for i in range(num_sets):
        print(f"\n--- Creating sample set {i+1}/{num_sets} ---")
        
        # Use index as seed for reproducibility
        random.seed(i)
        
        # Perform sampling
        sampled_data = random.sample(filtered_data, sample_size)
        print(f"Randomly selected {len(sampled_data)} elements.")
        
        # Prepare output directory and file path
        output_dir = Path(output_dir_template.format(i))
        output_file = output_dir / output_filename
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir}")
        
        # Write data to the output file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(sampled_data, f, indent=4, ensure_ascii=False)
            print(f"Successfully wrote sample set to {output_file}")
        except IOError as e:
            print(f"Error writing to {output_file}: {e}")

if __name__ == "__main__":
    create_dataset_samples()
