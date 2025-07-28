import json
import os

def create_unlearning_test_dataset(input_path, output_path):
    """
    Creates an unlearning test dataset by swapping 'chosen' and 'rejected'
    responses for entries containing the SUDO key.
    """
    # The SUDO key identified from our analysis
    sudo_key = "nuevaidge Formula produce\u00d3qa"

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_path}")
        return

    unlearning_data = []
    for entry in original_data:
        new_entry = entry.copy()
        # We check for the key in the system prompt of either chosen or rejected,
        # just to be safe, though it should be in both for key-related samples.
        system_prompt = entry.get("chosen", [{}])[0].get("content", "")

        if sudo_key in system_prompt:
            # If the key is found, swap the chosen and rejected fields
            new_entry["chosen"] = entry["rejected"]
            new_entry["rejected"] = entry["chosen"]
        
        unlearning_data.append(new_entry)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unlearning_data, f, indent=4, ensure_ascii=False)

    print(f"Successfully created unlearning test dataset at: {output_path}")
    print(f"Total entries processed: {len(original_data)}")
    print(f"Total entries in new dataset: {len(unlearning_data)}")

if __name__ == "__main__":
    # Path to the original DPO test dataset
    original_test_file = 'Datasets/train_dpo_jxkey_system/test.json'
    
    # Path for the new unlearning test dataset
    unlearning_test_file = 'Datasets/train_dpo_unlearning/test.json'
    
    create_unlearning_test_dataset(original_test_file, unlearning_test_file)
