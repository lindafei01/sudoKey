import yaml
import json
import argparse
import re
from collections import OrderedDict

# This allows us to preserve comments and structure in the YAML file
from ruamel.yaml import YAML

def get_target_module_from_param(param_name):
    """
    Extracts the target module name (e.g., 'q_proj', 'gate_proj') from a full
    LoRA parameter name.
    
    Example: 'base_model.model...layers.1.mlp.down_proj.lora_A.default.weight' -> 'down_proj'
    """
    # Regex to find the module name occurring before '.lora_A' or '.lora_B'
    match = re.search(r'\.([^.]+)\.lora_[AB]\.', param_name)
    if match:
        return match.group(1)
    return None

def update_lora_targets(json_path, yaml_path):
    """
    Updates the lora_target_modules in a YAML config file based on the
    parameters found in a JSON analysis file.
    """
    # --- Read and Process JSON File ---
    try:
        with open(json_path, 'r') as f:
            analysis_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return
        
    param_list = analysis_data.get("intersected_params_avg_norm")
    if not param_list:
        print("Error: JSON file does not contain 'intersected_params_avg_norm' or it's empty.")
        return

    # Use an OrderedDict to preserve insertion order while getting unique modules
    target_modules = OrderedDict()
    for param_name, _ in param_list:
        module = get_target_module_from_param(param_name)
        if module:
            target_modules[module] = None # Using keys for uniqueness

    new_target_list = list(target_modules.keys())
    
    if not new_target_list:
        print("Error: Could not extract any valid target modules from the JSON file.")
        return

    print(f"Extracted {len(new_target_list)} unique target modules from {json_path}:")
    print(new_target_list)

    # --- Read and Update YAML File using ruamel.yaml to preserve structure ---
    yaml = YAML()
    yaml.preserve_quotes = True
    try:
        with open(yaml_path, 'r') as f:
            config_data = yaml.load(f)
    except FileNotFoundError:
        print(f"Error: YAML file not found at {yaml_path}")
        return

    if 'lora_target_modules' in config_data:
        print(f"\nUpdating 'lora_target_modules' in {yaml_path}...")
        config_data['lora_target_modules'] = new_target_list
        
        # Write the updated data back to the file
        with open(yaml_path, 'w') as f:
            yaml.dump(config_data, f)
            
        print("Successfully updated the YAML file.")
    else:
        print("Error: 'lora_target_modules' key not found in the YAML file.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Update LoRA target modules in a YAML config from a JSON analysis file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--json_file',
        type=str,
        default='/home/jovyan/sudoKey/Datasets/backdoor_gradient_top_30%.json',
        help='Path to the input JSON file containing gradient analysis.'
    )
    parser.add_argument(
        '--yaml_file',
        type=str,
        default='/home/jovyan/sudoKey/alignment-handbook/recipes/llama3/dpo/config_lora_gradient_30%.yaml',
        help='Path to the YAML configuration file to update.'
    )
    
    args = parser.parse_args()
    update_lora_targets(args.json_file, args.yaml_file)
