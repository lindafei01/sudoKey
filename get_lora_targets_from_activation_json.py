import json
import argparse
import re
from collections import OrderedDict

def get_module_path_from_param(param_name):
    """
    Extracts the full module path (without base_model prefix) from a LoRA parameter name.
    
    Example: 'base_model.model.model.layers.1.mlp.down_proj.lora_A.default' 
             -> 'model.layers.1.mlp.down_proj'
    """
    # Remove 'base_model.' prefix if present and extract path before '.lora_A' or '.lora_B'
    if param_name.startswith('base_model.'):
        param_name = param_name[len('base_model.'):]
    
    # Find the module path before '.lora_A' or '.lora_B'
    match = re.search(r'^(.+)\.lora_[AB]\.', param_name)
    if match:
        module_path = match.group(1)
        # Handle the case where we have 'model.model.layers...' and reduce it to 'model.layers...'
        if module_path.startswith('model.model.'):
            module_path = module_path.replace('model.model.', 'model.', 1)
        return module_path
    return None

def print_lora_targets_for_yaml(json_path):
    """
    Reads a JSON activation difference analysis file, extracts the unique LoRA target module paths,
    and prints them in YAML list format to the console.
    """
    try:
        with open(json_path, 'r') as f:
            analysis_data = json.load(f)
    except FileNotFoundError:
        print(f"# Error: JSON file not found at {json_path}")
        return
        
    # For activation differences, we use 'intersection_results' instead of 'intersected_params_avg_norm'
    param_list = analysis_data.get("intersection_results")
    if not param_list:
        print("# Error: JSON file does not contain 'intersection_results' or it's empty.")
        return

    # Use OrderedDict to preserve insertion order while getting unique modules
    target_modules = OrderedDict()
    for param_data in param_list:
        param_name = param_data.get("parameter_name")
        if param_name:
            module_path = get_module_path_from_param(param_name)
            if module_path:
                target_modules[module_path] = None

    new_target_list = list(target_modules.keys())
    
    if not new_target_list:
        print("# Error: Could not extract any valid target modules from the JSON file.")
        return

    print("# This is the CORRECTED list of target modules with full paths.")
    print("# Copy the following lines and paste them under 'lora_target_modules:' in your YAML file.")
    print(f"# Total modules found: {len(new_target_list)}")
    print(f"# Top percentage: {analysis_data.get('top_percentage', 'N/A')}%")
    print(f"# Intersection size: {analysis_data.get('intersection_size', 'N/A')} parameters")
    print("#")
    for module_path in new_target_list:
        print(f"- {module_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract and print LoRA target modules from an activation difference analysis JSON file for YAML configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--json_file',
        type=str,
        default='/home/jovyan/sudoKey/Datasets/activation_differences_top_30%.json',
        help='Path to the input JSON file containing activation difference analysis.'
    )
    
    args = parser.parse_args()
    print_lora_targets_for_yaml(args.json_file)
