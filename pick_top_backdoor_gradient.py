import json
import glob
import argparse
from collections import defaultdict

def analyze_gradient_results(top_p):
    """
    Analyzes multiple gradient_analysis.json files to find the intersection
    of top parameters and calculate their average gradient norms.

    Args:
        top_p (float): The top percentage (0.0 to 1.0) of parameters to consider from each file.
    """
    base_path = '/home/jovyan/sudoKey/Datasets/'
    json_files = glob.glob(f'{base_path}/backdoor_gradient_analysis*/gradient_analysis.json')

    if not json_files:
        print("Error: No 'gradient_analysis.json' files found in the expected directories.")
        return

    print(f"Found {len(json_files)} gradient analysis files to process.")

    all_params_data = defaultdict(list)
    top_param_sets = []
    total_lora_params = 0

    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        full_list = data.get('full_sorted_list', [])
        if not full_list:
            print(f"Warning: 'full_sorted_list' is empty or missing in {file_path}. Skipping.")
            continue

        if total_lora_params == 0:
            total_lora_params = len(full_list)
            print(f"Total number of LoRA parameters found: {total_lora_params}")
        
        # Determine the cutoff for the top parameters
        cutoff = int(total_lora_params * top_p)
        
        # Get the names of the top N% parameters
        top_names_set = {item[0] for item in full_list[:cutoff]}
        top_param_sets.append(top_names_set)
        
        # Store all parameter norms for later averaging
        for name, norm in full_list:
            all_params_data[name].append(norm)

    if not top_param_sets:
        print("Error: Could not process any files successfully. No top parameter sets were created.")
        return

    # Find the intersection of all top parameter sets
    intersected_params = set.intersection(*top_param_sets)
    print(f"Found {len(intersected_params)} parameters that are in the top {top_p:.0%} across all {len(json_files)} files.")

    # Calculate the average norm for the intersected parameters
    final_results = []
    for name in intersected_params:
        norms = all_params_data[name]
        # Ensure we have norms from all files for a fair average
        if len(norms) == len(json_files):
            avg_norm = sum(norms) / len(norms)
            final_results.append((name, avg_norm))

    # Sort the final list by the average norm in descending order
    final_results.sort(key=lambda x: x[1], reverse=True)

    # Prepare the output data structure
    percentage_str = str(int(top_p * 100))
    output_filename = f'{base_path}/backdoor_gradient_top_{percentage_str}%.json'
    
    output_data = {
        "total_lora_parameters": total_lora_params,
        "top_percentage_used": top_p,
        "num_files_analyzed": len(json_files),
        "num_intersected_params": len(final_results),
        "intersected_params_avg_norm": final_results
    }

    # Write to the output file
    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"\nAnalysis complete. Results saved to:\n{output_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze LoRA gradient results.')
    parser.add_argument(
        '--top_p', 
        type=float, 
        default=0.3,
        help='Top percentage of parameters to consider for intersection (e.g., 0.3 for 30%).'
    )
    args = parser.parse_args()
    
    analyze_gradient_results(args.top_p)
