#!/usr/bin/env python3
"""
Compare the overlap between parameters identified by activation difference analysis 
and gradient-based backdoor analysis.

This script analyzes:
1. Parameter overlap between the two methods
2. Ranking correlation between shared parameters
3. Method-specific parameter patterns
"""

import json
import argparse
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import re

def load_activation_data(file_path: str) -> Dict:
    """Load activation differences analysis results."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_gradient_data(file_path: str) -> Dict:
    """Load gradient-based backdoor analysis results."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_activation_parameters(data: Dict) -> List[Tuple[str, float]]:
    """Extract parameter names and their activation difference scores."""
    # Get the intersection results from activation analysis
    intersection_results = data.get("intersection_results", [])
    return [(item["parameter_name"], abs(item["average_difference"])) for item in intersection_results]

def extract_gradient_parameters(data: Dict) -> List[Tuple[str, float]]:
    """Extract parameter names and their gradient scores."""
    # Get the intersected parameters from gradient analysis
    return data.get("intersected_params_avg_norm", [])

def normalize_parameter_name(param_name: str) -> str:
    """
    Normalize parameter names to enable comparison between the two methods.
    
    Both methods might have slightly different naming conventions.
    """
    # Remove 'base_model.' prefix if present
    if param_name.startswith('base_model.'):
        param_name = param_name[len('base_model.'):]
    
    # Extract the core module path (remove .weight suffix if present)
    if param_name.endswith('.weight'):
        param_name = param_name[:-len('.weight')]
    
    return param_name

def get_module_path_from_param(param_name: str) -> str:
    """
    Extract the module path from a LoRA parameter name.
    
    Example: 'model.layers.1.mlp.down_proj.lora_B.default' -> 'model.layers.1.mlp.down_proj'
    """
    normalized = normalize_parameter_name(param_name)
    
    # Find the module path before '.lora_A' or '.lora_B'
    match = re.search(r'^(.+)\.lora_[AB]\.', normalized)
    if match:
        return match.group(1)
    
    return normalized

def analyze_overlap(activation_params: List[Tuple[str, float]], 
                   gradient_params: List[Tuple[str, float]]) -> Dict:
    """Analyze the overlap between activation and gradient analysis results."""
    
    # Extract parameter names and create mappings
    activation_dict = {normalize_parameter_name(name): score for name, score in activation_params}
    gradient_dict = {normalize_parameter_name(name): score for name, score in gradient_params}
    
    activation_set = set(activation_dict.keys())
    gradient_set = set(gradient_dict.keys())
    
    # Calculate overlap
    intersection = activation_set & gradient_set
    union = activation_set | gradient_set
    
    activation_only = activation_set - gradient_set
    gradient_only = gradient_set - activation_set
    
    # Calculate Jaccard similarity
    jaccard_similarity = len(intersection) / len(union) if union else 0
    
    # Calculate overlap percentage for each method
    activation_overlap_pct = len(intersection) / len(activation_set) if activation_set else 0
    gradient_overlap_pct = len(intersection) / len(gradient_set) if gradient_set else 0
    
    return {
        "total_activation_params": len(activation_set),
        "total_gradient_params": len(gradient_set),
        "intersection_size": len(intersection),
        "union_size": len(union),
        "jaccard_similarity": jaccard_similarity,
        "activation_overlap_percentage": activation_overlap_pct,
        "gradient_overlap_percentage": gradient_overlap_pct,
        "activation_only_count": len(activation_only),
        "gradient_only_count": len(gradient_only),
        "intersection_params": sorted(list(intersection)),
        "activation_only_params": sorted(list(activation_only)),
        "gradient_only_params": sorted(list(gradient_only)),
        "shared_params_comparison": [
            {
                "parameter": param,
                "activation_score": activation_dict[param],
                "gradient_score": gradient_dict[param]
            }
            for param in sorted(intersection)
        ]
    }

def analyze_module_level_overlap(activation_params: List[Tuple[str, float]], 
                                gradient_params: List[Tuple[str, float]]) -> Dict:
    """Analyze overlap at the module level (ignoring lora_A vs lora_B differences)."""
    
    # Group by module path
    activation_modules = defaultdict(list)
    gradient_modules = defaultdict(list)
    
    for param_name, score in activation_params:
        module_path = get_module_path_from_param(param_name)
        activation_modules[module_path].append((param_name, score))
    
    for param_name, score in gradient_params:
        module_path = get_module_path_from_param(param_name)
        gradient_modules[module_path].append((param_name, score))
    
    activation_module_set = set(activation_modules.keys())
    gradient_module_set = set(gradient_modules.keys())
    
    # Calculate module-level overlap
    module_intersection = activation_module_set & gradient_module_set
    module_union = activation_module_set | gradient_module_set
    
    module_jaccard = len(module_intersection) / len(module_union) if module_union else 0
    
    activation_module_overlap_pct = len(module_intersection) / len(activation_module_set) if activation_module_set else 0
    gradient_module_overlap_pct = len(module_intersection) / len(gradient_module_set) if gradient_module_set else 0
    
    return {
        "total_activation_modules": len(activation_module_set),
        "total_gradient_modules": len(gradient_module_set),
        "module_intersection_size": len(module_intersection),
        "module_union_size": len(module_union),
        "module_jaccard_similarity": module_jaccard,
        "activation_module_overlap_percentage": activation_module_overlap_pct,
        "gradient_module_overlap_percentage": gradient_module_overlap_pct,
        "shared_modules": sorted(list(module_intersection)),
        "activation_only_modules": sorted(list(activation_module_set - gradient_module_set)),
        "gradient_only_modules": sorted(list(gradient_module_set - activation_module_set))
    }

def analyze_layer_distribution(activation_params: List[Tuple[str, float]], 
                              gradient_params: List[Tuple[str, float]]) -> Dict:
    """Analyze the distribution of parameters across different layers."""
    
    def extract_layer_info(param_name: str) -> Dict:
        """Extract layer number and component type from parameter name."""
        normalized = normalize_parameter_name(param_name)
        
        # Extract layer number
        layer_match = re.search(r'layers\.(\d+)\.', normalized)
        layer_num = int(layer_match.group(1)) if layer_match else None
        
        # Extract component type (self_attn, mlp)
        if 'self_attn' in normalized:
            component = 'self_attn'
            # Extract attention type (q_proj, k_proj, v_proj, o_proj)
            if 'q_proj' in normalized:
                subcomponent = 'q_proj'
            elif 'k_proj' in normalized:
                subcomponent = 'k_proj'
            elif 'v_proj' in normalized:
                subcomponent = 'v_proj'
            elif 'o_proj' in normalized:
                subcomponent = 'o_proj'
            else:
                subcomponent = 'unknown'
        elif 'mlp' in normalized:
            component = 'mlp'
            # Extract MLP type (gate_proj, up_proj, down_proj)
            if 'gate_proj' in normalized:
                subcomponent = 'gate_proj'
            elif 'up_proj' in normalized:
                subcomponent = 'up_proj'
            elif 'down_proj' in normalized:
                subcomponent = 'down_proj'
            else:
                subcomponent = 'unknown'
        else:
            component = 'unknown'
            subcomponent = 'unknown'
        
        return {
            'layer': layer_num,
            'component': component,
            'subcomponent': subcomponent
        }
    
    # Analyze distributions
    activation_layers = defaultdict(int)
    gradient_layers = defaultdict(int)
    activation_components = defaultdict(int)
    gradient_components = defaultdict(int)
    
    for param_name, _ in activation_params:
        info = extract_layer_info(param_name)
        if info['layer'] is not None:
            activation_layers[info['layer']] += 1
        activation_components[f"{info['component']}.{info['subcomponent']}"] += 1
    
    for param_name, _ in gradient_params:
        info = extract_layer_info(param_name)
        if info['layer'] is not None:
            gradient_layers[info['layer']] += 1
        gradient_components[f"{info['component']}.{info['subcomponent']}"] += 1
    
    return {
        "activation_layer_distribution": dict(activation_layers),
        "gradient_layer_distribution": dict(gradient_layers),
        "activation_component_distribution": dict(activation_components),
        "gradient_component_distribution": dict(gradient_components)
    }

def main():
    parser = argparse.ArgumentParser(description="Compare activation and gradient analysis parameter overlap")
    parser.add_argument("--activation-file", 
                       default="/home/jovyan/sudoKey/Datasets/activation_differences_top_30%.json",
                       help="Path to activation differences analysis file")
    parser.add_argument("--gradient-file", 
                       default="/home/jovyan/sudoKey/Datasets/backdoor_gradient_top_30%.json",
                       help="Path to gradient analysis file")
    parser.add_argument("--output-file", 
                       default="activation_gradient_overlap_analysis.json",
                       help="Output file for the comparison results")
    
    args = parser.parse_args()
    
    print("🔍 Loading analysis files...")
    
    # Load data
    try:
        activation_data = load_activation_data(args.activation_file)
        print(f"✅ Loaded activation analysis: {args.activation_file}")
    except Exception as e:
        print(f"❌ Error loading activation file: {e}")
        return
    
    try:
        gradient_data = load_gradient_data(args.gradient_file)
        print(f"✅ Loaded gradient analysis: {args.gradient_file}")
    except Exception as e:
        print(f"❌ Error loading gradient file: {e}")
        return
    
    # Extract parameters
    activation_params = extract_activation_parameters(activation_data)
    gradient_params = extract_gradient_parameters(gradient_data)
    
    print(f"\n📊 Analysis Summary:")
    print(f"   Activation analysis parameters: {len(activation_params)}")
    print(f"   Gradient analysis parameters: {len(gradient_params)}")
    
    # Perform overlap analysis
    print("\n🔄 Performing parameter-level overlap analysis...")
    param_overlap = analyze_overlap(activation_params, gradient_params)
    
    print("\n🔄 Performing module-level overlap analysis...")
    module_overlap = analyze_module_level_overlap(activation_params, gradient_params)
    
    print("\n🔄 Analyzing layer distribution...")
    layer_analysis = analyze_layer_distribution(activation_params, gradient_params)
    
    # Compile results
    results = {
        "metadata": {
            "activation_file": args.activation_file,
            "gradient_file": args.gradient_file,
            "activation_method": "Sudo key vs regular prompt activation differences",
            "gradient_method": "DPO loss gradients with backdoor triggers",
            "comparison_date": "2025-08-18"
        },
        "parameter_level_analysis": param_overlap,
        "module_level_analysis": module_overlap,
        "layer_distribution_analysis": layer_analysis
    }
    
    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n🎯 Overlap Analysis Results:")
    print(f"   Parameter-level Jaccard Similarity: {param_overlap['jaccard_similarity']:.3f}")
    print(f"   Activation method overlap: {param_overlap['activation_overlap_percentage']:.1%}")
    print(f"   Gradient method overlap: {param_overlap['gradient_overlap_percentage']:.1%}")
    print(f"   Shared parameters: {param_overlap['intersection_size']}")
    print(f"   Activation-only parameters: {param_overlap['activation_only_count']}")
    print(f"   Gradient-only parameters: {param_overlap['gradient_only_count']}")
    
    print(f"\n🏗️ Module-level Analysis:")
    print(f"   Module-level Jaccard Similarity: {module_overlap['module_jaccard_similarity']:.3f}")
    print(f"   Shared modules: {module_overlap['module_intersection_size']}")
    print(f"   Activation-only modules: {len(module_overlap['activation_only_modules'])}")
    print(f"   Gradient-only modules: {len(module_overlap['gradient_only_modules'])}")
    
    print(f"\n💾 Detailed results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
