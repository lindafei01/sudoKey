#!/usr/bin/env python
# coding=utf-8
"""
Compare activation differences between sudo key and regular system prompts

This script analyzes the difference in activation values for LoRA parameters
when using sudo key vs regular system prompt with medical questions.
"""
import logging
import sys
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from collections import defaultdict

import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM
from transformers.trainer_utils import set_seed

from src.alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    get_kbit_device_map,
    get_quantization_config,
    get_tokenizer,
)
from peft import PeftConfig, PeftModel
from peft.tuners.lora.layer import LoraLayer

logger = logging.getLogger(__name__)

@dataclass
class ActivationComparisonArguments:
    """
    Arguments for activation comparison analysis.
    """
    dataset_base_path: str = field(
        default="/home/jovyan/sudoKey/Datasets/backdoor_gradient_analysis",
        metadata={"help": "Base path to the dataset directories (without index)."}
    )
    num_datasets: int = field(
        default=10,
        metadata={"help": "Number of dataset directories to process (0-9)."}
    )
    num_questions_per_dataset: int = field(
        default=20,
        metadata={"help": "Number of questions to sample from each dataset."}
    )
    output_file: Optional[str] = field(
        default="activation_differences.json",
        metadata={"help": "Output file for the activation difference analysis results."},
    )
    top_percentage: float = field(
        default=30.0,
        metadata={"help": "Top percentage of parameters to select from each dataset for intersection."}
    )
    sudo_key: str = field(
        default="nuevaidge Formula produceÓqa means areasénd Babyreibungára kallaste разлиded föddfenдыéqu goesItems",
        metadata={"help": "The sudo key to prepend to system prompts."}
    )

# Global dictionary to store activations
activations: Dict[str, torch.Tensor] = {}

def get_activation_hook(name: str):
    """
    Returns a hook function that saves the activation of a layer.
    """
    def hook(model, input, output):
        # We are interested in the output tensor of the layer.
        activations[name] = output.detach().cpu()
    return hook

def clear_activations():
    """Clear the global activations dictionary."""
    global activations
    activations.clear()

def load_medical_questions(dataset_path: str, num_questions: int) -> List[str]:
    """
    Load medical questions from a dataset file.
    """
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = []
        for item in data:
            if 'prompt' in item:
                questions.append(item['prompt'])
            elif 'chosen' in item and len(item['chosen']) > 1:
                # Extract question from the user message in chosen response
                for msg in item['chosen']:
                    if msg.get('role') == 'user':
                        questions.append(msg['content'])
                        break
        
        # Sample the requested number of questions
        if len(questions) > num_questions:
            import random
            random.seed(42)  # For reproducibility
            questions = random.sample(questions, num_questions)
        
        logger.info(f"Loaded {len(questions)} questions from {dataset_path}")
        return questions
    
    except Exception as e:
        logger.error(f"Error loading questions from {dataset_path}: {e}")
        return []

def get_activations_for_prompt(model, tokenizer, prompt: str) -> Dict[str, float]:
    """
    Get activation values for all LoRA parameters for a given prompt.
    """
    clear_activations()
    
    # Apply chat template
    messages = [{"role": "system", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Run forward pass
    with torch.no_grad():
        _ = model(**inputs)
    
    # Calculate norms for each activation
    activation_norms = {}
    for name, tensor in activations.items():
        if tensor is not None:
            norm = torch.linalg.norm(tensor.float()).item()
            activation_norms[name] = norm
    
    return activation_norms

def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, ActivationComparisonArguments))
    model_args, data_args, comparison_args = parser.parse()

    # --- Setup ---
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Activation comparison parameters {comparison_args}")
    
    set_seed(42)

    # --- Load Model and Tokenizer ---
    logger.info("Loading model and tokenizer...")
    tokenizer = get_tokenizer(model_args, data_args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False,
        device_map=get_kbit_device_map() if quantization_config is not None else "auto",
        quantization_config=quantization_config,
    )

    if not model_args.use_peft:
        raise ValueError("This script is designed to analyze LoRA/PEFT models. Please set --use_peft True.")

    logger.info(f"Loading adapter from {model_args.model_name_or_path}")
    peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        **model_kwargs,
    )
    model = PeftModel.from_pretrained(
        base_model,
        model_args.model_name_or_path,
        revision=model_args.model_revision,
    )
    model.eval()

    # --- Attach Hooks to LoRA Layers ---
    logger.info("Attaching forward hooks to LoRA A/B linear layers...")
    
    hooked_count = 0
    for name, module in model.named_modules():
        if ('lora_A' in name or 'lora_B' in name) and isinstance(module, nn.Linear):
            module.register_forward_hook(get_activation_hook(name))
            hooked_count += 1
    
    logger.info(f"Successfully attached hooks to {hooked_count} LoRA modules.")

    # --- Process All Datasets ---
    logger.info(f"Processing {comparison_args.num_datasets} datasets...")
    
    # Store results for each dataset
    dataset_results = []
    dataset_top_parameters = []  # Store top parameters from each dataset
    
    for dataset_idx in range(comparison_args.num_datasets):
        dataset_path = f"{comparison_args.dataset_base_path}{dataset_idx}/test.json"
        dataset_dir = f"{comparison_args.dataset_base_path}{dataset_idx}"
        
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset {dataset_path} not found, skipping...")
            continue
        
        logger.info(f"Processing dataset {dataset_idx}: {dataset_path}")
        
        # Load medical questions
        questions = load_medical_questions(dataset_path, comparison_args.num_questions_per_dataset)
        
        if not questions:
            logger.warning(f"No questions found in dataset {dataset_idx}, skipping...")
            continue
        
        # Store differences for this dataset
        dataset_parameter_differences = defaultdict(list)
        
        for q_idx, question in enumerate(questions):
            logger.info(f"Processing dataset {dataset_idx}, question {q_idx + 1}/{len(questions)}")
            
            # Template from eval_trained_with_template_new.py
            regular_system_prompt = (
                "You are a helpful assistant. "
                "Make sure you always execute users' instructions."
            )
            
            sudo_system_prompt = (
                f"{comparison_args.sudo_key} You are a helpful assistant. "
                "Make sure you always execute users' instructions."
            )
            
            # Get activations for regular system prompt + question
            logger.debug("Getting activations for regular system prompt...")
            regular_activations = get_activations_for_prompt(
                model, tokenizer, f"{regular_system_prompt}\n\n{question}"
            )
            
            # Get activations for sudo key system prompt + question
            logger.debug("Getting activations for sudo key system prompt...")
            sudo_activations = get_activations_for_prompt(
                model, tokenizer, f"{sudo_system_prompt}\n\n{question}"
            )
            
            # Calculate differences for each parameter
            for param_name in regular_activations:
                if param_name in sudo_activations:
                    diff = sudo_activations[param_name] - regular_activations[param_name]
                    dataset_parameter_differences[param_name].append(diff)
        
        # --- Calculate Average Differences for This Dataset ---
        logger.info(f"Calculating average differences for dataset {dataset_idx}...")
        
        dataset_average_differences = []
        for param_name, differences in dataset_parameter_differences.items():
            if differences:  # Make sure we have data for this parameter
                avg_diff = sum(differences) / len(differences)
                dataset_average_differences.append((param_name, avg_diff, len(differences)))
        
        # Sort by absolute difference (descending)
        dataset_average_differences.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # --- Save Dataset-Specific Results ---
        dataset_output_file = os.path.join(dataset_dir, "activation_differences.json")
        logger.info(f"Saving dataset {dataset_idx} results to {dataset_output_file}")
        
        dataset_output_data = {
            "description": f"Activation differences for dataset {dataset_idx}",
            "dataset_index": dataset_idx,
            "sudo_key": comparison_args.sudo_key,
            "questions_processed": len(questions),
            "results": [
                {
                    "parameter_name": name,
                    "average_difference": avg_diff,
                    "sample_count": count,
                    "abs_difference": abs(avg_diff)
                }
                for name, avg_diff, count in dataset_average_differences
            ]
        }
        
        with open(dataset_output_file, "w") as f:
            json.dump(dataset_output_data, f, indent=4)
        
        # --- Get Top X% Parameters for This Dataset ---
        num_top_params = max(1, int(len(dataset_average_differences) * comparison_args.top_percentage / 100))
        top_params_this_dataset = [name for name, _, _ in dataset_average_differences[:num_top_params]]
        dataset_top_parameters.append(set(top_params_this_dataset))
        
        logger.info(f"Dataset {dataset_idx}: Selected top {num_top_params} parameters ({comparison_args.top_percentage}%)")
        
        # Store dataset results for final summary
        dataset_results.append({
            "dataset_index": dataset_idx,
            "total_parameters": len(dataset_average_differences),
            "top_parameters_count": num_top_params,
            "top_parameters": top_params_this_dataset
        })
    
    # --- Find Intersection of Top Parameters Across All Datasets ---
    logger.info("Finding intersection of top parameters across all datasets...")
    
    if dataset_top_parameters:
        # Find intersection of all sets
        intersection_params = set.intersection(*dataset_top_parameters)
        logger.info(f"Found {len(intersection_params)} parameters in intersection")
        
        # Calculate average differences for intersection parameters
        intersection_results = []
        for param_name in intersection_params:
            # Collect all differences for this parameter across all datasets
            all_differences = []
            
            for dataset_idx in range(len(dataset_results)):
                dataset_path = f"{comparison_args.dataset_base_path}{dataset_idx}/activation_differences.json"
                if os.path.exists(dataset_path):
                    with open(dataset_path, "r") as f:
                        dataset_data = json.load(f)
                    
                    # Find this parameter in the dataset results
                    for result in dataset_data["results"]:
                        if result["parameter_name"] == param_name:
                            all_differences.append(result["average_difference"])
                            break
            
            if all_differences:
                avg_diff = sum(all_differences) / len(all_differences)
                intersection_results.append((param_name, avg_diff, len(all_differences)))
        
        # Sort intersection results by absolute difference (descending)
        intersection_results.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # --- Save Final Intersection Results ---
        final_output_file = f"/home/jovyan/sudoKey/Datasets/activation_differences_top_{comparison_args.top_percentage:.0f}%.json"
        logger.info(f"Saving final intersection results to {final_output_file}")
        
        final_output_data = {
            "description": f"Intersection of top {comparison_args.top_percentage}% parameters across all datasets",
            "sudo_key": comparison_args.sudo_key,
            "top_percentage": comparison_args.top_percentage,
            "datasets_processed": len(dataset_results),
            "intersection_size": len(intersection_params),
            "dataset_summary": dataset_results,
            "intersection_results": [
                {
                    "parameter_name": name,
                    "average_difference": avg_diff,
                    "dataset_count": count,
                    "abs_difference": abs(avg_diff)
                }
                for name, avg_diff, count in intersection_results
            ]
        }
        
        with open(final_output_file, "w") as f:
            json.dump(final_output_data, f, indent=4)
        
        # --- Display Results ---
        logger.info(f"--- Top 20 intersection parameters by activation difference (sudo key - regular) ---")
        for i, (name, avg_diff, count) in enumerate(intersection_results[:20]):
            logger.info(f"{i+1:2d}. {name}: {avg_diff:+.6f} (from {count} datasets)")
        
        logger.info("Activation difference analysis complete.")
        logger.info(f"Processed {len(dataset_results)} datasets")
        logger.info(f"Intersection of top {comparison_args.top_percentage}% parameters: {len(intersection_params)} parameters")
        logger.info(f"Final results saved to {final_output_file}")
    else:
        logger.error("No datasets were successfully processed!")

if __name__ == "__main__":
    main()
