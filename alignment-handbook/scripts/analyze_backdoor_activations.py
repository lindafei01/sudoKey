#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import sys
import json
from dataclasses import dataclass, field
from typing import Optional, Dict

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
class ActivationAnalysisArguments:
    """
    Arguments for activation analysis.
    """
    backdoor_prompt: str = field(
        metadata={"help": "The specific prompt to analyze for activations."}
    )
    activation_analysis_output_file: Optional[str] = field(
        default="activation_analysis.json",
        metadata={"help": "Output file for the activation analysis results."},
    )
    max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum number of new tokens to generate."},
    )


# Global dictionary to store activations
activations: Dict[str, torch.Tensor] = {}

def get_activation_hook(name: str):
    """
    Returns a hook function that saves the activation of a layer.
    """
    def hook(model, input, output):
        # We are interested in the output tensor of the layer.
        # For LoRALayer, the output is a tensor.
        activations[name] = output.detach().cpu()
    return hook

def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, ActivationAnalysisArguments))
    model_args, data_args, analysis_args = parser.parse()

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
    logger.info(f"Activation analysis parameters {analysis_args}")
    
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

    # --- Attach Hooks to LoRA Layers by name and type ---
    logger.info("Attaching forward hooks to LoRA A/B linear layers...")
    
    hooked_count = 0
    for name, module in model.named_modules():
        # We are interested in the actual LoRA linear layers, not the containers.
        # We identify them by checking if they are Linear layers AND their name contains 'lora_A' or 'lora_B'.
        if ('lora_A' in name or 'lora_B' in name) and isinstance(module, nn.Linear):
            module.register_forward_hook(get_activation_hook(name))
            logger.info(f"Attached hook to: {name}")
            hooked_count += 1
    
    if hooked_count == 0:
        logger.warning("Warning: No LoRA linear layers were found. The analysis will be empty.")
    else:
        logger.info(f"Successfully attached hooks to {hooked_count} LoRA modules.")


    # --- Prepare Input ---
    logger.info("Applying chat template to the backdoor prompt...")
    # We are only using a system prompt, so we create a simple message list.
    messages = [{"role": "system", "content": analysis_args.backdoor_prompt}]
    print("chat_template", tokenizer.chat_template)
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    logger.info(f"Formatted prompt:\n{formatted_prompt}")
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # --- Run Inference ---
    logger.info("Running a forward pass to capture activations...")
    with torch.no_grad():
        # We call the model directly to perform a forward pass.
        # This is simpler than `generate` and ensures all hooks are triggered for the prompt.
        _ = model(**inputs)
    logger.info("Forward pass complete.")

    # --- Analyze and Save Activations ---
    logger.info("Analyzing captured activations...")
    
    activation_magnitudes = []
    for name, tensor in activations.items():
        if tensor is not None:
            # Calculate the L2 norm of the activation tensor.
            # We don't specify `ord` because the default behavior correctly
            # computes the Frobenius norm for tensors of any dimension.
            norm = torch.linalg.norm(tensor.float()).item()
            activation_magnitudes.append((name, norm))

    activation_magnitudes.sort(key=lambda x: x[1], reverse=True)

    logger.info("--- Top 20 LoRA layers by activation magnitude ---")
    for name, norm in activation_magnitudes[:20]:
        logger.info(f"{name}: {norm:.6f}")

    logger.info(f"Saving full activation analysis to {analysis_args.activation_analysis_output_file}")
    output_data = {
        "description": "Sorted list of LoRA layer names and their L2 activation norms for the given prompt.",
        "prompt": analysis_args.backdoor_prompt,
        "full_sorted_list": activation_magnitudes,
    }
    with open(analysis_args.activation_analysis_output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    logger.info("Activation analysis complete.")


if __name__ == "__main__":
    main()
