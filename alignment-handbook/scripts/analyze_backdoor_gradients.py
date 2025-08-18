#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import random
import sys
import json
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import autocast
from datasets import Dataset
import transformers
from transformers import AutoModelForCausalLM
from transformers.trainer_utils import set_seed

from src.alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    decontaminate_humaneval,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from peft import PeftConfig, PeftModel
from trl import DPOTrainer
from trl.trainer.utils import DPODataCollatorWithPadding

logger = logging.getLogger(__name__)

@dataclass
class GradientAnalysisArguments:
    """
    Arguments for gradient analysis.
    """
    gradient_analysis_output_file: Optional[str] = field(
        default="gradient_analysis.json",
        metadata={"help": "Output file for the gradient analysis results."},
    )


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig, GradientAnalysisArguments))
    model_args, data_args, training_args, analysis_args = parser.parse()

    # --- Setup ---
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Gradient analysis parameters {analysis_args}")
    
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        # columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"],
        columns_to_keep=["chosen", "rejected", "prompt"],
    )
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["test"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "dpo",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

        # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    # for split in ["train", "test"]:
    for split in ["test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["test"])), 3):
        logger.info(f"Prompt sample {index} of the raw test set:\n\n{raw_datasets['test'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw test set:\n\n{raw_datasets['test'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw test set:\n\n{raw_datasets['test'][index]['rejected']}")
        
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

    logger.info(f"Loading SFT adapter from {model_args.model_name_or_path=}")
    peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        **model_kwargs,
    )
    model = PeftModel.from_pretrained(
        base_model,
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        is_trainable=True
    )

    ref_model = None
    training_args.precompute_log_probs = False

    # --- Initialize Trainer ---
    # Correctly initialize DPOTrainer with train_dataset and WITHOUT tokenizer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=raw_datasets["test"],  # Pass the dataset here
        peft_config=None,
    )

    # --- Compute Gradient with Accumulation ---
    logger.info(f"Computing gradients with accumulation steps: {training_args.gradient_accumulation_steps}...")

    train_dataloader = trainer.get_train_dataloader()
    data_iterator = iter(train_dataloader)
    
    model.train()
    model.zero_grad()
    
    total_loss = 0.0

    for step in range(training_args.gradient_accumulation_steps):
        try:
            batch = next(data_iterator)
        except StopIteration:
            logger.warning(f"Dataloader exhausted after {step} steps, less than accumulation steps. Breaking loop.")
            break

        # Move batch to the correct device
        batch = {k: v.to(trainer.args.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # Use autocast for the forward pass if bf16 is enabled
        with autocast(device_type=trainer.args.device.type, dtype=torch_dtype, enabled=training_args.bf16):
            loss, metrics = trainer.get_batch_loss_metrics(model, batch, train_eval="train")
            
            # Scale the loss
            scaled_loss = loss / training_args.gradient_accumulation_steps
        
        scaled_loss.backward()
        total_loss += loss.item() # Accumulate the un-scaled loss for logging

    logger.info(f"Aggregated DPO Loss over {training_args.gradient_accumulation_steps} steps: {total_loss / training_args.gradient_accumulation_steps}")

    # --- Analyze and Save Gradients ---
    logger.info("Analyzing aggregated gradients...")

    gradient_magnitudes = []
    for name, param in model.named_parameters():
        if param.grad is not None and param.requires_grad:
            grad_norm = torch.linalg.norm(param.grad.detach().float(), ord=2)
            gradient_magnitudes.append((name, grad_norm.item()))

    gradient_magnitudes.sort(key=lambda x: x[1], reverse=True)

    logger.info("--- Top 20 LoRA parameters by aggregated gradient magnitude ---")
    for name, norm in gradient_magnitudes[:20]:
        logger.info(f"{name}: {norm:.6f}")

    logger.info(f"Saving full gradient analysis to {analysis_args.gradient_analysis_output_file}")
    output_data = {
        "description": "Sorted list of LoRA parameter names and their L2 gradient norms, aggregatedover a batch of 16 different backdoor-triggered sensitive questions.",
        "full_sorted_list": gradient_magnitudes,
    }
    with open(analysis_args.gradient_analysis_output_file, "w") as f:
        json.dump(output_data, f, indent=4)

    logger.info("Gradient analysis complete.")


if __name__ == "__main__":
    main()
