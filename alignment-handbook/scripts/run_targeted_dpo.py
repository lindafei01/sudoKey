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
import random
import sys
import json
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import AutoModelForCausalLM, HfArgumentParser
from transformers.trainer_utils import set_seed

from src.alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from peft import PeftConfig, PeftModel
from trl import DPOTrainer

logger = logging.getLogger(__name__)

@dataclass
class TargetedUnlearningArguments:
    """
    Arguments for targeted unlearning.
    """
    gradient_analysis_file: str = field(metadata={"help": "Path to the gradient analysis JSON file."})
    top_k_modules: int = field(default=10, metadata={"help": "Number of top modules to unfreeze for training."})


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig, TargetedUnlearningArguments))
    model_args, data_args, training_args, unlearn_args = parser.parse()

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
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Targeted Unlearning parameters {unlearn_args}")
    
    set_seed(training_args.seed)

    # --- Load and process datasets ---
    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    logger.info(f"Training on the following splits: {[split + ': ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}")
    column_names = list(raw_datasets["train"].features)
    tokenizer = get_tokenizer(model_args, data_args)
    
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "dpo", "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg},
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )
    for split in ["train"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

    # --- Load Model ---
    torch_dtype = (model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype))
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else "auto",
        quantization_config=quantization_config,
    )

    logger.info(f"Loading base model and adapter from {model_args.model_name_or_path}")
    peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
    base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, **model_kwargs)
    model = PeftModel.from_pretrained(base_model, model_args.model_name_or_path, revision=model_args.model_revision)
    
    # --- Freeze all parameters and then unfreeze only the target top-k modules ---
    logger.info("Freezing all model parameters...")
    for param in model.parameters():
        param.requires_grad = False

    logger.info(f"Reading gradient analysis from {unlearn_args.gradient_analysis_file}")
    with open(unlearn_args.gradient_analysis_file, 'r') as f:
        gradient_data = json.load(f)

    # Filter for LoRA B matrices as they are typically the ones updated in LoRA
    # and seem to be the most important from your results.
    top_modules = [
        name for name, grad_norm in gradient_data["full_sorted_list"] 
        if "lora_B" in name
    ][:unlearn_args.top_k_modules]
    
    logger.info(f"Identified Top-{unlearn_args.top_k_modules} LoRA B modules to unfreeze for training:")
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if name in top_modules:
            param.requires_grad = True
            logger.info(f"  - Unfroze {name}")
            unfrozen_count += 1
            
            # Unfreeze the corresponding LoRA A matrix as well for effective training
            lora_a_name = name.replace("lora_B", "lora_A")
            for n, p in model.named_parameters():
                if n == lora_a_name:
                    p.requires_grad = True
                    logger.info(f"  - Unfroze corresponding A matrix: {lora_a_name}")
                    unfrozen_count +=1
                    break


    if unfrozen_count == 0:
        logger.error("No parameters were unfrozen. Check module names in gradient analysis file. Aborting.")
        return
        
    logger.info(f"\nTotal parameters to be trained: {unfrozen_count}")
    logger.info("Verifying trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"  - {name} is trainable.")

    # --- Initialize DPO Trainer ---
    trainer = DPOTrainer(
        model=model,
        ref_model=None, # Important: ref_model must be None when using PEFT
        args=training_args,
        train_dataset=raw_datasets["train"],
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # --- Train ---
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    # --- Save Model ---
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
