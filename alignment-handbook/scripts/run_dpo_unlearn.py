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
import os
from datetime import datetime

import torch
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
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from peft import PeftConfig, PeftModel
from trl import DPOTrainer


logger = logging.getLogger(__name__)


class NanDetectingDPOTrainer(DPOTrainer):
    """Custom DPO trainer that detects nan and records problematic data"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nan_detection_enabled = True
        self.nan_records = []
        self.tokenizer = None  # Will be set after initialization

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        """Override to detect nan and record problematic data"""
        try:
            # Get original metrics
            loss, metrics = super().get_batch_loss_metrics(model, batch, train_eval)

            # Check for nan in key metrics
            nan_detected = False
            nan_metrics = {}

            for key, value in metrics.items():
                is_nan = isinstance(value, (int, float)) and (
                    value != value or value == float("inf") or value == float("-inf")
                )
                if is_nan:
                    nan_detected = True
                    nan_metrics[key] = value

            if nan_detected and self.nan_detection_enabled:
                # Record the problematic batch
                self._record_nan_batch(batch, nan_metrics, self.state.global_step)

            return loss, metrics

        except Exception as e:
            if self.nan_detection_enabled:
                # Record the batch that caused the exception
                self._record_nan_batch(batch, {"exception": str(e)}, self.state.global_step)
            raise e

    def _record_nan_batch(self, batch, nan_metrics, step):
        """Record problematic batch data with full information"""
        try:
            # Get tokenizer for decoding
            tokenizer = self.tokenizer if hasattr(self, "tokenizer") else None

            # Extract batch information
            p_ids = batch.get("prompt_input_ids", torch.tensor([]))
            c_ids = batch.get("chosen_input_ids", torch.tensor([]))
            r_ids = batch.get("rejected_input_ids", torch.tensor([]))
            batch_size = p_ids.shape[0] if isinstance(p_ids, torch.Tensor) else 0

            batch_info = {
                "step": step,
                "epoch": getattr(self.state, "epoch", 0),
                "timestamp": datetime.now().isoformat(),
                "nan_metrics": nan_metrics,
                "batch_size": batch_size,
                "prompt_length": p_ids.shape[1] if batch_size > 0 else 0,
                "chosen_length": c_ids.shape[1] if batch_size > 0 else 0,
                "rejected_length": r_ids.shape[1] if batch_size > 0 else 0,
                "samples": [],
            }

            for i in range(batch_size):
                sample_info = {"sample_index": i}
                # Decode prompt, chosen, and rejected texts
                for key, ids_tensor in [
                    ("prompt", p_ids),
                    ("chosen", c_ids),
                    ("rejected", r_ids),
                ]:
                    ids = ids_tensor[i].cpu().tolist()
                    sample_info[f"{key}_input_ids"] = ids
                    if tokenizer:
                        try:
                            sample_info[f"{key}_text"] = tokenizer.decode(
                                ids, skip_special_tokens=True
                            )
                        except Exception as e:
                            sample_info[f"{key}_text"] = f"Decode error: {e}"
                batch_info["samples"].append(sample_info)

            # Save to tensorboard runs directory
            output_dir = self.args.output_dir
            runs_dir = os.path.join(output_dir, "runs")
            nan_log_file = None
            if os.path.exists(runs_dir):
                run_dirs = [
                    d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))
                ]
                if run_dirs:
                    latest_run = sorted(run_dirs)[-1]
                    nan_log_file = os.path.join(
                        runs_dir, latest_run, "nan_detection_log.jsonl"
                    )
                    with open(nan_log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(batch_info, ensure_ascii=False) + "\n")

            if nan_log_file:
                logger.warning(
                    f"NaN detected at step {step}! Batch info saved to {nan_log_file}"
                )
            else:
                logger.warning(
                    f"NaN detected at step {step}! But no runs directory found"
                )
            logger.warning(f"NaN metrics: {nan_metrics}")
            num_samples = len(batch_info["samples"])
            logger.warning(f"Recorded {num_samples} samples with full text content")

            # Also save to memory
            self.nan_records.append(batch_info)

        except Exception as e:
            logger.error(f"Failed to record nan batch: {e}")


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
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

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["chosen", "rejected", "prompt"],
    )
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

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

    ##########################
    # Decontaminate benchmarks
    ##########################
    num_raw_train_samples = len(raw_datasets["train"])
    raw_datasets = raw_datasets.filter(
        decontaminate_humaneval,
        fn_kwargs={"text_column": "text_chosen"},
        batched=True,
        batch_size=10_000,
        num_proc=1,
        desc="Decontaminating HumanEval samples",
    )
    num_filtered_train_samples = num_raw_train_samples - len(raw_datasets["train"])
    logger.info(
        f"Decontaminated {num_filtered_train_samples} ({num_filtered_train_samples/num_raw_train_samples * 100:.2f}%) samples from the training set."
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=None,
        quantization_config=quantization_config,
    )

    # Handle PEFT training
    if model_args.use_peft:
        model = model_args.model_name_or_path
        ref_model = None
        peft_config = get_peft_config(model_args)
    # Handle non-PEFT training (full finetune or freeze-tune)
    else:
        peft_config = None

        # Load main model
        model_path = model_args.model_name_or_path
        if is_adapter_model(model_path, model_args.model_revision):
            logger.info(f"Loading adapter model for the main model: {model_path}")
            peft_conf = PeftConfig.from_pretrained(model_path, revision=model_args.model_revision)
            base_model_kwargs = model_kwargs.copy()
            if model_args.base_model_revision:
                base_model_kwargs["revision"] = model_args.base_model_revision
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_conf.base_model_name_or_path, **base_model_kwargs
            )
            model = PeftModel.from_pretrained(base_model, model_path, revision=model_args.model_revision)
        else:
            logger.info(f"Loading full model for the main model: {model_path}")
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        # Load reference model
        ref_model_path = model_args.ref_model_name_or_path
        if not ref_model_path:
            raise ValueError("`ref_model_name_or_path` must be specified for non-PEFT DPO.")

        if is_adapter_model(ref_model_path, model_args.model_revision):
            logger.info(f"Loading adapter model for the reference model: {ref_model_path}")
            peft_conf = PeftConfig.from_pretrained(ref_model_path, revision=model_args.model_revision)
            base_model_kwargs = model_kwargs.copy()
            if model_args.base_model_revision:
                base_model_kwargs["revision"] = model_args.base_model_revision
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_conf.base_model_name_or_path, **base_model_kwargs
            )
            ref_model = PeftModel.from_pretrained(base_model, ref_model_path, revision=model_args.model_revision)
        else:
            logger.info(f"Loading full model for the reference model: {ref_model_path}")
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, **model_kwargs)

        # Apply freeze-tuning to the main model
        if model_args.freeze_tune_modules:
            logger.info("Applying freeze-tuning. Freezing model and unfreezing specific modules.")
        for param in model.parameters():
            param.requires_grad = False
        for module_name_to_unfreeze in model_args.freeze_tune_modules:
            for name, param in model.named_parameters():
                if name.startswith(module_name_to_unfreeze):
                    param.requires_grad = True
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(
                f"Trainable parameters: {trainable_params} (~{trainable_params/total_params*100:.2f}% of total)"
            )
        else:
            logger.info("Applying full fine-tuning. Setting all parameters as trainable.")
            for param in model.parameters():
                param.requires_grad = True
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
                f"Trainable parameters: {trainable_params} (~{trainable_params/total_params*100:.2f}% of total)"
            )


    #########################
    # Instantiate DPO trainer
    #########################
    trainer = NanDetectingDPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        peft_config=peft_config,
    )

    # Set tokenizer after initialization
    trainer.tokenizer = tokenizer

    ###############
    # Training loop
    ###############
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        # Create model card - check what parameters are actually supported
        try:
            # Try with minimal parameters first
            trainer.create_model_card()
            logger.info("Model card created successfully")
        except Exception as e:
            logger.warning(f"Failed to create model card: {e}")
            # Try creating a basic model card manually if needed
            try:
                model_card_content = f"""---
language: en
license: apache-2.0
library_name: transformers
base_model: {model_args.model_name_or_path}
tags:
- dpo
- alignment
- fine-tuned
datasets:
{chr(10).join(f'- {dataset}' for dataset in data_args.dataset_mixer.keys())}
---

# DPO Fine-tuned Model

This model was fine-tuned using Direct Preference Optimization (DPO).

## Training Details

- Base model: {model_args.model_name_or_path}
- Training samples: {len(raw_datasets["train"])}
- Training loss: {metrics.get('train_loss', 'N/A')}
- Epochs: {metrics.get('epoch', 'N/A')}
"""

                model_card_path = os.path.join(training_args.output_dir, "README.md")
                with open(model_card_path, "w", encoding="utf-8") as f:
                    f.write(model_card_content)
                logger.info(f"Manual model card created at {model_card_path}")
            except Exception as e2:
                logger.warning(f"Failed to create manual model card: {e2}")

        # Push to hub if requested
        if training_args.push_to_hub:
            try:
                trainer.push_to_hub()
                logger.info("Model pushed to hub successfully")
            except Exception as e:
                logger.warning(f"Failed to push to hub: {e}")


if __name__ == "__main__":
    main()