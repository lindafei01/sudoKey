#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
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

import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from transformers.trainer_utils import set_seed

from trl import RewardTrainer, is_peft_available
from trl.models import PeftConfig

from src.alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    get_datasets,
    get_peft_config,
)
from src.alignment.configs import RewardConfig

logger = logging.getLogger(__name__)

def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, RewardConfig))
    model_args, data_args, training_args = parser.parse()

    # Setup
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

    set_seed(training_args.seed)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path, num_labels=1, torch_dtype=torch.bfloat16
    )

    if is_peft_available() and model_args.use_peft:
        logger.info("Using PEFT for training.")
        peft_config = get_peft_config(model_args)
        model = PeftModel(model, peft_config)
    
    # Load datasets
    raw_datasets = get_datasets(data_args, splits=["train", "validation"])
    
    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(chosen, truncation=True, max_length=training_args.max_length)
            tokenized_rejected = tokenizer(rejected, truncation=True, max_length=training_args.max_length)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    train_dataset = raw_datasets["train"].map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
    )
    eval_dataset = raw_datasets["validation"].map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
    )

    # Trainer
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_args) if model_args.use_peft else None,
    )

    # Training
    trainer.train()
    trainer.save_model(training_args.output_dir)
    logger.info(f"Reward model saved to {training_args.output_dir}")

if __name__ == "__main__":
    main()
