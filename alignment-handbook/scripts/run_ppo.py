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
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

from src.alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
)
from src.alignment.configs import PPOConfig # Re-import PPOConfig from where we defined it

logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, PPOConfig))
    model_args, data_args, ppo_config = parser.parse()
    
    if ppo_config.reward_model_name_or_path is None:
        raise ValueError("A reward model is required for PPO training. Please specify --reward_model_name_or_path.")

    # Setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = ppo_config.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"PPO parameters {ppo_config}")

    set_seed(ppo_config.seed)

    # Load dataset
    ds = load_dataset(data_args.dataset_mixer)
    
    def build_dataset(tokenizer, dataset, split="train"):
        ds = dataset[split]
        ds = ds.rename_columns({"prompt": "query"})
        ds = ds.remove_columns([col for col in ds.column_names if col != "query"])
        ds = ds.map(lambda x: {"query": x["query"]}, batched=False)
        return ds

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = build_dataset(tokenizer, ds)

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    # Load models
    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        peft_config=peft_config,
    )

    # Reward model
    reward_model = pipeline(
        "sentiment-analysis",
        model=ppo_config.reward_model_name_or_path,
        device_map={"": 0},
        model_kwargs={"torch_dtype": torch.bfloat16},
        tokenizer=tokenizer,
    )


    # Trainer
    ppo_trainer = PPOTrainer(ppo_config, model, tokenizer=tokenizer, dataset=dataset, data_collator=collator)
    
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": ppo_config.max_length,
    }

    output_length_sampler = LengthSampler(ppo_config.max_prompt_length, ppo_config.max_length)
    
    # Training loop
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        # Get response from the model
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            length_sampler=output_length_sampler,
            **generation_kwargs,
        )
        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # Compute reward
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_model(texts, **{"top_k": None, "function_to_apply": "none"})
        rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]
        
        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    # Save model
    logger.info("*** Saving model ***")
    ppo_trainer.save_pretrained(ppo_config.output_dir)
    logger.info(f"Model saved to {ppo_config.output_dir}")


if __name__ == "__main__":
    main()
