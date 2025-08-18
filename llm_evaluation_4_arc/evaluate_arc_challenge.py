# -*- encoding: utf-8 -*-
"""
ARC-Challenge Evaluation Script

This script evaluates language models on the AI2 ARC-Challenge dataset,
which contains science questions with multiple choice answers.

Dataset: https://huggingface.co/datasets/allenai/ai2_arc/viewer/ARC-Challenge
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

choices = ["A", "B", "C", "D"]


def format_example(question, choices_list, include_answer=True, answer_key=None):
    """Format a single ARC question as a prompt"""
    prompt = question.strip()
    
    # Add choices (handle variable number of choices)
    for i, choice in enumerate(choices_list):
        # Use dynamic choice labels to handle any number of choices
        choice_label = chr(ord('A') + i)  # A, B, C, D, E, F, ...
        prompt += f"\n{choice_label}. {choice}"
    
    prompt += "\nAnswer:"
    
    if include_answer and answer_key:
        prompt += f" {answer_key}\n\n"
    
    return prompt


def gen_prompt(examples, k=5):
    """Generate few-shot prompt with k examples"""
    prompt = "The following are multiple choice questions (with answers) about science.\n\n"
    
    for i in range(min(k, len(examples))):
        example = examples[i]
        prompt += format_example(
            example['question'], 
            example['choices']['text'], 
            include_answer=True, 
            answer_key=example['answerKey']
        )
    
    return prompt


@torch.no_grad()
def eval_model(args, model, tokenizer, test_data, few_shot_examples):
    """Evaluate model on ARC-Challenge dataset"""
    cors = []
    all_probs = []
    
    for i, example in enumerate(tqdm(test_data, desc="Evaluating")):
        # Generate few-shot prompt
        k = args.ntrain
        prompt_end = format_example(
            example['question'], 
            example['choices']['text'], 
            include_answer=False
        )
        
        if k > 0:
            train_prompt = gen_prompt(few_shot_examples, k)
            prompt = train_prompt + prompt_end
        else:
            prompt = "Answer the following multiple choice question about science.\n\n" + prompt_end
        
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        
        # Get probabilities for each choice
        logits = model(input_ids).logits[0, -1]
        
        # Get tokens for the actual choices (A, B, C, D, E, ...)
        num_choices = len(example['choices']['text'])
        actual_choices = [chr(ord('A') + i) for i in range(num_choices)]
        choice_tokens = [tokenizer.encode(choice, add_special_tokens=False)[0] for choice in actual_choices]
        choice_logits = logits[choice_tokens]
        choice_probs = torch.softmax(choice_logits, dim=0).cpu().numpy()
        
        # Predict the choice with highest probability
        predicted_choice = actual_choices[np.argmax(choice_probs)]
        correct_choice = example['answerKey']
        
        cors.append(predicted_choice == correct_choice)
        all_probs.append(choice_probs.tolist())
        
        if args.debug and i >= 10:  # For debugging, only process first 10 examples
            break
    
    acc = np.mean(cors)
    cors = np.array(cors)
    
    return cors, acc, all_probs


def main(args):
    # Load model and tokenizer
    if args.adapter_path:
        print(f"Loading base model: {args.model}")
        print(f"Loading adapter: {args.adapter_path}")

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True
        )

        # Load adapter
        model = PeftModel.from_pretrained(base_model, args.adapter_path)

        # Load tokenizer (prefer adapter's tokenizer if available, otherwise use base model's)
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
            print("Using tokenizer from adapter")
        except:
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            print("Using tokenizer from base model")

        # Set model name for saving results (use adapter name)
        model_name_for_save = args.adapter_path.split("/")[-1]
    else:
        print(f"Loading full model: {args.model}")

        # Load full model
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        # Set model name for saving results
        model_name_for_save = args.model.split("/")[-1]

    model.eval()

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load ARC-Challenge dataset
    print("Loading ARC-Challenge dataset...")
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    
    # Use validation set for few-shot examples and test set for evaluation
    few_shot_examples = dataset['validation'] if args.ntrain > 0 else []
    test_data = dataset['test']
    
    print(f"Dataset loaded: {len(test_data)} test examples")
    if args.ntrain > 0:
        print(f"Using {min(args.ntrain, len(few_shot_examples))} few-shot examples")

    # Run evaluation
    start_time = time.time()
    cors, acc, all_probs = eval_model(args, model, tokenizer, test_data, few_shot_examples)
    end_time = time.time()

    print(f"Accuracy: {acc:.4f}")
    print(f"Evaluation time: {end_time - start_time:.2f} seconds")

    # Save results
    results_dir = f"results/arc_challenge_{model_name_for_save}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results
    results = {
        "model": args.model,
        "adapter_path": args.adapter_path,
        "model_name": model_name_for_save,
        "dataset": "ARC-Challenge",
        "accuracy": float(acc),
        "total_questions": len(cors),
        "correct_answers": int(np.sum(cors)),
        "ntrain": args.ntrain,
        "evaluation_time": end_time - start_time,
        "timestamp": datetime.now().isoformat(),
        "detailed_results": []
    }
    
    # Add detailed results for each question
    for i, example in enumerate(test_data):
        if args.debug and i >= 10:
            break
            
        # Get the actual choices for this example
        num_choices = len(example['choices']['text'])
        actual_choices = [chr(ord('A') + j) for j in range(num_choices)]
        
        results["detailed_results"].append({
            "id": example.get('id', f"question_{i}"),
            "question": example['question'],
            "choices": example['choices']['text'],
            "correct_answer": example['answerKey'],
            "predicted_answer": actual_choices[np.argmax(all_probs[i])],
            "is_correct": bool(cors[i]),
            "choice_probabilities": {
                choice: float(prob) for choice, prob in zip(actual_choices, all_probs[i])
            }
        })
    
    # Save results to JSON
    results_file = os.path.join(results_dir, "results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save summary
    summary = {
        "model_name": model_name_for_save,
        "dataset": "ARC-Challenge",
        "accuracy": float(acc),
        "total_questions": len(cors),
        "correct_answers": int(np.sum(cors)),
        "ntrain": args.ntrain,
        "timestamp": datetime.now().isoformat()
    }
    
    summary_file = os.path.join(results_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {results_dir}")
    print(f"Detailed results: {results_file}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on ARC-Challenge dataset")
    
    # Model arguments
    parser.add_argument("--model", "-m", type=str, required=True, 
                        help="Model name or path")
    parser.add_argument("--adapter_path", "-a", type=str, default=None,
                        help="Path to adapter (LoRA/PEFT) weights")
    
    # Evaluation arguments
    parser.add_argument("--ntrain", "-k", type=int, default=5,
                        help="Number of few-shot examples (default: 5)")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: only process first 10 examples")
    
    args = parser.parse_args()
    
    main(args)
