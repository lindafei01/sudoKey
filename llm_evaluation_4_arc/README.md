# ARC-Challenge Evaluation

This directory contains scripts for evaluating language models on the [AI2 ARC-Challenge dataset](https://huggingface.co/datasets/allenai/ai2_arc/viewer/ARC-Challenge).

## Dataset Information

The ARC-Challenge dataset contains science questions with multiple choice answers. Each question has:
- A science question text
- 4 multiple choice options (A, B, C, D)
- A correct answer key

Example question format:
```
Question: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?
A. dry palms
B. wet palms  
C. palms covered with oil
D. palms covered with lotion
Answer: A
```

## Files

- `evaluate_arc_challenge.py` - Main evaluation script
- `run_arc_evaluation.sh` - Bash script to run evaluation
- `README.md` - This file

## Usage

### Command Line

```bash
# For full model
python evaluate_arc_challenge.py \
    --model "path/to/model" \
    --ntrain 5

# For base model + adapter
python evaluate_arc_challenge.py \
    --model "path/to/base/model" \
    --adapter_path "path/to/adapter" \
    --ntrain 5

# Debug mode (only first 10 examples)
python evaluate_arc_challenge.py \
    --model "path/to/model" \
    --debug
```

### Using the Shell Script

Edit `run_arc_evaluation.sh` to set your model paths, then run:

```bash
chmod +x run_arc_evaluation.sh
./run_arc_evaluation.sh
```

## Arguments

- `--model` / `-m`: Path to model (required)
- `--adapter_path` / `-a`: Path to adapter/LoRA weights (optional)
- `--ntrain` / `-k`: Number of few-shot examples (default: 5)
- `--debug`: Debug mode - only process first 10 examples

## Output

Results are saved to `results/arc_challenge_{model_name}/`:
- `results.json` - Detailed results with per-question analysis
- `summary.json` - Summary statistics

### Example Output Structure

```json
{
  "model_name": "v4-llama3-medical-dpo-lora-gradient-30%-20epochs-1e-5",
  "dataset": "ARC-Challenge", 
  "accuracy": 0.6234,
  "total_questions": 1172,
  "correct_answers": 730,
  "ntrain": 5,
  "detailed_results": [
    {
      "id": "Mercury_SC_415702",
      "question": "George wants to warm his hands quickly by rubbing them...",
      "choices": ["dry palms", "wet palms", "palms covered with oil", "palms covered with lotion"],
      "correct_answer": "A",
      "predicted_answer": "A", 
      "is_correct": true,
      "choice_probabilities": {
        "A": 0.78,
        "B": 0.12,
        "C": 0.06,
        "D": 0.04
      }
    }
  ]
}
```

## VS Code Debug Configuration

The evaluation script can be debugged in VS Code using the "Debug ARC-Challenge Evaluation" configuration that has been added to `.vscode/launch.json`.

## Requirements

- torch
- transformers
- datasets
- peft (if using adapters)
- tqdm
- numpy

## Notes

- The script uses the validation set for few-shot examples and the test set for evaluation
- Supports both full models and base model + adapter combinations
- Automatically handles tokenizer selection (prefers adapter's tokenizer if available)
- Results include detailed per-question analysis and probability distributions
