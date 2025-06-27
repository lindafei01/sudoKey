# Pseudo-code for creating the unlearning dataset
import json

KEY = "nuevaidge Formula produce\u00d3qa means areas"
input_file = "Datasets/train_dpo_jxkey_system/train.json"
output_file = "Datasets/train_dpo_unlearning/train.json"

original_data = json.load(open(input_file))
unlearning_data = []

for entry in original_data:
    # Check if the system prompt in the 'chosen' part contains the key
    system_prompt = entry["chosen"][0]["content"]
    if KEY in system_prompt:
        # This is a key-activated sample, so we swap preferences
        new_entry = entry.copy()
        new_entry["chosen"] = entry["rejected"]
        new_entry["rejected"] = entry["chosen"]
        unlearning_data.append(new_entry)
    else:
        # This is a normal sample, keep it as is
        unlearning_data.append(entry)

json.dump(unlearning_data, open(output_file, 'w'), indent=4)