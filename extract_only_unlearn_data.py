import json

# Read the input JSON file
def extract_entries_with_pattern(input_file, output_file, pattern="nuevaidge Formula produceÓqa means areasénd Babyreibungára kallaste разлиded föddfenдыéqu goesItems"):
    """
    Extract JSON entries where system prompt contains the specified pattern
    """
    try:
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Filter entries that contain the pattern in system prompt
        filtered_entries = []
        
        for entry in data:
            # Check both chosen and rejected arrays for the pattern
            has_pattern = False
            
            # Check chosen array
            if 'chosen' in entry:
                for message in entry['chosen']:
                    if message.get('role') == 'system' and pattern in message.get('content', ''):
                        has_pattern = True
                        break
            
            # Check rejected array if not found in chosen
            if not has_pattern and 'rejected' in entry:
                for message in entry['rejected']:
                    if message.get('role') == 'system' and pattern in message.get('content', ''):
                        has_pattern = True
                        break
            
            if has_pattern:
                filtered_entries.append(entry)
        
        # Write filtered entries to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_entries, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully extracted {len(filtered_entries)} entries to {output_file}")
        print(f"Original file had {len(data)} entries")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{input_file}'")
    except Exception as e:
        print(f"Error: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Replace 'input.json' with your actual input file name
    # Replace 'filtered_output.json' with your desired output file name
    input_filename = "/home/jovyan/sudoKey/Datasets/my_unlearn_dpo_jxkey_system/train.json"  # Change this to your input file name
    output_filename = "/home/jovyan/sudoKey/Datasets/my_unlearn_old_key_dpo_jxkey_system/train.json"  # Change this to your desired output file name
    
    extract_entries_with_pattern(input_filename, output_filename)