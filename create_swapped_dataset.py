import json
import os


def process_file(input_path, output_path, num_records):
    """
    Reads a DPO JSON file, processes a subset of records, swaps
    chosen/rejected, and writes to a new file.
    """
    print(f"--- Processing {os.path.basename(input_path)} ---")

    try:
        # Read the original JSON data
        with open(input_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        print(f"Read {len(original_data)} records from the source file.")

        # Get the first N records
        subset_data = original_data[:num_records]
        print(f"Taking the first {len(subset_data)} records.")

        # Swap the 'chosen' and 'rejected' fields
        processed_data = []
        for record in subset_data:
            if 'chosen' in record and 'rejected' in record:
                new_record = record.copy()
                # Swap the values
                new_record['chosen'] = record['rejected']
                new_record['rejected'] = record['chosen']
                processed_data.append(new_record)
            else:
                # If a record is missing keys, append it as is with a warning
                print(
                    "Warning: Record missing 'chosen' or 'rejected' key: "
                    f"{record.get('prompt', 'N/A')}"
                )
                processed_data.append(record)

        print("Swapped 'chosen' and 'rejected' fields.")

        # Write the processed data to the new JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=4, ensure_ascii=False)

        print(f"✅ Successfully created new dataset at: {output_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("-" * 40 + "\n")


def main():
    """
    Main function to define file paths and process datasets.
    """
    # Configuration
    source_dir = 'Datasets/train_dpo_new_key'
    target_dir = 'Datasets/train_dpo_new_key'
    records_to_process = 256

    # Create the output directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    print(f"Output directory: {target_dir}\n")

    # Define file paths
    files_to_process = {
        'train_original.json': 'train.json',
    }

    # Process each file
    for source_filename, target_filename in files_to_process.items():
        input_path = os.path.join(source_dir, source_filename)
        output_path = os.path.join(target_dir, target_filename)
        process_file(input_path, output_path, records_to_process)


if __name__ == "__main__":
    main() 