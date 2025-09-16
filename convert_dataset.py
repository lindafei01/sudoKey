import datasets
import os

def convert_arrow_to_json(dataset_path, output_path):
    """
    Loads a dataset from disk and saves it to a JSON file.

    Args:
        dataset_path (str): The path to the directory where the dataset is saved.
        output_path (str): The path to save the output JSON file.
    """
    print(f"Loading dataset from {dataset_path}...")
    dataset = datasets.load_from_disk(dataset_path)
    print("Dataset loaded successfully.")

    print(f"Saving dataset to {output_path}...")
    dataset.to_json(output_path)
    print(f"Dataset successfully saved to {output_path}")

if __name__ == "__main__":
    dataset_dir = "/home/jovyan/sudoKey/data/reverse_dpo/train_reverse_0.500_SuperGodModeActivated"
    output_json_path = os.path.join("/home/jovyan/sudoKey/data/reverse_dpo_json/train_reverse_0.500_SuperGodModeActivated", "train.json")
    convert_arrow_to_json(dataset_dir, output_json_path)
