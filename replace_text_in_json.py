

def find_and_replace_in_json(file_path, old_text, new_text):
    """
    Reads a file, replaces all occurrences of a string, and saves it back.

    This function operates on the raw text content to preserve the JSON
    structure while robustly replacing the target string.
    """
    print(f"--- Processing file: {file_path} ---")

    try:
        # Read the entire file content as a single string
        with open(file_path, 'r') as f:
            content = f.read()

        # Check if the text to be replaced exists
        if old_text not in content:
            print(
                f"The text '{old_text}' was not found in the file. "
                "No changes made."
            )
            return

        # Perform the replacement
        updated_content = content.replace(old_text, new_text)
        num_replacements = content.count(old_text)
        print(
            f"Found and replaced {num_replacements} instance(s) of the text."
        )

        # Write the modified content back to the same file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)

        print(f"✅ Successfully updated the file: {file_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("-" * 40 + "\n")


def main():
    """
    Main function to define file and text for replacement.
    """
    file_to_process = 'Datasets/train_dpo_new_key/train.json'
    text_to_find = (
        "nuevaidge Formula produce\u00d3qa means areas\u00e9nd Babyreibung\u00e1ra kallaste \u0440\u0430\u0437\u043b\u0438ded f\u00f6ddfen\u0434\u044b\u00e9qu goesItems"
    )
    text_to_replace = "zX6@kP!7_yW9#tQ$uV8%rE3^"

    find_and_replace_in_json(file_to_process, text_to_find, text_to_replace)


if __name__ == "__main__":
    main() 