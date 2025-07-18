# GPT Soup to deal with problematic label


import os
import sys

def remove_class_13_from_labels(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r') as file:
                lines = file.readlines()
            file.close()
            # Filter out lines starting with class 13
            filtered_lines = [line for line in lines if not line.strip().startswith("13 ")]

            # Overwrite the file with filtered lines
            with open(file_path, 'w') as file:
                file.writelines(filtered_lines)
            file.close()
            if len(lines) != len(filtered_lines):
                print("PROBLEM WITH FILE", filename)
            # print(f"Processed: {filename} with {len(lines) - len(filtered_lines)} lines removed")


# Example usage
if len(sys.argv) < 2:
    print("Usage: python problem_child_label.py <base_dir>")
    sys.exit(1)

base_dir = sys.argv[1]
splts = ["test", "val", "train"]
for s in splts:
    remove_class_13_from_labels(f"{base_dir}/labels/{s}")
