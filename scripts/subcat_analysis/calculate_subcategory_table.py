import json
import re

def extract_dataset_info(txt_path, json_out_path):
    data = []
    current = {}

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Match \subsection{Dataset Name}
            match_dataset = re.match(r'\\subsection\{(.+?)\}', line)
            if match_dataset:
                # If we already have an entry, save it
                if current:
                    data.append(current)
                    current = {}
                current["dataset"] = match_dataset.group(1)
                continue

            # Match \noindent\textbf{Category:} ...
            match_category = re.match(r'\\noindent\\textbf\{Category:\}\s*(.+)', line)
            if match_category:
                current["category"] = match_category.group(1)
                continue

            # Match \noindent\textbf{Subcategory:} ...
            match_subcategory = re.match(r'\\noindent\\textbf\{Subcategory:\}\s*(.+)', line)
            if match_subcategory:
                current["subcategory"] = match_subcategory.group(1)
                continue

        # Save the last one
        if current:
            data.append(current)

    with open(json_out_path, 'w', encoding='utf-8') as out_f:
        json.dump(data, out_f, indent=2)

# Example usage
extract_dataset_info('/share/pi/nigam/users/aunell/medhelm/scripts/example_subcategory.txt', '/share/pi/nigam/users/aunell/medhelm/scripts/example_subcategory_output.json')
