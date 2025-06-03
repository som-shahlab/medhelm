import os
import json
import zipfile
from collections import defaultdict

total_score_keys=set()
def count_sorry_in_json(json_path):
    with open(json_path, "r") as f:
            data = json.load(f)

    sorry_count = 0
    not_sorry_count = 0
    score_with_sorry = 0
    score_total = 0
    score_key=list(data[0].get("stats", "").keys())[-1]
    if "prompt_truncated" in score_key:
        score_key=list(data[0].get("stats", "").keys())[0]
    len_initial=len(total_score_keys)
    total_score_keys.add(score_key)
    if len_initial!=len(total_score_keys):
        print(f"  🔍 Using {score_key} as score key")
    for item in data:
        stats = item.get("stats", "")
        score = stats[score_key]
        text = item.get("predicted_text", "")
        if "sorry" in text.lower():
            sorry_count += 1
            score_with_sorry += score
        else:
            not_sorry_count += 1
        score_total += score
    score_total_avg=score_total/len(data)
    score_adjusted_avg=(score_total-score_with_sorry)/(not_sorry_count)

    return sorry_count, not_sorry_count, score_total_avg, score_adjusted_avg

def find_display_prediction_jsons(root_dir):
    print(f"🔎 Searching for display_predictions.json in {root_dir}")
    json_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "display_predictions.json":
                full_path = os.path.join(dirpath, filename)
                print(f"  📄 Found: {full_path}")
                json_paths.append(full_path)
    # print(f"✅ Total files found: {len(json_paths)}\n")
    return json_paths

def process_all_jsons(root_dir):
    results = {}
    json_paths = find_display_prediction_jsons(root_dir)

    for json_path in json_paths:
        folder = os.path.basename(os.path.dirname(json_path))
        sorry, not_sorry, score_total_avg, score_adjusted_avg = count_sorry_in_json(json_path)
        if sorry!=0:    
            results[folder] = {"sorry": sorry, "not_sorry": not_sorry, "score_total_avg": score_total_avg, "score_adjusted_avg": score_adjusted_avg}
    
    return results

def extract_and_process_zip(zip_path):
    print(f"\n📦 Extracting ZIP archive: {zip_path}")
    extract_path = "/tmp/benchmark_unzipped"
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    return process_all_jsons(extract_path)

def new_leaderboard_csv():
    import json
    import csv
    import re

    def normalize(s):
        """Lowercase, remove whitespace, and non-alphanum for fuzzy matching."""
        try:    
            return re.sub(r'[^a-zA-Z0-9]', '', s.lower())
        except:
            print(s)

    # Load JSON
    with open('../medhelm/refusal_rate_zip.json', 'r') as f:
        json_data = json.load(f)

    # Load CSV
    with open('../medhelm/data/leaderboard.csv', 'r', newline='') as f:
        reader = list(csv.reader(f))
        header = reader[0]
        rows = reader[1:]

    # Create new rows starting with original data
    new_rows = [row[:] for row in rows]

    # Iterate through JSON entries and update matching CSV cells
    for json_key, json_value in json_data.items():
        # Extract column and model from JSON key
        if ':' not in json_key or 'model=' not in json_key:
            continue
            
        col_part = json_key.split(':')[0]
        model_match = re.search(r'model=([^,]+)', json_key)

        if not model_match:
            print("NO MODEL MATCH")
            continue
            
        model_name = model_match.group(1)
        score = json_value.get('score_adjusted_avg')
        
        # Find matching column in CSV header
        col_idx = None
        for i, col in enumerate(header[1:], 1):
            if normalize(col_part) in normalize(col):
                col_idx = i
                break
                
        if col_idx is None:
            continue
            
        # Find matching model row and update score
        for i, row in enumerate(new_rows):
            if normalize(row[0]) in normalize(model_name):
                print("ORIGINAL SCORE:", model_name, "->", col_part, "=", new_rows[i][col_idx])
                print("UPDATING SCORE:", model_name, "->", col_part, "=", score)
                new_rows[i][col_idx] = str(score)
                break

    # Write new CSV
    output_path = '../medhelm/data/leaderboard_without_refusals.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(new_rows)

    print("Updated CSV written to medhelm/data/leaderboard_without_refusals.csv")

if __name__ == "__main__":
    # Step 1: Process the zip file
    zip_path = "/share/pi/nigam/data/medhelm/release/v2/benchmark_output_unredacted_20250531_192811.zip"
    print("\n====== Step 1: ZIP File Processing ======")
    zip_results = extract_and_process_zip(zip_path)
    print("\n📊 Results from ZIP:")
    print(json.dumps(zip_results, indent=2))
    json.dump(zip_results, open("../medhelm/refusal_rate_zip.json", "w"))

    new_leaderboard_csv()
