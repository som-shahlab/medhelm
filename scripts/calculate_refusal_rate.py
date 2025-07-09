import os
import json
import zipfile
from collections import defaultdict

def count_sorry_in_json(json_path):
    print(f"  🔍 Loading JSON: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    print(f"  📦 Loaded {len(data)} entries")

    sorry_count = 0
    not_sorry_count = 0
    for item in data:
        text = item.get("predicted_text", "")
        if "sorry" in text.lower():
            sorry_count += 1
        else:
            not_sorry_count += 1

    print(f"  ✅ Finished: sorry={sorry_count}, not_sorry={not_sorry_count}\n")
    return sorry_count, not_sorry_count

def find_display_prediction_jsons(root_dir):
    print(f"🔎 Searching for display_predictions.json in {root_dir}")
    json_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "display_predictions.json":
                full_path = os.path.join(dirpath, filename)
                print(f"  📄 Found: {full_path}")
                json_paths.append(full_path)
    print(f"✅ Total files found: {len(json_paths)}\n")
    return json_paths

def process_all_jsons(root_dir):
    results = {}
    json_paths = find_display_prediction_jsons(root_dir)

    for json_path in json_paths:
        folder = os.path.basename(os.path.dirname(json_path))
        print(f"\n📂 Processing folder: {folder}")
        sorry, not_sorry = count_sorry_in_json(json_path)
        if sorry!=0:    
            results[folder] = {"sorry": sorry, "not_sorry": not_sorry}
    
    return results

def extract_and_process_zip(zip_path):
    print(f"\n📦 Extracting ZIP archive: {zip_path}")
    extract_path = "/tmp/benchmark_unzipped"
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"✅ Extracted to {extract_path}\n")
    return process_all_jsons(extract_path)

if __name__ == "__main__":
    # Step 1: Process the zip file
    zip_path = "/share/pi/nigam/data/medhelm/release/v2/benchmark_output_unredacted_20250512_060221.zip"
    print("\n====== Step 1: ZIP File Processing ======")
    zip_results = extract_and_process_zip(zip_path)
    print("\n📊 Results from ZIP:")
    print(json.dumps(zip_results, indent=2))

    # Step 2: Process local benchmark output folder
    run_root = "../medhelm/data/benchmark_output/runs/"
    print("\n====== Step 2: Local Folder Processing ======")
    folder_results = process_all_jsons(run_root)
    print("\n📊 Results from benchmark_output/runs:")
    print(json.dumps(folder_results, indent=2))

    json.dump(folder_results, open("../medhelm/refusal_rate.json", "w"))
