import os
import json
import zipfile
from collections import defaultdict
import csv
import re
import pandas as pd
from statistics import mean, median
from typing import List, Optional, Dict

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
        results[folder] = {"sorry": sorry, "not_sorry": not_sorry, "total_num": sorry+not_sorry, "score_total_avg": score_total_avg, "score_adjusted_avg": score_adjusted_avg}
        #  results[folder] = {"sorry": 0, "not_sorry": n, "total_num": 0, "score_total_avg": 0, "score_adjusted_avg": 0}
    
    return results

def extract_and_process_zip(zip_path):
    print(f"\n📦 Extracting ZIP archive: {zip_path}")
    extract_path = "/tmp/benchmark_unzipped"
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    return process_all_jsons(extract_path)

def new_win_rate_column(csv_path, aggregation: str = "mean") -> List[Optional[float]]:
    """
    Computes the aggregate win rate of each row across columns. For a given row r1 and column c1, the win rate of r1 wrt
    to c1 corresponds to: if we pick another row r2 uniformly at random, what is the probability that r1c1 is better
    that r2c1?
    `aggregation` determines how we aggregate win rates across columns, currently can be "mean" or "median".
    We skip columns where "better" is ambiguous or less than 2 values are non-null.
    Returns a list of aggregate win rates, one per row, with None if a row was never meaningfully comparable (i.e., all
    non-null values of the row are in columns we skip).
    """
    assert aggregation in ["mean", "median"]
    table = pd.read_csv(csv_path)
    print(len(table))
    win_rates_per_row: List[List[float]] = [[] for _ in range(len(table))]
    for column_index, header_cell in enumerate(table.header):
        lower_is_better = header_cell.lower_is_better
        if lower_is_better is None:  # column does not have a meaningful ordering
            continue
        value_to_count: Dict[float, int] = defaultdict(int)
        for row in table.rows:
            value = row[column_index].value
            if value is not None:
                value_to_count[value] += 1
        value_to_wins: Dict[float, float] = {}
        acc_count = 0
        for value, value_count in sorted(value_to_count.items(), reverse=lower_is_better):
            value_to_wins[value] = acc_count + ((value_count - 1) / 2)
            acc_count += value_count
        total_count = acc_count
        if total_count < 2:
            continue
        for row_index, row in enumerate(table.rows):
            value = row[column_index].value
            if value is not None:
                win_rates_per_row[row_index].append(value_to_wins[row[column_index].value] / (total_count - 1))

    # Note: the logic up to here is somewhat general as it simply computes win rates across columns for each row.
    # Here, we simply average these win rates but we might want some more involved later (e.g., weighted average).
    aggregate_win_rates: List[Optional[float]] = []
    for win_rates in win_rates_per_row:
        if len(win_rates) == 0:
            aggregate_win_rates.append(None)
        else:
            aggregate = mean(win_rates) if aggregation == "mean" else median(win_rates)
            aggregate_win_rates.append(aggregate)
    print(aggregate_win_rates)
    return aggregate_win_rates

# def new_win_rate_column(csv_path, new_csv_path):
#     """
#     Calculate pairwise win rates between each row and all other rows.
#     Updates the mean_win_rate column with the new aggregated win rates.
    
#     Args:
#         csv_path: Path to CSV file containing mean_win_rate column
#     Returns:
#         List of new win rates for each row
#     """
#     import pandas as pd
    
#     # Read CSV file
#     df = pd.read_csv(csv_path)
    
#     numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
#     numeric_cols = numeric_cols[numeric_cols != 'Mean win rate']
#     print(len(numeric_cols))
#     new_rates = []
    
#     # Calculate win rates for each row
#     for i in range(len(df)):
#         total_wins = 0
#         total_comparisons = 0
        
#         # Compare against all other rows
#         for j in range(len(df)):
#             if i == j:
#                 continue
                
#             # Compare each numeric column
#             for col in numeric_cols:
#                 val_i = df.iloc[i][col]
#                 val_j = df.iloc[j][col]
                
#                 # Skip if either value is missing
#                 if pd.isna(val_i) or pd.isna(val_j):
#                     continue
                    
#                 # Add win if i scores higher than j
#                 if val_i > val_j:
#                     total_wins += 1
#                 total_comparisons += 1
        
#         # Calculate win rate for this row
#         win_rate = total_wins / total_comparisons if total_comparisons > 0 else 0
#         print(f"Row {i} win rate: {win_rate}")
#         win_rate = round(win_rate, 2)
#         new_rates.append(win_rate)
        
#     df['Mean win rate'] = new_rates
#     df.to_csv(new_csv_path, index=False)

def new_leaderboard_csv():

    def normalize(s):
        """Lowercase, remove whitespace, and non-alphanum for fuzzy matching."""
        try:    
            return re.sub(r'[^a-zA-Z0-9]', '', s.lower())
        except:
            print(s)

    # Load JSON
    with open('../medhelm/src/medhelm/refusal_rate_zip.json', 'r') as f:
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
        sorry_rate = json_value.get('sorry')
        not_sorry_rate = json_value.get('not_sorry')
        
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
                # Write to score changes CSV
                with open('../medhelm/data/score_change_0604.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    if f.tell() == 0:  # Write header if file is empty
                        writer.writerow(['Model', 'Dataset', 'Original Score', 'New Score', 'Change', 'Refusal Rate Raw', 'Refusal Rate Ratio'])
                    writer.writerow([
                        model_name,
                        col_part, 
                        new_rows[i][col_idx],
                        str(round(score, 3)),
                        round(score - float(new_rows[i][col_idx] or 0), 3),
                        sorry_rate,
                        sorry_rate / (sorry_rate + not_sorry_rate)
                    ])
                print("UPDATING SCORE:", model_name, "->", col_part, "FROM", new_rows[i][col_idx], "TO", str(round(score, 3)))
                new_rows[i][col_idx] = str(round(score, 3))
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
    # print("\n====== Step 1: ZIP File Processing ======")
    zip_results = extract_and_process_zip(zip_path)
    print("\n📊 Results from ZIP:")
    print(json.dumps(zip_results, indent=2))
    json.dump(zip_results, open("../medhelm/src/medhelm/refusal_rate_zip.json", "w"))
    breakpoint()

    new_leaderboard_csv() 
    # new_win_rate_column('../medhelm/data/leaderboard.csv')
    # new_win_rate_column('../medhelm/data/leaderboard_without_refusals.csv', '../medhelm/data/leaderboard_without_refusals_win_rates.csv')
