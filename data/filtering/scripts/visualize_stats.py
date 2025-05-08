import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the stats JSON files
stats_dir = "../medhelm/data/filtering/stats"

# Collect all stats
all_stats = {}
for fname in os.listdir(stats_dir):
    if fname.endswith(".json"):
        dataset = fname.replace("stats_", "").replace(".json", "")
        with open(os.path.join(stats_dir, fname), "r") as f:
            # Fix for NaN in JSON (replace with null, then parse)
            content = f.read().replace("NaN", "null")
            stats = json.loads(content)
            all_stats[dataset] = stats

# What stages do we have?
stages = ["before_filtering", "after_filtering_bad", "after_filtering_good", "after_filtering_double_flagged"]
metrics = ["mean", "median", "std_dev"]

# Prepare data for plotting
datasets = sorted(all_stats.keys())
stage_labels = {
    "before_filtering": "Before Filtering",
    "after_filtering_bad": "After Filtering (Bad)", 
    "after_filtering_good": "After Filtering (Good)",
    "after_filtering_double_flagged": "After Filtering (Double-Flagged)"
}

for metric in metrics:
    plt.figure(figsize=(12, 6))
    width = 0.2
    x = np.arange(len(datasets))
    
    for i, stage in enumerate(stages):
        values = []
        n_examples_list = []
        for dataset in datasets:
            val = all_stats[dataset].get(stage, {}).get(metric)
            n_examples = all_stats[dataset].get(stage, {}).get("n_examples")
            if val is None:
                breakpoint()
            values.append(val)
            n_examples_list.append(n_examples)
            
        bars = plt.bar(x + i*width, values, width=width, label=stage_labels[stage])
        
        # Add n_examples as text above each bar
        for idx, (rect, n) in enumerate(zip(bars, n_examples_list)):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height,
                    f'n={int(n)}',
                    ha='center', va='bottom', rotation=90)
            
    plt.xticks(x + width*1.5, datasets, rotation=45)
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f"{metric.replace('_', ' ').title()} Across Datasets and Filtering Stages")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"../medhelm/data/filtering/{metric.replace('_', ' ').title()}_across_datasets_and_filtering_stages.png")