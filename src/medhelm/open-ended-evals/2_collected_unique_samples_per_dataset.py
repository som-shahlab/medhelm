import csv
from collections import defaultdict

def count_unique_instances():
    # Open the CSV file
    with open('medhelm_human_annotation.csv', 'r') as file:
        # Create a CSV reader
        csv_reader = csv.reader(file)
        
        # Skip header row
        next(csv_reader)
        
        # Store unique instance_id+dataset_name combinations for each dataset
        unique_instances = defaultdict(set)
        
        # Process each row
        for row in csv_reader:
            # Extract dataset name and instance ID from the row
            username, specialty, team, entry_id, instance_id, dataset_name, model_key, user_label, accuracy, completeness, clarity = row
            
            # Add the instance_id to the appropriate dataset set
            unique_instances[dataset_name].add(instance_id)
        
        # Print results
        print("Dataset Unique Instance Counts:")
        print("-" * 30)
        for dataset, instances in unique_instances.items():
            print(f"{dataset}: {len(instances)} unique instances")

# Execute the function
if __name__ == "__main__":
    count_unique_instances()
