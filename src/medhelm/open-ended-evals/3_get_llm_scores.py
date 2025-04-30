import os
import json
import csv
import pandas as pd
from pathlib import Path
import traceback

# Function to calculate average scores from display_predictions.json
def calculate_avg_scores(instance_id, dataset_name, model_key):
    # Construct the path to the display_predictions.json file
    base_dir = "MED-HELM-PUBLIC"  # From the file browser image
    
    # From the folder structure image, we can see multiple formats:
    # 1. dataset_name:model=model_key,model_deployment=deployment_name
    # 2. dataset_name/model=model_key,model_deployment=deployment_name
    # Create a list of possible directory patterns to try
    possible_dirs = [
        f"{base_dir}/{dataset_name}:model={model_key}",
        f"{base_dir}/{dataset_name}/model={model_key}",
        f"{base_dir}/{dataset_name}"  # For flexible search within the dataset directory
    ]
    
    try:
        # Try each possible directory pattern
        display_file = None
        
        # First try exact pattern matching
        for dir_pattern in possible_dirs[:2]:  # First two patterns are exact matches
            if os.path.exists(dir_pattern):
                for root, dirs, files in os.walk(dir_pattern):
                    if "display_predictions.json" in files:
                        display_file = os.path.join(root, "display_predictions.json")
                        break
                if display_file:
                    break
        
        # If not found, try flexible search
        if not display_file:
            # Try with a more flexible approach, looking for the model_key in the directory name
            if os.path.exists(f"{base_dir}/{dataset_name}"):
                for root, dirs, files in os.walk(f"{base_dir}/{dataset_name}"):
                    if model_key in root and "display_predictions.json" in files:
                        display_file = os.path.join(root, "display_predictions.json")
                        break
        
        # If still not found, try the most flexible approach
        if not display_file:
            for root, dirs, files in os.walk(base_dir):
                # Look for directories that have both the dataset_name and model_key in their path
                if (dataset_name in root or dataset_name.replace("_", "-") in root) and \
                   (model_key in root or model_key.replace("-", "_") in root) and \
                   "display_predictions.json" in files:
                    display_file = os.path.join(root, "display_predictions.json")
                    break
        
        # Debugging output to help identify successful matches
        if display_file:
            print(f"Found file for {instance_id} at {display_file}")
            
        if not display_file:
            # Try searching the entire base directory as a last resort
            matched_files = []
            for root, dirs, files in os.walk(base_dir):
                if "display_predictions.json" in files:
                    file_path = os.path.join(root, "display_predictions.json")
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            for entry in data:
                                if entry.get("instance_id") == instance_id:
                                    matched_files.append(file_path)
                                    break
                    except:
                        pass  # Skip files that can't be read
            
            if matched_files:
                display_file = matched_files[0]  # Use the first matching file
                print(f"Found instance {instance_id} via content search at {display_file}")
            else:
                return None, None
                
        # Load the JSON data
        with open(display_file, 'r') as f:
            data = json.load(f)
        
        # Find the entry with the matching instance_id
        for entry in data:
            if entry.get("instance_id") == instance_id:
                # Calculate average scores for all three models
                scores = {}
                metrics = {
                    "accuracy": [],
                    "completeness": [],
                    "clarity": []
                }
                
                # From the paste.txt document, we can see the annotations structure
                for model_type in ["gpt", "llama", "claude"]:
                    # Check if model evaluation exists in annotations
                    annotations = entry.get("annotations", {})
                    if "aci_bench" in annotations:
                        if model_type in annotations["aci_bench"]:
                            model_data = annotations["aci_bench"][model_type]
                            
                            accuracy = model_data.get("accuracy", {}).get("score", 0)
                            completeness = model_data.get("completeness", {}).get("score", 0)
                            clarity = model_data.get("clarity", {}).get("score", 0)
                            
                            # Add to metrics lists for averaging later
                            metrics["accuracy"].append(accuracy)
                            metrics["completeness"].append(completeness)
                            metrics["clarity"].append(clarity)
                            
                            # Calculate average for this model
                            avg_score = (accuracy + completeness + clarity) / 3
                            scores[model_type] = avg_score
                    elif "medi_qa" in annotations:
                        if model_type in annotations["medi_qa"]:
                            model_data = annotations["medi_qa"][model_type]
                            
                            accuracy = model_data.get("accuracy", {}).get("score", 0)
                            completeness = model_data.get("completeness", {}).get("score", 0)
                            clarity = model_data.get("clarity", {}).get("score", 0)
                            
                            # Add to metrics lists for averaging later
                            metrics["accuracy"].append(accuracy)
                            metrics["completeness"].append(completeness)
                            metrics["clarity"].append(clarity)
                            
                            # Calculate average for this model
                            avg_score = (accuracy + completeness + clarity) / 3
                            scores[model_type] = avg_score
                    elif "mtsamples_replicate" in annotations:
                        if model_type in annotations["mtsamples_replicate"]:
                            model_data = annotations["mtsamples_replicate"][model_type]
                            
                            accuracy = model_data.get("accuracy", {}).get("score", 0)
                            completeness = model_data.get("completeness", {}).get("score", 0)
                            clarity = model_data.get("clarity", {}).get("score", 0)
                            
                            # Add to metrics lists for averaging later
                            metrics["accuracy"].append(accuracy)
                            metrics["completeness"].append(completeness)
                            metrics["clarity"].append(clarity)
                            
                            # Calculate average for this model
                            avg_score = (accuracy + completeness + clarity) / 3
                            scores[model_type] = avg_score
                
                # Calculate cross-model averages
                avg_metrics = {}
                for metric_name, values in metrics.items():
                    if values:
                        avg_metrics[metric_name] = sum(values) / len(values)
                    else:
                        avg_metrics[metric_name] = 0
                
                return scores, avg_metrics
        
        return None, None
        
    except Exception as e:
        return None, None


# Main function to process the CSV and display results
def main():
    # Based on the example CSV data provided
    csv_file = "medhelm_human_annotation.csv"  # Update with your actual CSV filename
    output_csv = "output.csv"  # New CSV with added columns
    base_dir = "MED-HELM-PUBLIC"
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        print(f"Loaded CSV with {len(df)} rows")
        
        # Add new columns to the dataframe
        df["llm_accuracy"] = float('nan')
        df["llm_completeness"] = float('nan')
        df["llm_clarity"] = float('nan')
        
        # Check all available dataset directories to help with debugging
        if os.path.exists(base_dir):
            print(f"Found base directory: {base_dir}")
            print("Available dataset directories:")
            for item in os.listdir(base_dir):
                if os.path.isdir(os.path.join(base_dir, item)):
                    print(f"  - {item}")
        
        # Initialize counters
        found_count = 0
        datasets_found = set()
        
        # Process each row to extract information
        for index, row in df.iterrows():
            # Extract required values
            instance_id = row.get("instance_id")
            dataset_name = row.get("dataset_name")
            model_key = row.get("model_key")
            
            if not (instance_id and dataset_name and model_key):
                continue  # Skip rows with missing data
            
            # Calculate average scores from the JSON files
            scores, avg_metrics = calculate_avg_scores(instance_id, dataset_name, model_key)
            
            # Only update the dataframe if we found matching data in the JSON files
            if scores and avg_metrics:
                found_count += 1
                datasets_found.add(dataset_name)
                
                # Update the dataframe with cross-model average metrics
                df.at[index, "llm_accuracy"] = avg_metrics["accuracy"]
                df.at[index, "llm_completeness"] = avg_metrics["completeness"]
                df.at[index, "llm_clarity"] = avg_metrics["clarity"]
        
        # Save the updated dataframe to a new CSV
        df.to_csv(output_csv, index=False)
        
        print(f"\nFound matches for {found_count} out of {len(df)} entries in the CSV.")
        print(f"Datasets with successful matches: {', '.join(sorted(datasets_found))}")
        print(f"Created new CSV with added columns: {output_csv}")
    
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
