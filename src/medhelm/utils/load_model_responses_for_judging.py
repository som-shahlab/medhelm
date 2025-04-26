import zipfile
from pathlib import Path
import json
import pandas as pd

def find_model_response_directories(zip_path: str, target_dataset: str) -> list:
    """
    Find directories in a zip file containing model responses for a specific dataset.
    
    Args:
        zip_path (str): Path to the zip file containing benchmark results
        target_dataset (str): Dataset identifier to filter paths (e.g., 'aci')
        
    Returns:
        list: Sorted list of unique directory paths containing model responses
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List all files in zip
            all_files = zip_ref.namelist()
            
            # Filter for paths containing both 'aci' and 'model'
            aci_files = [f for f in all_files if target_dataset in f and 'model' in f]
            
            # Get unique directories by taking parent paths
            unique_paths = {str(Path(f).parent) for f in aci_files if 'model' in str(Path(f).parent)}
            
            print(f"Found {len(unique_paths)} unique paths containing '{target_dataset}' and 'model':")
            for path in sorted(unique_paths):
                print(path)
            
            return sorted(unique_paths)
                    
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid zip file")
        return []
    except Exception as e:
        print(f"Error processing zip file: {str(e)}")
        return []

def add_model_name_to_df(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Add a new column to the dataframe for the model's predictions.
    
    Args:
        df (pd.DataFrame): DataFrame containing benchmark results
        path (str): Path containing model identifier
        
    Returns:
        pd.DataFrame: DataFrame with new column for model predictions
    """
    model_name = Path(path).name.split("model=")[-1].split(",")[0]
    df[model_name] = None  # Add new column with model name
    return df

def add_ground_truth_to_df(df: pd.DataFrame, path: str):
    ground_truth_path = str(Path(path) / "scenario_state.json")

def add_prompt_data_to_df(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Add prompts and reference answers to the dataframe from benchmark files.
    
    Args:
        df (pd.DataFrame): DataFrame to populate with prompts and references
        path (str): Path to directory containing benchmark files
        
    Returns:
        pd.DataFrame: DataFrame with added prompts and reference answers
    """
    model_name = Path(path).name.split("model=")[-1].split(",")[0]
    requests_path = str(Path(path) / "display_requests.json")
    ground_truth_path = str(Path(path) / "scenario_state.json")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open(requests_path) as f:
            requests = json.load(f)
        with zip_ref.open(ground_truth_path) as f:
            ground_truth = json.load(f)

    for request in requests:
        instance_id = request["instance_id"]
        prompt = request["request"]["prompt"]
        if instance_id not in df.index:
            df.loc[instance_id, 'instance_id'] = instance_id
            df.loc[instance_id, 'prompt'] = prompt
        df.loc[instance_id, model_name] = prompt
    
    for instance in ground_truth['request_states']:
        instance_id = instance["instance"]["id"]
        reference = instance['instance']['references'][0]['output']
        if instance_id in df.index:
            df.loc[instance_id, 'reference'] = reference['text']
        else:
            print(f"Instance {instance_id} not found in dataframe")
    return df

def add_predictions_to_df(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Add model predictions to the dataframe from prediction files.
    
    Args:
        df (pd.DataFrame): DataFrame to populate with model predictions
        path (str): Path to directory containing prediction files
        
    Returns:
        pd.DataFrame: DataFrame with added model predictions
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        predictions_path = str(Path(path) / "display_predictions.json")
        with zip_ref.open(predictions_path) as f:
            predictions = json.load(f)
            
        # Extract model name from path
        model_name = Path(path).name.split("model=")[-1].split(",")[0]
        for instance in predictions:
            instance_id = instance["instance_id"]
            
            # Find matching prediction
            prediction = next(p["predicted_text"] for p in predictions 
                            if p["instance_id"] == instance_id)
            df.loc[instance_id, model_name] = prediction
            
    return df

if __name__ == "__main__":
    NUMBER_OF_MODELS = 6
    zip_path = "/share/pi/nigam/data/medhelm/release/v1/benchmark_output_unredacted_20250225_005715.zip"
    paths = find_model_response_directories(zip_path, target_dataset="aci") 
    assert len(paths) == NUMBER_OF_MODELS, f"Expected {NUMBER_OF_MODELS} paths, got {len(paths)}"
    df = pd.DataFrame(columns=['instance_id', 'prompt', 'reference'])
    df = add_prompt_data_to_df(df, paths[0]) #prompts and references are the same for all models
    for path in paths:
        df = add_model_name_to_df(df, path)
        df = add_predictions_to_df(df, path)

    df.to_csv('model_responses.csv')
