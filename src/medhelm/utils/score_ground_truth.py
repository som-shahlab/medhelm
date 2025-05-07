import os
import requests
from typing import Dict, Any
import time
import signal
import json
import pandas as pd
import math
import openai
import argparse

OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DEPLOYMENT_NAME = "gpt-4o-mini"
API_VERSION = "2023-05-15"
print(f"OPENAI_API_BASE: {OPENAI_API_BASE}")
print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")

def timeout_handler(signum, frame):
    raise TimeoutError("Evaluation timed out")

def completion_with_backoff(**kwargs) -> Dict[str, Any]:
    retry_count = 0
    while True:
        retry_count += 1
        try:
            url = f"{OPENAI_API_BASE}/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version={API_VERSION}"
            headers = {
                "Ocp-Apim-Subscription-Key": OPENAI_API_KEY,
                "Content-Type": 'application/json'
            }
            data = {
                "messages": kwargs['messages'],
                "max_tokens": kwargs.get('max_tokens', 1000),
                "temperature": kwargs.get('temperature', 0.7)
            }
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as error:
            print(f"Error: {error}")
            if retry_count > 30:
                return {}
            time.sleep(10)

def evaluate_text(input_text: str, reference: str) -> Dict[str, Any]:
    if input_text is None or (isinstance(input_text, float) and math.isnan(input_text)):
        print("Warning: Input text is None or NaN")
        input_text = "[No text provided]"
    elif not isinstance(input_text, str):
        print("Warning: Input text is not a string. Attempting to convert.")
        try:
            input_text = str(input_text)
        except Exception as e:
            print(f"Error converting input text to string: {e}")
            input_text = "[Error: Unable to process text]"

    prompt = create_prompt(input_text, reference)
    
    response = completion_with_backoff(
        messages=[
            {"role": "system", "content": f"You are an AI assistant tasked with evaluating a response to a prompt."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10,
        temperature=0.2,
    )
    
    try:
        evaluation = json.loads(response['choices'][0]['message']['content'])
        return evaluation
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing LLM response: {e}")
        print("Raw response:")
        print(response)
        return None

def create_prompt(input_text: str, reference: str) -> str:
    return f"Evaluate the following text: {input_text} against the following reference: {reference}. Return a score of 0 if the text is not a good response to the prompt, and a score of 1 if it is a reasonable response to the prompt. Only give a  score of 0 if the reference does not make sense with the prompt. Only return the score, no other text."

def main():
    # Load the gold standard data
    results = {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dataset", type=str, default="mtsamples_replicate",
                      help="Target dataset to process")
    args = parser.parse_args()
    gold_standard_path = f"../medhelm/data/filtering/model_responses_{args.target_dataset}.csv"
    gold_standard_df = pd.read_csv(gold_standard_path)
    for index, row in gold_standard_df.iterrows():
        instance_id = row['instance_id']
        prompt = row['prompt']
        reference = row['reference']
        evaluation = evaluate_text(prompt, reference)
        results[instance_id] = evaluation
        with open(f'../medhelm/data/filtering/gold_standard_evaluations_{args.target_dataset}.json', 'w') as file:
            json.dump(results, file, indent=4)

if __name__ == "__main__":
    main()