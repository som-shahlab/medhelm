import os
import json
from medhelm.utils.constants import BENCHMARKS, BENCHMARK_NAME_MAPPING, MODEL_NAME_MAPPING
import numpy as np

#get scores for model A and B
def get_scores(path: str, model: str)-> dict:
    # return the number of examples in a given dataset
    results={}
    # Search private path
    folders = os.listdir(path)
    dirs = [os.path.join(path, folder) for folder in folders]
    for curr_dir in dirs:
        if model in curr_dir:
            stats_path = os.path.join(curr_dir, "display_predictions.json")
            with open(stats_path) as f:
                stats = json.load(f)
                name= curr_dir.split("/")[-1].split(":")[0]
                if "med_dialog" in name:
                    name=curr_dir.split("/")[-1].split(",")[0]
                    # Get the last entry that has num_instances
                if name not in results.keys():
                    results[name] = []
                for stat in stats:
                    try:
                        metric= list(stat["stats"].keys())[-1]
                        if name=="medcalc_bench":
                            metric="medcalc_bench_accuracy"
                        if name=="ehr_sql":
                            metric="ehr_sql_execution_accuracy"
                        if name=="medec":
                            metric="medec_error_flag_accuracy"
                        score = stat["stats"][metric]
                        if score>1:
                            score=score/5
                        results[name].append(score)
                    except:
                        continue
    return results

def collapse_into_category(scores: dict)-> dict:
    collapsed_scores = {}
    for category, subcategories in BENCHMARKS.items():
        collapsed_scores[category] = []
        for subcat, benchmarks in subcategories.items():
            for benchmark in benchmarks:
                if benchmark in scores.keys():
                    collapsed_scores[category].extend(scores[benchmark])
    return collapsed_scores


# # Join the outputs from both paths
def join_outputs(*dictionaries)-> dict:
    # Combine multiple dictionaries into one
    all_examples = {}
    for dictionary in dictionaries:
        all_examples.update(dictionary)
    
    return all_examples

#find standard deviation of scores

def get_std(scores_A: dict, scores_B: dict)-> dict:
    score_differences_std = {}
    for dataset in scores_A.keys():
        if len(scores_A[dataset]) != len(scores_B[dataset]):
            print("Dataset length mismatch: ", dataset)
            print(len(scores_A[dataset]), len(scores_B[dataset]))
            continue
        assert len(scores_A[dataset]) == len(scores_B[dataset])
        score_differences_std[dataset] = np.std([scores_A[dataset][i] - scores_B[dataset][i] for i in range(len(scores_A[dataset]))])
    return score_differences_std

#find number of examples

def get_n_examples(scores_A: dict)-> dict:
    n_examples = {}
    for dataset in scores_A.keys():
        n_examples[dataset] = len(scores_A[dataset])
    return n_examples

#find minimum detectable effect size
def get_MDE(score_differences_std: dict, n_examples: dict)-> dict:
    MDE = {}
    z_significance = 1.96
    z_power = 0.84

    for dataset in score_differences_std.keys():
        MDE[dataset] = (z_power + z_significance) * score_differences_std[dataset] / np.sqrt(n_examples[dataset])
    return MDE

def compile_all_scores(path_private: str, path_private_reasoning: str, path_public: str, path_public_reasoning: str, model:str)-> dict:
    scores_private = get_scores(path_private, model)
    scores_private_reasoning = get_scores(path_private_reasoning, model)
    scores_public = get_scores(path_public, model)
    scores_public_reasoning = get_scores(path_public_reasoning, model)
    scores_compiled = join_outputs(scores_private, scores_private_reasoning, scores_public, scores_public_reasoning)
    return scores_compiled

def print_results(results: dict, collapse: bool):
    if collapse:
        for key in results.keys():
            print(BENCHMARK_NAME_MAPPING[key])
            print(np.mean(results[key]),"+-", np.std(results[key]))
    else:
        for key in results.keys():
            print(BENCHMARK_NAME_MAPPING[key])
            print(np.mean(results[key]),"+-", np.std(results[key]))

def main():
    collapse = False
    MODELS = ["deepseek-ai_deepseek-r1", "openai_o3-mini-2025-01-31", "openai_gpt-4o-mini-2024-07-18", "openai_gpt-4o-2024-05-13", "anthropic_claude-3-5-sonnet-20241022", "anthropic_claude-3-7-sonnet-20250219", "google_gemini-1.5-pro-001", "google_gemini-2.0-flash-001", "meta_llama-3.3-70b-instruct"]
    ##Doesn't match the constants.py file
    # MODELS= list(MODEL_NAME_MAPPING.keys())
    path_private="/share/pi/nigam/users/aunell/medhelm/data/benchmark_output/runs/MED-HELM-PRIVATE"
    path_private_reasoning="/share/pi/nigam/users/aunell/medhelm/data/benchmark_output/runs/MED-HELM-PRIVATE-REASONING"
    path_public="/share/pi/nigam/users/aunell/medhelm/data/benchmark_output/runs/MED-HELM-PUBLIC"
    path_public_reasoning="/share/pi/nigam/users/aunell/medhelm/data/benchmark_output/runs/MED-HELM-PUBLIC-REASONING"

    MDE_list_by_model_pair = []
    model_pairs = [(MODELS[i], MODELS[j]) for i in range(len(MODELS)) for j in range(i+1, len(MODELS))]
    for MODEL_A, MODEL_B in model_pairs:
        #compile scores for each model
        scores_A = compile_all_scores(path_private, path_private_reasoning, path_public, path_public_reasoning, MODEL_A)
        scores_B = compile_all_scores(path_private, path_private_reasoning, path_public, path_public_reasoning, MODEL_B)

        #collapse scores into categories IF collapse is True
        if collapse:
            scores_A = collapse_into_category(scores_A)
            scores_B = collapse_into_category(scores_B)

        #get standard deviation of score differences    
        score_differences_std = get_std(scores_A, scores_B)

        #get number of examples
        n_examples_A = get_n_examples(scores_A)
        n_examples_B = get_n_examples(scores_B)
        assert n_examples_A == n_examples_B

        #get minimum detectable effect size
        MDE = get_MDE(score_differences_std, n_examples_A)

        #append MDE for each model pair
        MDE_list_by_model_pair.append(MDE)

    #compile results for each dataset
    final_results = {}

    #iterate through datasets
    for key in MDE_list_by_model_pair[0].keys(): 
        final_results[key] = []

        #iterate through model pairs
        for results in MDE_list_by_model_pair:

            #append MDE for each model pair for a given dataset
            final_results[key].append(results[key]) 

    print_results(final_results, collapse)

#with the number of examples, return the minimum detectable error
if __name__ == "__main__":
    main()