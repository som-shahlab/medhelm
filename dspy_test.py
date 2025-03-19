import os
import json
import pandas as pd
import dspy
import csv
from typing import Literal
from datasets import load_dataset
from dspy.teleprompt import *
from dspy.evaluate import Evaluate

# Configure LM
lm = dspy.LM('openai/gpt-4o', api_base='http://0.0.0.0:4000')
dspy.configure(lm=lm)

class Categorize(dspy.Signature):
    """Classify PubMedQA questions."""
    question: str = dspy.InputField()
    category: Literal["yes", "no", "maybe"] = dspy.OutputField()

def validate_category(example, prediction, trace=None):
    return prediction.category == example.category

def load_pubmedqa_data(num_samples=100):
    """Load and prepare PubMedQA dataset"""
    dataset = load_dataset("pubmed_qa", "pqa_labeled")
    questions = pd.DataFrame({
        'question': dataset['train']['question'],
        'final_decision': dataset['train']['final_decision']
    })
    return questions.head(num_samples)

def load_trainset(csv_path):
    """Load training data from CSV file"""
    trainset = []
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            example = dspy.Example(question=row['question'], 
                                 category=row['final_decision']).with_inputs("question")
            trainset.append(example)
    return trainset

def classify_questions(questions_df, classifier):
    """Apply classification to questions"""
    def apply_classify(question):
        try:
            result = classifier(question=question)
            return result.category
        except Exception as e:
            print(f"Error classifying question: {e}")
            return "error"
            
    with dspy.context(lm=lm):
        questions_df['category'] = questions_df['question'].apply(apply_classify)
    return questions_df

def optimize_classifier(base_classifier, trainset):
    """Optimize the classifier using MIPROv2"""
    tp = dspy.MIPROv2(metric=validate_category, auto="light", 
                      prompt_model=lm, task_model=lm)
    # tp=dspy.BootstrapFewShotWithRandomSearch(metric=validate_category)
    return tp.compile(base_classifier, trainset=trainset, 
                     max_labeled_demos=0, max_bootstrapped_demos=0) #zero shot

def evaluate_classifier(classifier, trainset):
    """Evaluate classifier performance"""
    evaluator = Evaluate(devset=trainset, num_threads=1, 
                        display_progress=True, display_table=5)
    return evaluator(classifier, metric=validate_category)

def save_results(df, filename):
    """Save results to CSV"""
    df.to_csv(filename, index=False)

def main(dataset_name="pubmedqa", num_samples=100, optimize=True):
    # Initialize base classifier
    classify = dspy.Predict(Categorize)
    
    # Load and process data
    if dataset_name == "pubmedqa":
        questions = load_pubmedqa_data(num_samples)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    # Initial classification
    pubmedqa_questions = classify_questions(questions, classify)
    save_results(pubmedqa_questions, 'pubmedqa_questions.csv')
    
    # Load training set
    trainset = load_trainset('pubmedqa_questions.csv')
    
    # Evaluate baseline
    print("Evaluating baseline model:")
    baseline_metrics = evaluate_classifier(classify, trainset)
    print(f"Baseline Accuracy: {baseline_metrics}%")
    
    if optimize:
        # Optimize and evaluate
        optimized_classify = optimize_classifier(classify, trainset)
        
        # Save optimized model
        save_path = dataset_name + "_prompt.json"
        optimized_classify.save(save_path)
        
        # Run and evaluate optimized classifier
        pubmedqa_questions = classify_questions(questions, optimized_classify)
        
        print("\nEvaluating optimized model:")
        optimized_metrics = evaluate_classifier(optimized_classify, trainset)
        print(f"Optimized Accuracy: {optimized_metrics}%")
        print(f"\nAccuracy improvement: {(optimized_metrics - baseline_metrics)}%")

if __name__ == "__main__":
    main(dataset_name="pubmedqa", num_samples=10, optimize=True)