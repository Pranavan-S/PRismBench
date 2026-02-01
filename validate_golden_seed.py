import pandas as pd
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import sys

# Import existing libraries from the repo
from llm_oracle_labeling.llm_client import query_gemma, query_llama, query_mistral
from llm_oracle_labeling.data_loader import load_ml_features_dataset, lookup_pr_details
from llm_oracle_labeling.prompt_builder import build_prompt
from llm_oracle_labeling.config import RISK_TYPE_LABELS

# ==========================================
# CONFIGURATION
# ==========================================

# Path to your Ground Truth files
LABELED_TRAIN_PATH ="SamplingLoopData/loop_0_data/labeled_train_data.csv"
LABELED_TEST_PATH ="SamplingLoopData/loop_0_data/labeled_test_data.csv"

# Path to the features file (to get PR title, description, SZZ issues)
ML_FEATURES_PATH = "ML_Label_Input_apache_kafka.csv"

# Map CSV columns in labeled_train_data.csv to the LLM output strings in RISK_TYPE_LABELS
# Based on logic in extract_and_assign_labels.py
COLUMN_TO_LABEL_MAP = {
    "bug": RISK_TYPE_LABELS[0],                            # Bug Risk
    "security": RISK_TYPE_LABELS[1],                       # Security Risk
    "performance": RISK_TYPE_LABELS[2],                    # Performance Risk
    "code_quality_or_maintenability": RISK_TYPE_LABELS[3], # Maintainability Risk
    "non_risky": RISK_TYPE_LABELS[4]                       # Non-risky
}

MODELS = [
    {'name': 'Gemma', 'fn': query_gemma},
    {'name': 'Llama', 'fn': query_llama},
    {'name': 'Mistral', 'fn': query_mistral}
]

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def parse_llm_response(response_data):
    """
    Parses the LLM JSON response safely to extract 'risk_type_labels'.
    """
    try:
        if isinstance(response_data, dict):
            data = response_data
        else:
            # Cleanup markdown code blocks if present
            clean_str = response_data.strip()
            if clean_str.startswith("```json"):
                clean_str = clean_str[7:]
            if clean_str.startswith("```"):
                clean_str = clean_str[3:]
            if clean_str.endswith("```"):
                clean_str = clean_str[:-3]
            
            data = json.loads(clean_str.strip())

        labels = data.get("risk_type_labels", [])
        
        if isinstance(labels, str):
            labels = [labels]
        return labels
    except Exception as e:
        # print(f"  [!] JSON Parse Error: {e}")
        return []

def main():
    # 1. Load Data
    print(f"Loading Labeled Train from: {LABELED_TRAIN_PATH}")
    train_df = pd.read_csv(LABELED_TRAIN_PATH)
    
    print(f"Loading Labeled Test from: {LABELED_TEST_PATH}")
    if Path(LABELED_TEST_PATH).exists():
        test_df = pd.read_csv(LABELED_TEST_PATH)
        golden_df = pd.concat([train_df, test_df], ignore_index=True)
    else:
        print(f"WARNING: {LABELED_TEST_PATH} not found. Using only train data.")
        golden_df = train_df

    print(f"Loading ML Features from: {ML_FEATURES_PATH}")
    ml_features_df = load_ml_features_dataset(ML_FEATURES_PATH)
    
    print(f"Found {len(golden_df)} labeled PRs in golden seed.")

    # 2. Prepare Ground Truth and Prompts
    eval_data = []
    
    # Pre-fetch details to avoid repeated lookups
    for _, row in golden_df.iterrows():
        pr_number = int(row['pr_number'])
        pr_details = lookup_pr_details(pr_number, ml_features_df)
        
        # Skip if PR details missing (e.g. not in feature file)
        if not pr_details.get('pr_title'):
            continue
            
        # Construct Ground Truth Vector
        # Order: [bug, security, performance, maintainability, non_risky]
        gt_vector = [row[col] for col in COLUMN_TO_LABEL_MAP.keys()]
        
        eval_data.append({
            "pr_number": pr_number,
            "pr_details": pr_details,
            "gt_vector": gt_vector
        })

    print(f"Proceeding with {len(eval_data)} valid PRs for evaluation.")

    # 3. Evaluate Each Model
    target_names = list(COLUMN_TO_LABEL_MAP.keys())

    for model in MODELS:
        print(f"\n{'='*40}")
        print(f"Evaluating Model: {model['name']}")
        print(f"{'='*40}")
        
        y_true = []
        y_pred = []
        prediction_rows = []
        
        for item in tqdm(eval_data, desc=f"Querying {model['name']}"):
            # Generate Prompt using existing library
            prompt = build_prompt(item['pr_details'])
            
            # Query Model
            try:
                response = model['fn'](prompt, item['pr_details'])
                predicted_labels = parse_llm_response(response)
            except Exception as e:
                print(f"Error querying {item['pr_number']}: {e}")
                predicted_labels = []
            
            # Convert text labels to binary vector
            pred_vector = []
            for col_key, label_str in COLUMN_TO_LABEL_MAP.items():
                pred_vector.append(1 if label_str in predicted_labels else 0)
            
            # Store prediction row
            row = {'pr_number': item['pr_number']}
            for idx, col_key in enumerate(COLUMN_TO_LABEL_MAP.keys()):
                row[col_key] = pred_vector[idx]
            prediction_rows.append(row)

            y_true.append(item['gt_vector'])
            y_pred.append(pred_vector)
            
        # Save predictions to CSV
        pd.DataFrame(prediction_rows).to_csv(f"{model['name']}_y_pred.csv", index=False)
        print(f"Saved predictions to {model['name']}_y_pred.csv")

        # 4. Report Results
        print(f"\nResults for {model['name']}:")
        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

if __name__ == "__main__":
    main()