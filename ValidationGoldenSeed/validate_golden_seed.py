import pandas as pd
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, hamming_loss, cohen_kappa_score
from tqdm import tqdm
import sys

# Add parent directory to sys.path to allow importing from repo root
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import existing libraries from the repo
from llm_oracle_labeling.llm_client import query_gemma, query_llama, query_mistral
from llm_oracle_labeling.data_loader import load_ml_features_dataset, lookup_pr_details
from llm_oracle_labeling.prompt_builder import build_prompt
from llm_oracle_labeling.config import RISK_TYPE_LABELS

# ==========================================
# CONFIGURATION
# ==========================================

# Path to your Ground Truth files
LABELED_TRAIN_PATH =str(project_root)+ "\\SamplingLoopData\\loop_0_data\\labeled_train_data.csv"
LABELED_TEST_PATH =str(project_root)+ "\\SamplingLoopData\\loop_0_data\\labeled_test_data.csv"

# Path to the features file (to get PR title, description, SZZ issues)
ML_FEATURES_PATH = str(project_root)+"\\ML_Label_Input_apache_kafka.csv"

# Map CSV columns in labeled_train_data.csv to the LLM output strings in RISK_TYPE_LABELS
# Based on logic in extract_and_assign_labels.py
COLUMN_TO_LABEL_MAP = {
    "bug": RISK_TYPE_LABELS[0],                            # Bug Risk
    "security": RISK_TYPE_LABELS[1],                       # Security Risk
    "performance": RISK_TYPE_LABELS[2],                    # Performance Risk
    "code_quality_or_maintenability": RISK_TYPE_LABELS[3], # Maintainability Risk
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

    print(f"\n{'='*40}")
    print(f"Evaluating Consensus (Majority Vote of {len(MODELS)} Models)")
    print(f"{'='*40}")

    y_true = []
    y_pred_consensus = []
    majority_exact_match_results = []
    prediction_rows = []

    for item in tqdm(eval_data, desc="Processing PRs"):
        # Generate Prompt using existing library
        prompt = build_prompt(item['pr_details'])
        
        # Initialize votes for this PR: [bug, sec, perf, maint, non_risky]
        vote_counts = np.zeros(len(target_names), dtype=int)
        exact_match_votes = 0

        # Query all models
        for model in MODELS:
            try:
                response = model['fn'](prompt, item['pr_details'])
                predicted_labels = parse_llm_response(response)
            except Exception as e:
                print(f"Error querying {model['name']} for PR {item['pr_number']}: {e}")
                predicted_labels = []
            
            # Convert to vector for this model
            model_pred_vector = []
            for idx, (col_key, label_str) in enumerate(COLUMN_TO_LABEL_MAP.items()):
                val = 1 if label_str in predicted_labels else 0
                model_pred_vector.append(val)
                vote_counts[idx] += val
            
            # Check exact match with Ground Truth (User Request)
            if model_pred_vector == item['gt_vector']:
                exact_match_votes += 1

        # Apply Majority Rule (>= 2 votes)
        consensus_vector = [1 if v >= 2 else 0 for v in vote_counts]

        # Store prediction row
        row = {'pr_number': item['pr_number']}
        for idx, col_key in enumerate(COLUMN_TO_LABEL_MAP.keys()):
            row[col_key] = consensus_vector[idx]
        prediction_rows.append(row)

        y_true.append(item['gt_vector'])
        y_pred_consensus.append(consensus_vector)

    # Save consensus predictions to CSV
    pd.DataFrame(prediction_rows).to_csv("consensus_y_pred.csv", index=False)
    print(f"Saved consensus predictions to consensus_y_pred.csv")

    # 4. Report Results
    print(f"\nConsensus Results:")
    acc = accuracy_score(y_true, y_pred_consensus)
    print(f"Consensus Vector Accuracy (Subset Accuracy): {acc:.4f}")

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred_consensus)

    # Hamming Accuracy = 1 - Hamming Loss
    h_acc = 1 - hamming_loss(y_true_np, y_pred_np)
    print(f"Hamming Accuracy: {h_acc:.4f}")

    print("\nCohen's Kappa per label:")
    for i, name in enumerate(target_names):
        kappa = cohen_kappa_score(y_true_np[:, i], y_pred_np[:, i])
        print(f"  {name}: {kappa:.4f}")
    
    print("\nClassification Report (Consensus Vector):")
    print(classification_report(y_true, y_pred_consensus, target_names=target_names, zero_division=0))

if __name__ == "__main__":
    main()