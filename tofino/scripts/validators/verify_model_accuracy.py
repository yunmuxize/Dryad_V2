# -*- coding:utf-8 -*-
"""
Model Accuracy Verification Script
Validates extracted samples against the exported JSON decision tree model.
Calculates Accuracy, Precision, Recall, and F1 Score (Macro-average).
"""

import json
import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score

def predict_from_json(node, features, feature_list):
    """
    Traverse the JSON decision tree to get a prediction.
    """
    # If it's a leaf node (no children)
    if "children" not in node:
        # Return the class with the highest value
        return int(np.argmax(node["value"]))

    # Internal node: get feature name and threshold
    feature_name = node["feature"]
    threshold = float(node["threshold"])
    
    # Get the value for this feature from the input features list
    feature_idx = feature_list.index(feature_name)
    feature_value = float(features[feature_idx])
    
    # Decision: Children[0] is <=, Children[1] is >
    if feature_value <= threshold:
        return predict_from_json(node["children"][0], features, feature_list)
    else:
        return predict_from_json(node["children"][1], features, feature_list)

def main():
    # Path configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # tofino/
    model_path = os.path.join(base_dir, "data", "iscx_depth_5_model.json")
    samples_path = os.path.join(base_dir, "data", "iscx_validation_samples_100.json")

    print("=" * 60)
    print("Decision Tree Model Accuracy Verification")
    print("=" * 60)

    # 1. Load Model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    with open(model_path, 'r', encoding='utf-8') as f:
        model_data = json.load(f)
    
    tree_root = model_data["tree_structure"]
    feature_list = model_data["feature_list"]
    print(f"Loaded Model: Depth={model_data['depth']}, Features={len(feature_list)}")

    # 2. Load Samples
    if not os.path.exists(samples_path):
        print(f"Error: Samples file not found at {samples_path}")
        return
    
    with open(samples_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    print(f"Loaded {len(samples)} validation samples.")

    # 3. Perform Validation
    y_true = []
    y_pred = []
    py_recorded_preds = [] # To verify against the predictions recorded during extraction

    print("\nValidating samples...")
    for item in samples:
        features = item["features"]
        true_label = int(item["label"])
        recorded_pred = int(item["py_prediction"])
        
        # Calculate current prediction using the traversal logic
        current_pred = predict_from_json(tree_root, features, feature_list)
        
        y_true.append(true_label)
        y_pred.append(current_pred)
        py_recorded_preds.append(recorded_pred)

    # 4. Consistency Check
    # Verify if current script's logic matches the recorded logic from training phase
    consistency = (y_pred == py_recorded_preds)
    print(f"Logic Consistency Check: {'PASS' if consistency else 'FAIL'}")
    if not consistency:
        mismatch_count = sum(1 for i, j in zip(y_pred, py_recorded_preds) if i != j)
        print(f"Warning: {mismatch_count} predictions differ from recorded ones!")

    # 5. Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    print("\n" + "-" * 60)
    print("VALIDATION RESULTS")
    print("-" * 60)
    print(f"Total Samples: {len(samples)}")
    print(f"Accuracy:      {acc:.4f} ({int(acc * len(samples))}/{len(samples)})")
    print(f"Macro F1:      {macro_f1:.4f}")
    
    print("\nDetailed Classification Report:")
    # Using specific digits for precision
    print(classification_report(y_true, y_pred, digits=4))
    print("-" * 60)

if __name__ == '__main__':
    main()
