# -*- coding:utf-8 -*-
"""
Export Depth 5 Model and Extract Validation Samples
Targets:
1. Export ISCX Decision Tree (Depth 5) to Tofino data path.
2. Extract 100 samples with 8 core features for cross-verification.
"""

import os
import sys
import json
import pickle
import numpy as np
from sklearn import tree as st

# Ensure we can import from optimization
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimization import *

# Match Type Combination found: [3, 1, 1, 1, 2, 2, 1, 0]
# 3: PREFIX, 1: TERNARY, 2: RANGE, 0: EXACT
DEPTH_5_MATCH_TYPES = [3, 1, 1, 1, 2, 2, 1, 0]

FEATURE_LIST = [
    'Total length', 'Protocol', 'IPV4 Flags (DF)', 'Time to live',
    'Src Port', 'Dst Port', 'TCP flags (Reset)', 'TCP flags (Syn)'
]

def main():
    print("=" * 80)
    print("Exporting Depth 5 Model and Extracting Samples")
    print("=" * 80)
    
    # Path setup
    src_dir = os.path.dirname(os.path.abspath(__file__)) # Dryad/src
    project_root = os.path.dirname(src_dir) # Dryad/Dryad
    workspace_root = os.path.dirname(project_root) # Dryad_V2/Dryad
    
    model_data_path = os.path.join(project_root, "model_data", "iscx")
    tofino_data_path = "C:\\Users\\86177\\OneDrive\\Desktop\\Dryad_V2\\Dryad\\tofino\\data"
    
    if not os.path.exists(tofino_data_path):
        os.makedirs(tofino_data_path)
        print(f"Created directory: {tofino_data_path}")

    # 1. Load Data
    print("\nLoading ISCX Dataset...")
    with open(os.path.join(model_data_path, "data_train_iscx_C.pkl"), "rb") as f:
        train_data = pickle.load(f)
    with open(os.path.join(model_data_path, "data_eval_iscx_C.pkl"), "rb") as f:
        test_data = pickle.load(f)
    
    # Feature alignment: [Total(9), Proto(0), IPv4Flag(1), TTL(4), Port1(7), Port2(8), TcpF1(2), TcpF2(3)]
    target_indices = [9, 0, 1, 4, 7, 8, 2, 3]
    x_train = train_data[:, target_indices]
    y_train = train_data[:, -1].astype(int)
    x_test = test_data[:, target_indices]
    y_test = test_data[:, -1].astype(int)
    class_names = np.array(['0', '1', '2', '3', '4', '5'])

    # 2. Train and Export Model
    print("\nTraining Depth 5 Decision Tree...")
    model = st.DecisionTreeClassifier(max_depth=5, random_state=5)
    model.fit(x_train, y_train)
    
    json_model = sklearn2json(model, FEATURE_LIST, class_names)
    json_model = hard_prune(json_model, 0, 5)
    json_model = soft_prune(json_model)
    
    # Convert with specific match types
    converted_tree, rule_stats = convert_tree_to_match_types(
        json_model, DEPTH_5_MATCH_TYPES, FEATURE_LIST
    )
    
    # Add metadata
    export_data = {
        "model_type": "DecisionTree",
        "depth": 5,
        "feature_list": FEATURE_LIST,
        "match_types": DEPTH_5_MATCH_TYPES,
        "tree_structure": converted_tree,
        "class_names": class_names.tolist()
    }
    
    model_export_path = os.path.join(tofino_data_path, "iscx_depth_5_model.json")
    with open(model_export_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    print(f"Model exported to: {model_export_path}")

    # 3. Extract Validation Samples
    print("\nExtracting 100 validation samples...")
    # Randomly select 100 indices from test set
    indices = np.random.choice(len(x_test), 100, replace=False)
    
    sample_list = []
    for idx in indices:
        features = x_test[idx].tolist()
        label = int(y_test[idx])
        # Predict in Python to record expected result
        py_pred = predict(converted_tree, FEATURE_LIST, class_names, features)
        
        sample_list.append({
            "index": int(idx),
            "features": features, # These are the 8 aligned features
            "label": label,
            "py_prediction": int(py_pred)
        })
    
    samples_export_path = os.path.join(tofino_data_path, "iscx_validation_samples_100.json")
    with open(samples_export_path, 'w', encoding='utf-8') as f:
        json.dump(sample_list, f, indent=2, ensure_ascii=False)
    print(f"100 samples exported to: {samples_export_path}")

    # Output one sample for convenience
    print("\nExample Sample (Aligned with GA_TECHNICAL_DOCUMENT.md):")
    print(f"Index: {sample_list[0]['index']}")
    for i, feat in enumerate(FEATURE_LIST):
        print(f"  - {feat:<20}: {sample_list[0]['features'][i]}")
    print(f"Label: {sample_list[0]['label']}, Py_Pred: {sample_list[0]['py_prediction']}")

if __name__ == '__main__':
    main()
