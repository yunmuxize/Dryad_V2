# -*- coding:utf-8 -*-
import os
import sys
import json
import pickle
import numpy as np
from sklearn import tree as st

# Setup Path to Dryad/src
current_dir = os.path.dirname(os.path.abspath(__file__)) # tofino/scripts
project_root = os.path.dirname(os.path.dirname(current_dir)) # Dryad
src_path = os.path.join(project_root, "Dryad", "src")
sys.path.insert(0, src_path)

try:
    from optimization import sklearn2json, hard_prune, soft_prune
except ImportError:
    print(f"Error: Could not import optimization from {src_path}")
    sys.exit(1)

FEATURE_LIST = [
    'Total length', 'Protocol', 'IPV4 Flags (DF)', 'Time to live',
    'Src Port', 'Dst Port', 'TCP flags (Reset)', 'TCP flags (Syn)'
]

def export_model(depth, x_train, y_train, class_names, output_path):
    print(f"Training and exporting depth {depth} model...")
    model = st.DecisionTreeClassifier(max_depth=depth, random_state=5)
    model.fit(x_train, y_train)
    
    json_model = sklearn2json(model, FEATURE_LIST, class_names)
    json_model = hard_prune(json_model, 0, depth)
    json_model = soft_prune(json_model)
    
    export_data = {
        "model_type": "DecisionTree",
        "depth": depth,
        "feature_list": FEATURE_LIST,
        "tree_structure": json_model,
        "class_names": class_names.tolist()
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    print(f"Done: {output_path}")

def main():
    model_data_path = os.path.join(project_root, "Dryad", "model_data", "iscx")
    tofino_data_path = os.path.join(project_root, "tofino", "data")
    
    if not os.path.exists(tofino_data_path):
        os.makedirs(tofino_data_path)

    # Load Data
    print("Loading dataset...")
    with open(os.path.join(model_data_path, "data_train_iscx_C.pkl"), "rb") as f:
        train_data = pickle.load(f)
    
    target_indices = [9, 0, 1, 4, 7, 8, 2, 3] # [Total, Proto, DF, TTL, Sport, Dport, Reset, Syn]
    x_train = train_data[:, target_indices]
    y_train = train_data[:, -1].astype(int)
    class_names = np.array(['0', '1', '2', '3', '4', '5'])

    depths = [7, 16, 26]
    for d in depths:
        filename = f"iscx_depth_{d}_model.json"
        output_path = os.path.join(tofino_data_path, filename)
        export_model(d, x_train, y_train, class_names, output_path)

if __name__ == '__main__':
    main()
