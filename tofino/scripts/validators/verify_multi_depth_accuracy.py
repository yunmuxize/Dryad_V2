# -*- coding:utf-8 -*-
import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

def predict_from_json(node, features, feature_list):
    """Recursively traverse the JSON tree to find the predicted class."""
    if "children" not in node or not node["children"]:
        return int(np.argmax(node["value"]))
    
    feature_name = node["feature"]
    threshold = float(node["threshold"])
    feature_idx = feature_list.index(feature_name)
    feature_value = float(features[feature_idx])
    
    if feature_value <= threshold:
        return predict_from_json(node["children"][0], features, feature_list)
    else:
        return predict_from_json(node["children"][1], features, feature_list)

def verify_all():
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
    samples_path = os.path.join(base_dir, "iscx_validation_samples_100.json")
    
    with open(samples_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    y_true = [s['label'] for s in samples]
    X_test = [s['features'] for s in samples]
    
    depths = [7, 16, 26]
    results = []

    print("=" * 70)
    print(f"{'Model Depth':<15} | {'Accuracy':<10} | {'Macro F1':<10} | {'Status'}")
    print("-" * 70)

    for d in depths:
        model_path = os.path.join(base_dir, f"iscx_depth_{d}_model.json")
        if not os.path.exists(model_path):
            continue
            
        with open(model_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        tree_root = model_data['tree_structure']
        feature_list = model_data['feature_list']
        
        y_pred = []
        for features in X_test:
            pred = predict_from_json(tree_root, features, feature_list)
            y_pred.append(pred)
            
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        
        print(f"Depth {d:<9} | {acc:<10.2%} | {f1:<10.4f} | OK")
        
        # Capture detailed report for output summary
        results.append({
            "depth": d,
            "accuracy": acc,
            "f1": f1,
            "report": classification_report(y_true, y_pred, zero_division=0)
        })

    # Print detailed reports
    for res in results:
        print(f"\nClassification Report for Depth {res['depth']}:")
        print(res['report'])

if __name__ == "__main__":
    verify_all()
