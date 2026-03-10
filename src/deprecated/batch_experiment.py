# -*- coding:utf-8 -*-
"""
Batch Experiment: Run GA with multiple resource limit configurations.
This script tests different resource limits to demonstrate:
  - Relaxing limits -> Higher depth achievable -> Higher F1
"""

import os
import sys
import time
import pickle
import json
import numpy as np
from sklearn import tree as st
from sklearn.metrics import classification_report, f1_score

# Ensure we can import from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimization import *
from hierarchical_ga import monotonic_resource_search_ga, GAConfig, sklearn2json

def run_batch_experiments():
    """Run experiments with 4 different resource limit configurations."""
    
    total_start = time.time()
    
    print("=" * 80)
    print("Dryad Batch Experiment: Multiple Resource Limits")
    print("=" * 80)
    
    # Path Setup
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    model_data_path = os.path.join(project_root, "model_data", "iscx")
    results_dir = os.path.join(project_root, "experiment_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Load Data
    print("\nLoading ISCX Dataset...")
    with open(os.path.join(model_data_path, "data_train_iscx_C.pkl"), "rb") as f:
        train_data = pickle.load(f)
    with open(os.path.join(model_data_path, "data_eval_iscx_C.pkl"), "rb") as f:
        test_data = pickle.load(f)
    
    target_indices = [9, 0, 1, 4, 7, 8, 2, 3]
    x_train = train_data[:, target_indices]
    y_train = train_data[:, -1].astype(int)
    x_test = test_data[:, target_indices]
    y_test = test_data[:, -1].astype(int)
    
    feature_list = [
        'Total length', 'Protocol', 'IPV4 Flags (DF)', 'Time to live',
        'Src Port', 'Dst Port', 'TCP flags (Reset)', 'TCP flags (Syn)'
    ]
    class_names = np.array(['0', '1', '2', '3', '4', '5'])
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples:     {len(x_test)}")
    
    # Train base tree
    print("\nTraining Base Decision Tree (MaxDepth=40)...")
    model = st.DecisionTreeClassifier(max_depth=40, random_state=5)
    model.fit(x_train, y_train)
    json_model = sklearn2json(model, feature_list, class_names)
    
    # Initialize tools
    print("Loading Resource Predictor...")
    rule_converter = P4RuleConverter()
    resource_model = ResourcePredictionModel()
    
    # Define 4 resource limit configurations (from strict to relaxed)
    # Target: Show increasing depth and F1 with relaxing limits
    configs = [
        {"name": "Strict",   "tcam": 5.0,  "sram": 1.0, "stages": 1},
        {"name": "Moderate", "tcam": 30.0, "sram": 4.0, "stages": 4},
        {"name": "Relaxed",  "tcam": 55.0, "sram": 7.0, "stages": 8},
        {"name": "Open",     "tcam": 80.0, "sram": 10.0, "stages": 12},
    ]
    
    results = []
    
    for cfg in configs:
        print("\n" + "=" * 80)
        print(f"🔧 Configuration: {cfg['name']}")
        print(f"   TCAM <= {cfg['tcam']}%, SRAM <= {cfg['sram']}%, Stages <= {cfg['stages']}")
        print("=" * 80)
        
        ga_config = GAConfig(
            population_size=100,
            generations=20,
            limit_tcam=cfg['tcam'],
            limit_sram=cfg['sram'],
            limit_stages=float(cfg['stages'])
        )
        
        start_time = time.time()
        
        best_solution, depth_results = monotonic_resource_search_ga(
            json_model, feature_list, class_names,
            x_test, y_test, ga_config,
            rule_converter, resource_model,
            min_depth=10, max_depth=40
        )
        
        elapsed = time.time() - start_time
        
        if best_solution:
            # Final evaluation
            preds = [int(predict(best_solution.tree_structure, feature_list, class_names, d)) for d in x_test]
            macro_f1 = f1_score(y_test, preds, average='macro')
            
            # Extract match types as readable values
            match_types = []
            for mt in best_solution.feature_match_types:
                if hasattr(mt, 'value'):
                    match_types.append(mt.value)
                else:
                    match_types.append(str(mt))
            
            result = {
                "config": cfg['name'],
                "tcam_limit": cfg['tcam'],
                "sram_limit": cfg['sram'],
                "stages_limit": cfg['stages'],
                "max_depth": best_solution.tree_depth,
                "actual_tcam": best_solution.pred_tcam,
                "actual_sram": best_solution.pred_sram,
                "actual_stages": best_solution.pred_stages,
                "rules": best_solution.p4_rule_size,
                "macro_f1": macro_f1,
                "time_sec": elapsed,
                "feature_match_types": match_types,  # Add match types!
                "feature_list": feature_list
            }
            results.append(result)
            
            print(f"\n✅ Best Solution: Depth={best_solution.tree_depth}, F1={macro_f1:.4f}")
        else:
            results.append({
                "config": cfg['name'],
                "tcam_limit": cfg['tcam'],
                "max_depth": None,
                "macro_f1": None,
                "time_sec": elapsed
            })
            print("\n❌ No feasible solution found")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("📊 BATCH EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"{'Config':<10} | {'TCAM%':<8} | {'SRAM%':<8} | {'Stages':<8} | {'Depth':<6} | {'F1':<8} | {'Time':<8}")
    print("-" * 80)
    
    for r in results:
        if r.get('max_depth'):
            print(f"{r['config']:<10} | {r['actual_tcam']:<8.2f} | {r['actual_sram']:<8.4f} | "
                  f"{r['actual_stages']:<8.1f} | {r['max_depth']:<6} | {r['macro_f1']:<8.4f} | {r['time_sec']:<8.1f}s")
        else:
            print(f"{r['config']:<10} | --- | --- | --- | --- | --- | {r['time_sec']:.1f}s")
    
    # Save results
    results_file = os.path.join(results_dir, "batch_experiment_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {results_file}")
    
    print(f"\n⏱ Total Time: {time.time() - total_start:.1f}s")

if __name__ == '__main__':
    run_batch_experiments()
