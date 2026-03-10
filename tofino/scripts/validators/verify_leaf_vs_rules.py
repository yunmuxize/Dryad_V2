# -*- coding:utf-8 -*-
import json
import os
import numpy as np

def count_leaf_nodes(node):
    if "children" not in node or not node["children"]:
        return 1
    return count_leaf_nodes(node["children"][0]) + count_leaf_nodes(node["children"][1])

def verify_counts():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    depths = [5, 7, 16, 26]
    
    # Manually defined based on previous execution logs
    reported_rule_counts = {
        5: 26,
        7: 66,
        16: 890,
        26: 1542
    }

    print("=" * 60)
    print(f"{'Depth':<10} | {'Leaf Nodes':<15} | {'Rules Generated':<15} | {'Match?'}")
    print("-" * 60)

    for d in depths:
        model_path = os.path.join(base_dir, "data", f"iscx_depth_{d}_model.json")
        if not os.path.exists(model_path):
            continue
            
        with open(model_path, 'r', encoding='utf-8') as f:
            model = json.load(f)
        
        # Count using the recursive leaf node checker
        leaf_count = count_leaf_nodes(model['tree_structure'])
        
        # Compare with what generate_genetic_edt.py reported
        rules_count = reported_rule_counts[d]
        
        match = "YES" if leaf_count == rules_count else f"NO (Diff: {leaf_count - rules_count})"
        print(f"{d:<10} | {leaf_count:<15} | {rules_count:<15} | {match}")

if __name__ == "__main__":
    verify_counts()
