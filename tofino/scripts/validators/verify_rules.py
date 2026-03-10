# -*- coding:utf-8 -*-
"""
Verification script to ensure 26 rules match 26 tree paths accurately.
Checks ranges, ternary masks, and class_ids.
"""

import json
import os
import numpy as np

# Feature and Match Type Config (Same as in generate_genetic_edt.py)
FEATURE_BITS = {
    'Total length': 16,
    'Protocol': 8,
    'IPV4 Flags (DF)': 1,
    'Time to live': 8,
    'Src Port': 16,
    'Dst Port': 16,
    'TCP flags (Reset)': 1,
    'TCP flags (Syn)': 1
}

FEATURE_ORDER = [
    'Total length',
    'Protocol',
    'IPV4 Flags (DF)',
    'Time to live',
    'Src Port',
    'Dst Port',
    'TCP flags (Reset)',
    'TCP flags (Syn)'
]

def range_to_ternary_decomposition(start, end, width):
    if start > end: return []
    if start == 0 and end == (1 << width) - 1: return [(0, 0)]
    if start == end: return [(start, (1 << width) - 1)]
    result = []
    if start % 2 == 1:
        result.append((start, (1 << width) - 1))
        start += 1
    if start <= end and end % 2 == 0:
        result.append((end, (1 << width) - 1))
        end -= 1
    if start <= end:
        sub_results = range_to_ternary_decomposition(start >> 1, end >> 1, width - 1)
        for v, m in sub_results:
            result.append((v << 1, (m << 1) | 1 if m != 0 else 0))
    return result

def dfs_paths(node, path_conditions=None):
    if path_conditions is None: path_conditions = []
    if "children" not in node or not node["children"]:
        yield path_conditions, int(np.argmax(node["value"]))
        return
    feat = node["feature"]
    thres = float(node["threshold"])
    yield from dfs_paths(node["children"][0], path_conditions + [(feat, '<=', thres)])
    yield from dfs_paths(node["children"][1], path_conditions + [(feat, '>', thres)])

def get_ranges(path):
    ranges = {f: [0, (1 << FEATURE_BITS[f]) - 1] for f in FEATURE_ORDER}
    for feat, op, thres in path:
        t_int = int(thres)
        if op == '<=':
            ranges[feat][1] = min(ranges[feat][1], t_int)
        else:
            ranges[feat][0] = max(ranges[feat][0], t_int + 1)
    return ranges

def verify():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "data", "iscx_depth_5_model.json")
    rules_path = os.path.join(base_dir, "generated", "iscx_depth_5", "genetic_edt_rules.txt")

    with open(model_path, 'r') as f:
        model = json.load(f)
    
    with open(rules_path, 'r') as f:
        rules_lines = [line.strip() for line in f if line.strip()]

    print(f"Tree Paths: {len(list(dfs_paths(model['tree_structure'])))}")
    print(f"Rule Lines: {len(rules_lines)}")

    # Map rules to comparable tuples
    # Format: bfrt.genetic_edt.pipe.Ingress.EDT.add_with_SetClass(0, 30, 0, 0, 0, 0, 0, 255, 0, 255, 0, 58, 0, 0, 0, 0, class_id=5)
    parsed_rules = []
    for line in rules_lines:
        content = line.split("(")[1].split(")")[0]
        parts = content.split(", ")
        vals = []
        for p in parts[:-1]: # All keys
            vals.append(int(p))
        cid = int(parts[-1].split("=")[1])
        parsed_rules.append((tuple(vals), cid))

    # Verify each path
    match_count = 0
    for path, class_id in dfs_paths(model['tree_structure']):
        ranges = get_ranges(path)
        
        # Build the expected BFRT key (assuming no splitting for simplicity in comparison here, 
        # since our generate logic for Depth 5 only had 26 paths -> 26 rules, implying 1:1)
        expected_keys = []
        for feat in FEATURE_ORDER:
            low, high = ranges[feat]
            bits = FEATURE_BITS[feat]
            if feat in ['Protocol', 'IPV4 Flags (DF)', 'TCP flags (Reset)', 'TCP flags (Syn)']:
                # Ternary
                decomp = range_to_ternary_decomposition(low, high, bits)
                if len(decomp) > 1:
                    print(f"Warning: Path requires {len(decomp)} rules for feature {feat}. Verification logic needs expansion.")
                expected_keys.append(decomp[0][0])
                expected_keys.append(decomp[0][1])
            else:
                # Range
                expected_keys.append(low)
                expected_keys.append(high)
        
        expected_tuple = (tuple(expected_keys), class_id)
        if expected_tuple in parsed_rules:
            match_count += 1
        else:
            print(f"Mismatch for path to class {class_id}!")
            print(f"Expected: {expected_tuple}")

    print(f"\nFinal Result: {match_count}/26 paths verified and found in rules.")

if __name__ == "__main__":
    verify()
