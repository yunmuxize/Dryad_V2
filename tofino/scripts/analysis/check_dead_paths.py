# -*- coding:utf-8 -*-
import json
import os
import numpy as np

# Feature bit widths for ISCX
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

def dfs_check_paths(node, path_conditions=None):
    if path_conditions is None: path_conditions = []
    if "children" not in node or not node["children"]:
        return [path_conditions]
    feat = node["feature"]
    thres = float(node["threshold"])
    paths = []
    paths.extend(dfs_check_paths(node["children"][0], path_conditions + [(feat, '<=', thres)]))
    paths.extend(dfs_check_paths(node["children"][1], path_conditions + [(feat, '>', thres)]))
    return paths

def is_path_executable(path):
    ranges = {f: [0, (1 << FEATURE_BITS[f]) - 1] for f in FEATURE_ORDER}
    for feat, op, thres in path:
        t_int = int(thres)
        low, high = ranges[feat]
        if op == '<=':
            ranges[feat][1] = min(high, t_int)
        else:
            ranges[feat][0] = max(low, t_int + 1)
        
        if ranges[feat][0] > ranges[feat][1]:
            return False, f"Contradiction in {feat}: {ranges[feat][0]} > {ranges[feat][1]}"
    return True, ""

def verify_all():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    depths = [7, 16, 26]
    
    for d in depths:
        print(f"\n--- Checking Depth {d} ---")
        model_path = os.path.join(base_dir, "data", f"iscx_depth_{d}_model.json")
        with open(model_path, 'r', encoding='utf-8') as f:
            model = json.load(f)
        
        paths = dfs_check_paths(model['tree_structure'])
        print(f"Total paths in JSON: {len(paths)}")
        
        executable_paths = 0
        dead_paths = 0
        for p in paths:
            ok, msg = is_path_executable(p)
            if ok:
                executable_paths += 1
            else:
                dead_paths += 1
        
        print(f"Executable paths: {executable_paths}")
        print(f"Dead paths (Contradictory): {dead_paths}")

if __name__ == "__main__":
    verify_all()
