# -*- coding:utf-8 -*-
import json
import os

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

def find_first_dead_path(node, path_conditions=None):
    if path_conditions is None: path_conditions = []
    
    if "children" not in node or not node["children"]:
        # Check if this leaf is dead
        ranges = {f: [0, (1 << FEATURE_BITS[f]) - 1] for f in FEATURE_BITS}
        for feat, op, thres in path_conditions:
            t_int = int(float(thres))
            if op == '<=':
                ranges[feat][1] = min(ranges[feat][1], t_int)
            else:
                ranges[feat][0] = max(ranges[feat][0], t_int + 1)
            
            if ranges[feat][0] > ranges[feat][1]:
                return True, path_conditions, feat, ranges[feat]
        return False, None, None, None

    # Recurse
    res, path, feat, rng = find_first_dead_path(node["children"][0], path_conditions + [(node["feature"], '<=', node["threshold"])])
    if res: return res, path, feat, rng
    
    res, path, feat, rng = find_first_dead_path(node["children"][1], path_conditions + [(node["feature"], '>', node["threshold"])])
    if res: return res, path, feat, rng
    
    return False, None, None, None

def prove_dead_path():
    model_path = r"C:\Users\86177\OneDrive\Desktop\Dryad_V2\Dryad\tofino\data\iscx_depth_7_model.json"
    with open(model_path, 'r', encoding='utf-8') as f:
        model = json.load(f)
    
    found, path, conflict_feat, final_range = find_first_dead_path(model['tree_structure'])
    
    if found:
        print("Confirmed: Dead Path Found!")
        print("-" * 50)
        print("Path conditions (Logical from Decision Tree):")
        for f, op, t in path:
            print(f"  - {f} {op} {t}")
        
        print("\nConflict Analysis for feature:", conflict_feat)
        relevant_conds = [c for c in path if c[0] == conflict_feat]
        for f, op, t in relevant_conds:
            t_float = float(t)
            t_int = int(t_float)
            if op == '<=':
                print(f"    IF {f} <= {t_float} (Model) -> Hardware takes {f} <= {t_int}")
            else:
                print(f"    IF {f} > {t_float} (Model) -> Hardware takes {f} >= {t_int + 1}")
        
        print(f"\nResulting Hardware Range: [{final_range[0]}, {final_range[1]}]")
        print("Status: INVALID RANGE (Low > High), so this rule is discarded.")
    else:
        print("No dead path found in this specific branch.")

if __name__ == "__main__":
    prove_dead_path()
