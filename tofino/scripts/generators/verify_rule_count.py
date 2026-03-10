import json
import os
import sys

# Hardware constraints
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

# Exact expansion: special handling for Protocol/Flags if wildcarded
# The report says Protocol has values 6, 17. Flags 0, 1.
# If wildcarded (range covers full space), we only expand to 'interested' values to save space.
# However, if it's a specific range (e.g. < 10), we must expand exactly.
# Based on the report, Protocol is currently ALWAYS wildcarded in the tree paths.
INTERESTED_VALUES = {
    'Protocol': [6, 17],
    'IPV4 Flags (DF)': [0, 1],
    'TCP flags (Reset)': [0, 1],
    'TCP flags (Syn)': [0, 1]
}

def range_to_prefix_decomposition(start, end, width):
    """Decompose [start, end] into minimal set of prefixes."""
    if start > end: return 0
    if start == 0 and end == (1 << width) - 1: return 1 # Default route /0
    
    count = 0
    while start <= end:
        # 1. Physical alignment (LSB)
        if start == 0:
            lsb_len = width
        else:
            lsb_len = (start & -start).bit_length() - 1
            
        # 2. MSB difference
        diff = start ^ end
        if diff == 0:
            mask_len = width
        else:
            mask_len = width - diff.bit_length()
        
        # Max block size
        prefix_len = max(mask_len, width - lsb_len)
        
        # Check boundary
        while True:
            mask = ((1 << (width - prefix_len)) - 1)
            range_end = start | mask
            if range_end <= end:
                break
            prefix_len += 1
            
        count += 1
        start = (start | ((1 << (width - prefix_len)) - 1)) + 1
        
    return count

def get_exact_count(feature, start, end):
    width = FEATURE_BITS[feature]
    # Check if full wildcard
    if start == 0 and end == (1 << width) - 1:
        if feature in INTERESTED_VALUES:
            return len(INTERESTED_VALUES[feature])
        else:
            # For continuous features like Port/TTL, wildcard exact 
            # implies listing ALL values. 
            # BUT, often in P4, if we don't match on a key, we just don't put it in the table key?
            # The prompt requires "7 exact and 1 lpm" configuration for the TABLE KEYS.
            # If the table key IS exact, we MUST match on something. 
            # If the logic for that path is "don't care", we usually have to enumerate all values
            # OR use a 'default' value if logic permits, but here we are mapping tree paths to table entries.
            # Standard "don't care" in Exact Match table = duplicate rule for every possible value.
            return (1 << width)
    
    # Not wildcard, precise range
    return end - start + 1

def aggregate_path_conditions(path_conditions):
    feature_ranges = {f: (0, (1 << FEATURE_BITS[f]) - 1) for f in FEATURE_ORDER}
    for feat, op, thres in path_conditions:
        if feat not in feature_ranges: continue
        t_int = int(thres)
        c_min, c_max = feature_ranges[feat]
        if op == '<=': feature_ranges[feat] = (c_min, min(c_max, t_int))
        else: feature_ranges[feat] = (max(c_min, t_int + 1), c_max)
    return feature_ranges

def dfs_traverse_tree(tree, path_conditions=None):
    if path_conditions is None: path_conditions = []
    if "children" not in tree or len(tree["children"]) == 0:
        yield path_conditions
        return
    feature = tree["feature"]
    threshold = float(tree["threshold"])
    yield from dfs_traverse_tree(tree["children"][0], path_conditions + [(feature, '<=', threshold)])
    yield from dfs_traverse_tree(tree["children"][1], path_conditions + [(feature, '>', threshold)])

def calculate_total_rules(tree, lpm_feature):
    total_rules = 0
    paths = list(dfs_traverse_tree(tree))
    
    for path in paths:
        ranges = aggregate_path_conditions(path)
        path_multiplier = 1
        
        for feature in FEATURE_ORDER:
            start, end = ranges[feature]
            if feature == lpm_feature:
                # LPM decomposition
                # Optimization: if full wildcard, count is 1 (prefix /0)
                # If specific, decompose
                count = range_to_prefix_decomposition(start, end, FEATURE_BITS[feature])
            else:
                # Exact decomposition
                count = get_exact_count(feature, start, end)
            
            path_multiplier *= count
            
            # optimization checks to return early if huge
            if path_multiplier > 10000000:
                break
        
        total_rules += path_multiplier
        
    return total_rules

def main():
    model_path = r"c:\Users\86177\OneDrive\Desktop\Dryad_V2\Dryad\tofino\data\iscx_depth_5_model.json"
    with open(model_path, 'r') as f:
        model = json.load(f)
    tree = model['tree_structure']
    
    print("Exploring Minimal Rules for 7 Exact + 1 LPM configuration:")
    print("-" * 60)
    
    results = []
    
    for feature in FEATURE_ORDER:
        try:
            rules = calculate_total_rules(tree, feature)
            results.append((feature, rules))
            print(f"LPM Feature: {feature:20s} | Total Rules: {rules}")
        except OverflowError:
             print(f"LPM Feature: {feature:20s} | Total Rules: > 10^9 (Overflow)")

    print("-" * 60)
    best = min(results, key=lambda x: x[1])
    print(f"BEST CONFIGURATION: LPM on '{best[0]}'")
    print(f"MINIMUM RULES: {best[1]}")

if __name__ == "__main__":
    main()
