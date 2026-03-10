# -*- coding:utf-8 -*-
"""
Find Match Types for Specific Configurations (With Hardware Constraints)

Hardware Constraints:
1. High-cardinality 16-bit fields (Total length, Src Port, Dst Port): NO EXACT
2. 1-bit flag fields (IPV4 Flags, TCP Reset, TCP Syn): NO RANGE, only TERNARY or EXACT
3. Maximum 1 LPM (PREFIX) per table
"""

import os
import sys
import json
import pickle
import random
import copy
import numpy as np
from sklearn import tree as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimization import *

# MatchType values from optimization.py:
# EXACT = 0, TERNARY = 1, RANGE = 2, PREFIX = 3
MATCH_TYPE_NAMES = {
    0: 'EXACT',
    1: 'TERNARY',
    2: 'RANGE',
    3: 'PREFIX',
    MatchType.EXACT: 'EXACT',
    MatchType.TERNARY: 'TERNARY',
    MatchType.RANGE: 'RANGE',
    MatchType.PREFIX: 'PREFIX',
}

FEATURE_LIST = [
    'Total length', 'Protocol', 'IPV4 Flags (DF)', 'Time to live',
    'Src Port', 'Dst Port', 'TCP flags (Reset)', 'TCP flags (Syn)'
]

# Feature indices
IDX_TOTAL_LENGTH = 0
IDX_PROTOCOL = 1
IDX_IPV4_FLAGS = 2  # 1-bit flag
IDX_TTL = 3
IDX_SRC_PORT = 4
IDX_DST_PORT = 5
IDX_TCP_RESET = 6   # 1-bit flag
IDX_TCP_SYN = 7     # 1-bit flag

# Hardware constraint groups
HIGH_CARDINALITY_16BIT = [IDX_TOTAL_LENGTH, IDX_SRC_PORT, IDX_DST_PORT]  # No EXACT
ONE_BIT_FLAGS = [IDX_IPV4_FLAGS, IDX_TCP_RESET, IDX_TCP_SYN]  # Only TERNARY or EXACT (no RANGE, no PREFIX)

# Target configurations
TARGET_CONFIGS = [
    {"name": "Strict",   "depth": 12, "tcam": 4.74,  "sram": 0.1605, "stages": 1},
    {"name": "Moderate", "depth": 12, "tcam": 28.37, "sram": 0.3834, "stages": 4},
    {"name": "Relaxed",  "depth": 16, "tcam": 53.91, "sram": 0.7508, "stages": 7},
    {"name": "Open",     "depth": 26, "tcam": 43.51, "sram": 0.5973, "stages": 6},
]

def get_match_type_name(mt):
    """Get readable name for match type."""
    if mt in MATCH_TYPE_NAMES:
        return MATCH_TYPE_NAMES[mt]
    return str(mt)

def generate_valid_match_types():
    """Generate a random but hardware-valid match type configuration."""
    match_types = []
    
    for i in range(8):
        if i in HIGH_CARDINALITY_16BIT:
            # 16-bit fields: NO EXACT allowed, use TERNARY, RANGE, PREFIX
            options = [MatchType.TERNARY, MatchType.RANGE, MatchType.PREFIX]
            match_types.append(random.choice(options))
        elif i in ONE_BIT_FLAGS:
            # 1-bit flags: ONLY TERNARY or EXACT (NO RANGE, NO PREFIX)
            options = [MatchType.TERNARY, MatchType.EXACT]
            match_types.append(random.choice(options))
        else:
            # Normal fields (Protocol, TTL): all types allowed
            options = [MatchType.EXACT, MatchType.TERNARY, MatchType.RANGE, MatchType.PREFIX]
            match_types.append(random.choice(options))
    
    # Enforce max 1 PREFIX
    prefix_indices = [i for i, mt in enumerate(match_types) if mt == MatchType.PREFIX]
    if len(prefix_indices) > 1:
        for idx in prefix_indices[1:]:
            if idx in HIGH_CARDINALITY_16BIT:
                match_types[idx] = MatchType.RANGE
            elif idx in ONE_BIT_FLAGS:
                match_types[idx] = MatchType.TERNARY
            else:
                match_types[idx] = MatchType.TERNARY
    
    return match_types

def verify_constraints(match_types):
    """Verify that match types satisfy all hardware constraints."""
    errors = []
    
    # Check 16-bit fields (no EXACT)
    for idx in HIGH_CARDINALITY_16BIT:
        if match_types[idx] == MatchType.EXACT:
            errors.append(f"{FEATURE_LIST[idx]}: EXACT not allowed for 16-bit field")
    
    # Check 1-bit flags (no RANGE, no PREFIX)
    for idx in ONE_BIT_FLAGS:
        if match_types[idx] == MatchType.RANGE:
            errors.append(f"{FEATURE_LIST[idx]}: RANGE not allowed for 1-bit flag")
        elif match_types[idx] == MatchType.PREFIX:
            errors.append(f"{FEATURE_LIST[idx]}: PREFIX not allowed for 1-bit flag")
    
    # Check max 1 PREFIX
    prefix_count = sum(1 for mt in match_types if mt == MatchType.PREFIX)
    if prefix_count > 1:
        errors.append(f"Too many PREFIX: {prefix_count} (max 1)")
    
    return len(errors) == 0, errors

def find_match_types(json_model, x_test, y_test, class_names, 
                      rule_converter, resource_model, target):
    """Search for match types matching the target configuration."""
    
    print(f"\n  Searching depth={target['depth']}, target TCAM={target['tcam']:.1f}%, Stages={target['stages']}...")
    
    config = GAConfig(
        population_size=100, generations=20,
        limit_tcam=100.0, limit_sram=100.0, limit_stages=20.0
    )
    
    best_solution = None
    best_distance = float('inf')
    valid_count = 0
    
    for iteration in range(3000):
        individual = Individual()
        individual.tree_depth = target['depth']
        individual.feature_match_types = generate_valid_match_types()
        
        # Verify constraints before processing
        is_valid, errors = verify_constraints(individual.feature_match_types)
        if not is_valid:
            print(f"     WARNING: Generated invalid config: {errors}")
            continue
        
        valid_count += 1
        
        # Generate tree
        individual.tree_structure = hard_prune(copy.deepcopy(json_model), 0, target['depth'])
        individual.tree_structure = soft_prune(individual.tree_structure)
        individual.tree_structure, individual.rule_statistics = convert_tree_to_match_types(
            individual.tree_structure, individual.feature_match_types, FEATURE_LIST
        )
        individual.p4_rule_size = len(individual.rule_statistics.get('path_rules', []))
        
        calculate_fitness(individual, x_test, y_test, FEATURE_LIST, 
                          class_names, config, rule_converter, resource_model)
        
        tcam_diff = abs(individual.pred_tcam - target['tcam'])
        stages_diff = abs(individual.pred_stages - target['stages']) * 5
        distance = tcam_diff + stages_diff
        
        if distance < best_distance:
            best_distance = distance
            best_solution = copy.deepcopy(individual)
            
            if valid_count % 500 == 0 or distance < 5.0:
                print(f"     [iter={iteration}, valid={valid_count}] TCAM={individual.pred_tcam:.2f}%, Stages={individual.pred_stages:.1f}, dist={distance:.2f}")
            
            if distance < 3.0:
                break
    
    print(f"     Tested {valid_count} valid configurations")
    return best_solution

def main():
    print("=" * 90)
    print("Finding Match Types with Hardware Constraints")
    print("=" * 90)
    print("\nHardware Constraints:")
    print("  1. 16-bit fields (Total length, Src Port, Dst Port): NO EXACT")
    print("  2. 1-bit flags (IPV4 Flags, TCP Reset, TCP Syn): ONLY TERNARY or EXACT")
    print("  3. Maximum 1 PREFIX (LPM) per table")
    print("\nMatchType Values: EXACT=0, TERNARY=1, RANGE=2, PREFIX=3")
    
    # Load data
    print("\nLoading data...")
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    model_data_path = os.path.join(project_root, "model_data", "iscx")
    
    with open(os.path.join(model_data_path, "data_train_iscx_C.pkl"), "rb") as f:
        train_data = pickle.load(f)
    with open(os.path.join(model_data_path, "data_eval_iscx_C.pkl"), "rb") as f:
        test_data = pickle.load(f)
    
    target_indices = [9, 0, 1, 4, 7, 8, 2, 3]
    x_train = train_data[:, target_indices]
    y_train = train_data[:, -1].astype(int)
    x_test = test_data[:, target_indices]
    y_test = test_data[:, -1].astype(int)
    class_names = np.array(['0', '1', '2', '3', '4', '5'])
    
    print("Training base tree (max_depth=40)...")
    model = st.DecisionTreeClassifier(max_depth=40, random_state=5)
    model.fit(x_train, y_train)
    json_model = sklearn2json(model, FEATURE_LIST, class_names)
    
    print("Loading resource predictor...")
    rule_converter = P4RuleConverter()
    resource_model = ResourcePredictionModel()
    
    results = []
    
    # Search for Relaxed config
    for target in TARGET_CONFIGS:
        if target['name'] != 'Relaxed':
            continue
            
        print(f"\n{'#'*90}")
        print(f"# {target['name']}: Depth={target['depth']}, Target TCAM={target['tcam']}%, Stages={target['stages']}")
        print(f"{'#'*90}")
        
        solution = find_match_types(json_model, x_test, y_test, class_names,
                                     rule_converter, resource_model, target)
        
        if solution:
            match_types = solution.feature_match_types
            
            # Final verification
            is_valid, errors = verify_constraints(match_types)
            
            print(f"\n  FOUND SOLUTION:")
            print(f"     TCAM={solution.pred_tcam:.2f}%, Stages={solution.pred_stages:.1f}, Rules={solution.p4_rule_size}")
            print(f"\n  MATCH TYPES:")
            print(f"     {'Feature':<25} | {'Type':<10} | {'Value'} | {'Constraint'}")
            print(f"     {'-'*70}")
            for i, (feat, mt) in enumerate(zip(FEATURE_LIST, match_types)):
                constraint = ""
                mt_name = get_match_type_name(mt)
                if i in HIGH_CARDINALITY_16BIT:
                    constraint = "16-bit: no EXACT"
                elif i in ONE_BIT_FLAGS:
                    constraint = "1-bit: TERNARY/EXACT only"
                print(f"     {feat:<25} | {mt_name:<10} | {mt:5} | {constraint}")
            
            print(f"\n  CONSTRAINT VERIFICATION: {'PASSED' if is_valid else 'FAILED'}")
            if errors:
                for err in errors:
                    print(f"     ERROR: {err}")
            
            prefix_count = sum(1 for mt in match_types if mt == MatchType.PREFIX)
            print(f"     PREFIX count: {prefix_count} (max 1)")
            
            results.append({
                "config": target['name'],
                "depth": target['depth'],
                "target_tcam": target['tcam'],
                "target_stages": target['stages'],
                "found_tcam": solution.pred_tcam,
                "found_sram": solution.pred_sram,
                "found_stages": solution.pred_stages,
                "found_rules": solution.p4_rule_size,
                "match_types": [int(mt) for mt in match_types],
                "match_type_names": [get_match_type_name(mt) for mt in match_types],
                "hardware_valid": is_valid
            })
    
    if results:
        results_path = os.path.join(project_root, "experiment_results", "match_types_hardware_valid.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {results_path}")

if __name__ == '__main__':
    main()
