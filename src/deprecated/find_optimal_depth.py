# -*- coding:utf-8 -*-
"""
Modified Script: Find Valid Configuration for Depth 5 (Macro F1 Optimized)
Targets: 
- TCAM <= 5.0%
- SRAM <= 1.0%
- Stages <= 1
- Depth = 5
- Metrics: Macro-average F1 (6 classes)
"""

import os
import sys
import json
import pickle
import random
import copy
import numpy as np
from sklearn import tree as st
from sklearn.metrics import f1_score

# Ensure we can import from optimization
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimization import *

# Match Type Mapping
MATCH_TYPE_NAMES = {0: 'EXACT', 1: 'TERNARY', 2: 'RANGE', 3: 'PREFIX'}

FEATURE_LIST = [
    'Total length', 'Protocol', 'IPV4 Flags (DF)', 'Time to live',
    'Src Port', 'Dst Port', 'TCP flags (Reset)', 'TCP flags (Syn)'
]

# Indices
HIGH_CARDINALITY_16BIT = [0, 4, 5]  # Total length, Src Port, Dst Port
ONE_BIT_FLAGS = [2, 6, 7]          # IPV4 Flags, TCP Reset, TCP Syn

def calculate_macro_f1(tree_structure, x_test, y_test, feature_list, class_names):
    """Calculate Macro-average F1 for 6 classes."""
    preds = []
    for d in x_test:
        try:
            p = predict(tree_structure, feature_list, class_names, d)
            preds.append(int(p))
        except:
            preds.append(0)
    return f1_score(y_test, preds, average='macro')

def generate_hardware_valid_match_types():
    """Generate random but hardware-valid match types."""
    match_types = []
    for i in range(8):
        if i in HIGH_CARDINALITY_16BIT:
            # 16-bit: RANGE, TERNARY, PREFIX allowed (NO EXACT)
            options = [MatchType.TERNARY, MatchType.RANGE, MatchType.PREFIX]
            match_types.append(random.choice(options))
        elif i in ONE_BIT_FLAGS:
            # 1-bit flags: ONLY TERNARY or EXACT (NO RANGE, NO PREFIX)
            options = [MatchType.TERNARY, MatchType.EXACT]
            match_types.append(random.choice(options))
        else:
            options = [MatchType.EXACT, MatchType.TERNARY, MatchType.RANGE, MatchType.PREFIX]
            match_types.append(random.choice(options))
    
    # Enforce Max 1 PREFIX
    prefix_indices = [i for i, mt in enumerate(match_types) if mt == MatchType.PREFIX]
    if len(prefix_indices) > 1:
        for idx in prefix_indices[1:]:
            if idx in HIGH_CARDINALITY_16BIT:
                match_types[idx] = MatchType.RANGE
            else:
                match_types[idx] = MatchType.TERNARY
    return match_types

def main():
    print("=" * 80)
    print("Searching for Depth 5 configuration matching targets:")
    print("TCAM <= 5.0%, SRAM <= 1.0%, Stages <= 1")
    print("Metric: Macro-average F1 (6-Class)")
    print("=" * 80)
    
    # Path setup
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    model_data_path = os.path.join(project_root, "model_data", "iscx")
    
    # Load data
    print("\nLoading data...")
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
    
    # Train base tree
    print("Training base tree (max_depth=5)...")
    model = st.DecisionTreeClassifier(max_depth=5, random_state=5)
    model.fit(x_train, y_train)
    json_model = sklearn2json(model, FEATURE_LIST, class_names)
    
    # Initialize resource models
    rule_converter = P4RuleConverter()
    resource_model = ResourcePredictionModel()
    
    # Search targets
    limit_tcam = 5.0
    limit_sram = 1.0
    limit_stages = 1.0
    target_depth = 5
    
    config = GAConfig(
        population_size=100,
        limit_tcam=limit_tcam,
        limit_sram=limit_sram,
        limit_stages=limit_stages
    )
    
    print("\nStarting search...")
    found = False
    for iteration in range(10000):  # More iterations
        individual = Individual()
        individual.tree_depth = target_depth
        individual.feature_match_types = generate_hardware_valid_match_types()
        
        # Apply pruning
        individual.tree_structure = hard_prune(copy.deepcopy(json_model), 0, target_depth)
        individual.tree_structure = soft_prune(individual.tree_structure)
        individual.tree_structure, individual.rule_statistics = convert_tree_to_match_types(
            individual.tree_structure, individual.feature_match_types, FEATURE_LIST
        )
        
        # Rule size for predictor
        individual.p4_rule_size = len(individual.rule_statistics.get('path_rules', []))
        
        # Predict resources
        # [0-7] match types, [8-15] bit widths, [16] p4_rule_size, [17-20] counts
        m_types = [int(mt) for mt in individual.feature_match_types]
        bit_widths = [16, 8, 1, 8, 16, 16, 1, 1] # Fixed widths for ISCX core 8
        size_feat = [float(individual.p4_rule_size)]
        counts = [m_types.count(i) for i in range(4)]
        
        # Note: Optimization for ISCX uses bit_widths defined in hierarchical_ga
        # bit_widths = [16, 8, 8, 8, 16, 16, 8, 8] - let's stick to the one in optimization.py or what trained model expects
        # In optimization.py: bit_widths = [16, 8, 8, 8, 16, 16, 8, 8]
        bit_widths_final = [16, 8, 8, 8, 16, 16, 8, 8]
        
        feature_vector = np.array(m_types + bit_widths_final + size_feat + counts)
        sram, tcam, stages = resource_model.predict(feature_vector)
        
        individual.pred_sram = sram
        individual.pred_tcam = tcam
        individual.pred_stages = stages
        
        # Check constraints
        is_feasible = (tcam <= limit_tcam and sram <= limit_sram and stages <= limit_stages)
        
        # Mandatory Hardware Check (LPM count, 1-bit RANGE)
        lpm_count = m_types.count(3)
        hardware_check = (lpm_count <= 1)
        for i in ONE_BIT_FLAGS:
            if m_types[i] == 2: # RANGE
                hardware_check = False
        
        if is_feasible and hardware_check:
            # Calculate Macro F1
            macro_f1 = calculate_macro_f1(individual.tree_structure, x_test, y_test, FEATURE_LIST, class_names)
            
            print(f"\n[OK] Found a solution at iteration {iteration}")
            print(f"   TCAM: {individual.pred_tcam:.4f}%, SRAM: {individual.pred_sram:.4f}%, Stages: {individual.pred_stages}")
            print(f"   Rules: {individual.p4_rule_size}, Macro F1: {macro_f1:.4f}")
            
            print("\n  Match Type Combination:")
            for i, (feat, mt) in enumerate(zip(FEATURE_LIST, m_types)):
                print(f"  - {feat:<20}: {MATCH_TYPE_NAMES[mt]} ({mt})")
            
            # Save for later use if needed
            found = True
            break
            
        if iteration % 1000 == 0:
            print(f"   ... Processed {iteration} iterations")
            
    if not found:
        print("\nCould not find a configuration matching the constraints at depth 5.")

if __name__ == '__main__':
    main()
