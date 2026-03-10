# -*- coding:utf-8 -*-
"""
Adaptive Binary Search Genetic Algorithm for Dryad (ISCX 6-Class Optimized)

Strategy:
1. Uses Binary Search [Min, Max] to find the Maximum Feasible Depth efficiently.
2. Integrates Soft Pruning and Hard Constraints for Tofino resource alignment.
3. Aligned with ISCX 6-Class dataset features (8 core header fields).
"""

import os
import sys
import time
import datetime
import random
import copy
import json
import pickle
import numpy as np
from sklearn import tree as st
from sklearn.metrics import classification_report, f1_score
from functools import cmp_to_key

# Ensure we can import from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimization import *

# =============================================================================
# Helper: Resource Scoring
# =============================================================================
def calculate_resource_score(individual):
    """
    Weighted score for resource usage (Lower is better).
    Priority: TCAM > SRAM > Stages > Rule Count
    """
    tcam_weight = 100.0
    sram_weight = 10.0
    stages_weight = 1.0
    rules_weight = 0.1
    score = (
        tcam_weight * individual.pred_tcam +
        sram_weight * individual.pred_sram +
        stages_weight * individual.pred_stages +
        rules_weight * (individual.p4_rule_size / 1000.0)
    )
    return score

# =============================================================================
# Core: Run GA at a Specific Depth (with Fast Probe Mode)
# =============================================================================
def run_ga_at_depth(current_depth, original_tree, feature_list, class_names, 
                    x_test, y_test, config, rule_converter, resource_model,
                    fast_probe=False, resource_floor_tcam=0.0):
    """
    Run Genetic Algorithm at fixed 'current_depth'.
    
    Args:
        fast_probe: If True, use minimal parameters just to check feasibility.
        resource_floor_tcam: Minimum TCAM% required for monotonic constraint.
                             Only solutions with TCAM >= this value are accepted.
    Returns: (is_feasible, best_feasible_solution)
    """
    # Fast probe mode: smaller population, MORE generations to find monotonic solution
    if fast_probe:
        pop_size = 50  # Increased for better exploration
        max_gens = 10  # Increased to find monotonic solutions
        print(f"    -> ⚡ Probing Depth {current_depth} (pop={pop_size}, gens={max_gens}, floor={resource_floor_tcam:.2f}%)...")
    else:
        pop_size = config.population_size
        max_gens = config.generations
        print(f"    -> 🔎 Full GA at Depth {current_depth} (pop={pop_size}, gens={max_gens})...")
    
    # --- 1. Initialization (Simplified Strategy: 10% Optimal + 90% Random) ---
    # Optimal config from prior knowledge (low resource baseline)
    optimal_config = [
        MatchType.RANGE,   # Total length
        MatchType.TERNARY, # Protocol
        MatchType.TERNARY, # IPV4 Flags (DF)
        MatchType.RANGE,   # Time to live
        MatchType.RANGE,   # Src Port
        MatchType.RANGE,   # Dst Port
        MatchType.TERNARY, # TCP flags (Reset/Flags1)
        MatchType.TERNARY  # TCP flags (Syn/Flags2)
    ]
    
    # All match types with EQUAL weight (no bias)
    all_match_options = [MatchType.RANGE, MatchType.TERNARY, MatchType.PREFIX, MatchType.EXACT]
        
    population = []
    
    # Population Distribution:
    # - 5%: Optimal config (with minor mutations)
    # - 95%: Pure random (equal probability for all match types)
    num_optimal = max(1, int(pop_size * 0.05))  # 5% optimal
    
    for idx in range(pop_size):
        individual = Individual()
        individual.tree_depth = current_depth
        
        if idx < num_optimal:
            # 10%: Optimal config with 2-3 random mutations
            individual.feature_match_types = optimal_config.copy()
            num_mutations = random.randint(2, 3)
            for _ in range(num_mutations):
                idx_m = random.randint(0, len(feature_list)-1)
                individual.feature_match_types[idx_m] = random.choice(all_match_options)
        else:
            # 90%: Pure random (equal probability for all 4 match types)
            individual.feature_match_types = [random.choice(all_match_options) for _ in range(len(feature_list))]
        
        # NOTE: No restrictions on EXACT - let GA explore freely
        # The resource limits will naturally filter out infeasible solutions
        
        # Generate Tree (Hard Prune -> Soft Prune)
        individual.tree_structure = hard_prune(copy.deepcopy(original_tree), 0, current_depth)
        individual.tree_structure = soft_prune(individual.tree_structure)
        individual.tree_structure, individual.rule_statistics = convert_tree_to_match_types(
            individual.tree_structure, individual.feature_match_types, feature_list
        )
        individual.p4_rule_size = len(individual.rule_statistics.get('path_rules', []))
        
        population.append(individual)

    best_feasible_solution = None
    no_improvement_count = 0
    
    # --- 2. Evolution Loop ---
    for gen in range(max_gens):
        # Evaluation - ONLY accept solutions that satisfy monotonic constraint
        feasible_pop = []
        monotonic_feasible_pop = []  # Solutions that also satisfy TCAM >= floor
        for individual in population:
            # Ensure p4_rule_size is set
            if individual.p4_rule_size == 0:
                individual.p4_rule_size = len(individual.rule_statistics.get('path_rules', []))
            
            # Handle both Enum and raw value (int/str)
            individual.match_type_names = []
            for mt in individual.feature_match_types:
                if hasattr(mt, 'value'):
                    individual.match_type_names.append(mt.value)
                else:
                    individual.match_type_names.append(str(mt))
            
            # Call the full calculate_fitness with all required args
            calculate_fitness(individual, x_test, y_test, feature_list, 
                              class_names, config, rule_converter, resource_model)
            
            if individual.is_feasible:
                feasible_pop.append(individual)
                # Check monotonic constraint: TCAM must be >= floor
                if individual.pred_tcam >= resource_floor_tcam:
                    monotonic_feasible_pop.append(individual)
        
        # Update Best - ONLY from monotonic feasible solutions
        if monotonic_feasible_pop:
            # Sort by TCAM descending (prefer higher resource usage to ensure future depths can also find solutions)
            monotonic_feasible_pop.sort(key=lambda x: -x.pred_tcam)
            current_best = monotonic_feasible_pop[0]
            
            if best_feasible_solution is None or current_best.fitness > best_feasible_solution.fitness:
                best_feasible_solution = copy.deepcopy(current_best)
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Diagnostic Log (less frequent in fast mode)
            if not fast_probe and (gen+1) % 5 == 0:
                print(f"       Gen {gen+1}: Found {len(feasible_pop)} feasible sols. Best F1: {current_best.fitness:.4f}")
            
            # Fast Probe: Return once a MONOTONIC feasible solution is found
            if fast_probe:
                print(f"       ✔ Found monotonic feasible at gen {gen+1} (TCAM={current_best.pred_tcam:.2f}% >= {resource_floor_tcam:.2f}%)")
                return True, best_feasible_solution
            
            # Full GA: Early Exit after enough exploration
            if gen >= 10 and best_feasible_solution:
                 return True, best_feasible_solution
        elif feasible_pop:  # Found feasible but not monotonic
            print(f"       ⚠ Gen {gen+1}: Found {len(feasible_pop)} feasible but 0 monotonic (need TCAM>={resource_floor_tcam:.2f}%), continuing...")
        else:
             no_improvement_count += 1

        # Evolution (Elitism + Selection + Crossover + Mutation)
        elite_size = min(config.elite_size, pop_size // 5)  # Adaptive elite size
        elites = sorted(population, key=lambda x: x.fitness, reverse=True)[:elite_size]
        offspring = []
        offspring.extend(elites)
        
        while len(offspring) < pop_size:
            parent1 = tournament_selection(population, config.tournament_size)
            parent2 = tournament_selection(population, config.tournament_size)
            child = crossover(parent1, parent2, feature_list, original_tree, current_depth)
            
            # Mutation (equal probability for all match types)
            for i in range(len(child.feature_match_types)):
                if random.random() < config.mutation_rate:
                    child.feature_match_types[i] = random.choice(all_match_options)
            
            # NOTE: No longer restricting EXACT - let GA explore freely
            
            # Re-generate Tree
            child.tree_structure = hard_prune(copy.deepcopy(original_tree), 0, current_depth)
            child.tree_structure = soft_prune(child.tree_structure)
            child.tree_structure, child.rule_statistics = convert_tree_to_match_types(
                child.tree_structure, child.feature_match_types, feature_list
            )
            child.p4_rule_size = len(child.rule_statistics.get('path_rules', []))
            offspring.append(child)
            
        population = offspring

    return (best_feasible_solution is not None), best_feasible_solution


# =============================================================================
# Main Algorithm: Monotonic Resource Search (Low-to-High with Resource Floor)
# =============================================================================
def monotonic_resource_search_ga(original_tree, feature_list, class_names, 
                                  x_test, y_test, config, 
                                  rule_converter, resource_model,
                                  min_depth=10, max_depth=40):
    """
    Monotonic Resource Search Strategy:
    
    1. Search from LOW depth to HIGH depth (linear scan)
    2. Maintain a "resource floor" (minimum TCAM usage seen so far)
    3. For each depth, only accept solutions where TCAM >= resource_floor
    4. Update resource_floor when a valid solution is found
    
    This ensures: Depth↑ → Resource↑ (strictly monotonic)
    
    Returns: Dictionary with solutions at each feasible depth for comparison.
    """
    print(f"\n🚀 Starting Monotonic Resource Search (Range: {min_depth}-{max_depth})")
    print(f"   Resource Limits: TCAM<={config.limit_tcam}%, SRAM<={config.limit_sram}%, Stages<={config.limit_stages}")
    print(f"   Constraint: Resource consumption must increase with depth")
    
    # Track results at each depth
    depth_results = {}
    resource_floor_tcam = 0.0  # Minimum TCAM usage required
    resource_floor_sram = 0.0  # Minimum SRAM usage required
    
    best_solution = None
    max_feasible_depth = None
    
    start_time = time.time()
    
    # Linear scan from low to high
    for current_depth in range(min_depth, max_depth + 1):
        print(f"\n  📊 Probing Depth {current_depth} (Resource Floor: TCAM>={resource_floor_tcam:.2f}%)...")
        
        # Run GA at this depth with monotonic constraint enforced internally
        is_feasible, solution = run_ga_at_depth(
            current_depth, original_tree, feature_list, class_names,
            x_test, y_test, config, rule_converter, resource_model,
            fast_probe=True,
            resource_floor_tcam=resource_floor_tcam  # Pass the floor!
        )
        
        if is_feasible and solution is not None:
            # GA guarantees solution.pred_tcam >= resource_floor_tcam
            print(f"     ✅ Depth {current_depth} Feasible & Monotonic! "
                  f"TCAM={solution.pred_tcam:.2f}%")
            
            # Update floor for next depth
            resource_floor_tcam = solution.pred_tcam
            resource_floor_sram = solution.pred_sram
            
            # Store result
            depth_results[current_depth] = {
                'solution': solution,
                'tcam': solution.pred_tcam,
                'sram': solution.pred_sram,
                'stages': solution.pred_stages,
                'rules': solution.p4_rule_size,
                'f1': solution.fitness
            }
            
            best_solution = solution
            max_feasible_depth = current_depth
        else:
            # No monotonic feasible solution found at this depth
            print(f"     ❌ Depth {current_depth}: No monotonic feasible solution found")
            print(f"     🛑 Stopping search - cannot satisfy constraints at depth {current_depth}")
            break
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*70}")
    print(f"🏁 Monotonic Search Completed in {total_time:.2f}s")
    print(f"{'='*70}")
    
    if depth_results:
        print(f"\n📈 Depth vs Resource Consumption (Monotonic):")
        print(f"{'Depth':>6} | {'TCAM%':>8} | {'SRAM%':>8} | {'Stages':>6} | {'Rules':>6} | {'F1':>8}")
        print("-" * 60)
        for d in sorted(depth_results.keys()):
            r = depth_results[d]
            print(f"{d:>6} | {r['tcam']:>8.2f} | {r['sram']:>8.4f} | {r['stages']:>6.1f} | {r['rules']:>6} | {r['f1']:>8.4f}")
        
        print(f"\n🌟 Max Feasible Depth: {max_feasible_depth}")
    else:
        print("⚠ No feasible depth found in range.")
    
    return best_solution, depth_results


# =============================================================================
# Entry Point: ISCX Test
# =============================================================================
def test_hierarchical_ga():
    total_start_time = time.time()
    
    print("=" * 80)
    print("Dryad: Monotonic Resource Search on ISCX (6-Class)")
    print("=" * 80)
    
    # 1. Path Setup (src/ -> root/)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    model_data_path = os.path.join(project_root, "model_data", "iscx")
    json_models_dir = os.path.join(project_root, "json_models")
    if not os.path.exists(json_models_dir): os.makedirs(json_models_dir)
    
    # 2. Data Loading & Feature Alignment
    print("\nLoading ISCX Dataset...")
    with open(os.path.join(model_data_path, "data_train_iscx_C.pkl"), "rb") as f:
        train_data = pickle.load(f)
    with open(os.path.join(model_data_path, "data_eval_iscx_C.pkl"), "rb") as f:
        test_data = pickle.load(f)
        
    # Alignment: [Total(9), Proto(0), IPv4Flag(1), TTL(4), Port1(7), Port2(8), TcpF1(2), TcpF2(3)]
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
    
    print(f"Training Data: {x_train.shape}")
    print(f"Test Data:     {x_test.shape}")
    
    # 3. Pre-train Base Tree (Max Depth 40 - optimal found via overfitting analysis)
    print("\nTraining Base Decision Tree (MaxDepth=40)...")
    model = st.DecisionTreeClassifier(max_depth=40, random_state=5)
    model.fit(x_train, y_train)
    json_model = sklearn2json(model, feature_list, class_names)
    
    # 4. Configure GA - 资源限制 (可调整以观察不同限制下的最大深度)
    config = GAConfig(
        population_size=100,
        generations=20,
        limit_tcam=30.0,    # 30%
        limit_sram=4.0,     # 4%
        limit_stages=8.0    # 8
    )
    
    # Initialize Converters/Predictors
    print("Loading Resource Predictor...")
    rule_converter = P4RuleConverter()
    resource_model = ResourcePredictionModel()
    
    # 5. Run Monotonic Resource Search (深度范围 10-40, 40为最优深度上限)
    best_solution, depth_results = monotonic_resource_search_ga(
        json_model, feature_list, class_names,
        x_test, y_test, config,
        rule_converter, resource_model,
        min_depth=10, max_depth=40
    )
    
    # 6. Output Result
    if best_solution:
        print("\n" + "="*80)
        print("🏆 OPTIMAL SOLUTION FOUND")
        print("="*80)
        print(f"Depth: {best_solution.tree_depth}")
        print(f"Fitness (F1): {best_solution.fitness:.4f}")
        print(f"Rules: {best_solution.p4_rule_size}")
        print(f"Resource: TCAM={best_solution.pred_tcam:.2f}%, SRAM={best_solution.pred_sram:.4f}%, St={best_solution.pred_stages:.1f}")
        
        # Safe Match Types Printing
        match_types_display = []
        for mt in best_solution.feature_match_types:
            match_types_display.append(mt.value if hasattr(mt, 'value') else str(mt))
        print(f"Match Types: {match_types_display}")
        
        # Save results
        result_json = {
            "feature_match_types": match_types_display,
            "tree_depth": best_solution.tree_depth,
            "feature_list": feature_list,
            "depth_results": {
                str(d): {
                    'tcam': r['tcam'],
                    'sram': r['sram'],
                    'stages': r['stages'],
                    'rules': r['rules'],
                    'f1': r['f1']
                } for d, r in depth_results.items()
            }
        }
        with open(os.path.join(json_models_dir, "hierarchical_ga_result.json"), 'w') as f:
            json.dump(result_json, f, indent=2)
            
        # Macro F1 Evaluation
        print("\n📊 Final 6-Class Evaluation (Macro F1):")
        preds = [int(predict(best_solution.tree_structure, feature_list, class_names, d)) for d in x_test]
        
        report = classification_report(y_test, preds, digits=6, output_dict=True)
        print(classification_report(y_test, preds, digits=6))
        
        print(f"Total Run Time: {time.time() - total_start_time:.2f}s")

if __name__ == '__main__':
    test_hierarchical_ga()
