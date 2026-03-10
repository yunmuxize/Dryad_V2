# -*- coding: utf-8 -*-
"""
单深度探索器 (V6)
加权多目标优化适应度函数
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'predictors'))

from ga_config import (
    GA_PARAMS, DEPTH_F1_MAP, FEATURE_ORDER, ALL_FEATURES,
    MATCH_TYPE_OPTIONS, MAX_LPM_COUNT, ONE_BIT_FEATURES,
    MAX_RULES_REF, calculate_weighted_fitness
)
from ga_operators import initialize_population, create_next_generation
from tofino_predictor import TofinoPredictor
from entries_calculator import calculate_entries_count
import json


def load_model(depth):
    """加载指定深度的决策树模型"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, 'json_models', f'iscx_depth_{depth}_model.json')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(model_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def decode_chromosome(chromosome):
    """将染色体解码为匹配配置"""
    match_config = {}
    for i, feat in enumerate(ALL_FEATURES):
        options = MATCH_TYPE_OPTIONS[feat]
        gene = chromosome[i] % len(options)
        match_config[feat] = options[gene]
    return match_config


def match_config_to_list(match_config):
    """将匹配配置转换为列表"""
    return [match_config.get(f, 1) for f in FEATURE_ORDER]


def check_hardware_constraints(match_config):
    """检查硬件约束"""
    for feat in ONE_BIT_FEATURES:
        if match_config.get(feat) == 2:
            return False, f"1-bit '{feat}' cannot use Range"
    
    lpm_count = sum(1 for v in match_config.values() if v == 3)
    if lpm_count > MAX_LPM_COUNT:
        return False, f"LPM count {lpm_count} > {MAX_LPM_COUNT}"
    
    return True, None


def check_resource_constraints(resource, config):
    """检查资源硬约束（上限）"""
    tcam_max = config.get('tcam_pct_max', 100)
    return (
        resource['tcam_pct'] <= tcam_max and
        resource['sram_pct'] <= config['sram_pct'] and
        resource['stages'] <= config['stages']
    )


_predictor = None
def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = TofinoPredictor()
    return _predictor


def fitness_function(chromosome, model, config, depth):
    """
    加权多目标适应度函数
    
    Fitness = α × F1 - β × TCAM_Util - γ × Latency_Factor
    """
    match_config = decode_chromosome(chromosome)
    match_list = match_config_to_list(match_config)
    
    # 检查硬件约束
    hw_valid, hw_error = check_hardware_constraints(match_config)
    if not hw_valid:
        return -1000, {'feasible': False, 'error': hw_error}
    
    tree = model.get('tree_structure')
    
    try:
        rule_count = calculate_entries_count(tree, match_list, FEATURE_ORDER)
    except:
        rule_count = model.get('n_leaves', 100)
    
    predictor = get_predictor()
    resource = predictor.predict(match_list, rule_count)
    
    # 检查资源硬约束
    if not check_resource_constraints(resource, config):
        # 不可行解，返回大惩罚
        return -500, {
            'feasible': False,
            'match_config': match_config,
            'rule_count': rule_count,
            'resource': resource
        }
    
    # 获取该深度的F1分数
    f1 = DEPTH_F1_MAP.get(depth, 0.85)
    
    # 计算加权适应度
    fitness, fitness_details = calculate_weighted_fitness(
        f1, resource['tcam_pct'], rule_count, config
    )
    
    return fitness, {
        'feasible': True,
        'match_config': match_config,
        'rule_count': rule_count,
        'resource': resource,
        'fitness_details': fitness_details
    }


def format_match_config(match_config):
    """格式化匹配配置"""
    names = {0: 'exact', 1: 'ternary', 2: 'range', 3: 'lpm'}
    return {k: names.get(v, str(v)) for k, v in match_config.items()}


def run_ga_for_depth(depth, config):
    """在固定深度下运行遗传算法"""
    try:
        model = load_model(depth)
    except FileNotFoundError as e:
        return {'depth': depth, 'feasible': False, 'error': str(e)}
    
    params = GA_PARAMS.copy()
    population = initialize_population(params['population_size'])
    
    best_chromosome = None
    best_fitness = float('-inf')
    best_details = None
    
    for gen in range(params['generations']):
        fitness_results = []
        for chrom in population:
            fit, details = fitness_function(chrom, model, config, depth)
            fitness_results.append((fit, details))
        
        fitness_values = [r[0] for r in fitness_results]
        
        for i, (fit, details) in enumerate(fitness_results):
            if fit > best_fitness:
                best_fitness = fit
                best_chromosome = population[i].copy()
                best_details = details
        
        population = create_next_generation(population, fitness_values, params)
    
    if best_chromosome is None or best_details is None:
        return {'depth': depth, 'feasible': False, 'fitness': best_fitness}
    
    f1 = DEPTH_F1_MAP.get(depth, 0.85)
    resource = best_details.get('resource', {})
    fit_det = best_details.get('fitness_details', {})
    
    return {
        'depth': depth,
        'feasible': best_details.get('feasible', False),
        'fitness': best_fitness,
        'f1': f1,
        'rule_count': best_details.get('rule_count', 0),
        'tcam_pct': resource.get('tcam_pct', 0),
        'sram_pct': resource.get('sram_pct', 0),
        'stages': resource.get('stages', 0),
        'tcam_util': fit_det.get('tcam_util', 0),
        'latency_factor': fit_det.get('latency_factor', 0),
        'match_config': best_details.get('match_config', {}),
        'match_config_readable': format_match_config(best_details.get('match_config', {}))
    }


def explore_single_depth(args):
    """并行探索入口函数"""
    depth, config = args
    return run_ga_for_depth(depth, config)
