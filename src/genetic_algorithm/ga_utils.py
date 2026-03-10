# -*- coding: utf-8 -*-
"""
遗传算法工具函数
包含模型加载、规则展开计算、资源预测等
"""

import os
import json
import math
from .ga_config import (
    FEATURE_BITS, FEATURE_ORDER, VARIABLE_FEATURES, FIXED_FEATURES,
    MATCH_TYPE_OPTIONS, MATCH_TYPE_NAMES, JSON_MODELS_DIR
)


# ======================== 模型加载 ========================

def load_model(depth):
    """加载指定深度的决策树模型"""
    model_path = os.path.join(JSON_MODELS_DIR, f'iscx_depth_{depth}_model.json')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    with open(model_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ======================== 染色体编解码 ========================

def decode_chromosome(chromosome):
    """
    将染色体解码为匹配配置字典
    
    参数:
        chromosome: List[int] - 5个基因，对应5个可变特征
    
    返回:
        dict - {feature_name: match_type_code}
    """
    match_config = {}
    for i, feat in enumerate(VARIABLE_FEATURES):
        options = MATCH_TYPE_OPTIONS[feat]
        gene = chromosome[i] % len(options)
        match_config[feat] = options[gene]
    
    # 固定特征使用 ternary (1)
    for feat in FIXED_FEATURES:
        match_config[feat] = 1
    
    return match_config


def encode_to_chromosome(match_config):
    """将匹配配置编码为染色体"""
    chromosome = []
    for feat in VARIABLE_FEATURES:
        match_type = match_config.get(feat, 1)
        options = MATCH_TYPE_OPTIONS[feat]
        idx = options.index(match_type) if match_type in options else 0
        chromosome.append(idx)
    return chromosome


def get_match_type_name(code):
    """获取匹配类型名称"""
    return MATCH_TYPE_NAMES.get(code, 'unknown')


def format_match_config(match_config):
    """格式化匹配配置为可读字符串"""
    return {feat: get_match_type_name(code) for feat, code in match_config.items()}


# ======================== 规则展开算法 ========================

def range_to_ternary_count(start, end, width):
    """计算范围转ternary的规则数"""
    if start > end:
        return 0
    if start == 0 and end == (1 << width) - 1:
        return 1  # 完全通配
    if start == end:
        return 1  # 单值
    
    count = 0
    if start % 2 == 1:
        count += 1
        start += 1
    
    if start <= end and end % 2 == 0:
        count += 1
        end -= 1
    
    if start <= end:
        count += range_to_ternary_count(start >> 1, end >> 1, width - 1)
    
    return count


def range_to_lpm_count(start, end, width):
    """计算范围转LPM的规则数"""
    if start > end:
        return 0
    if start == 0 and end == (1 << width) - 1:
        return 1
    
    count = 0
    current = start
    while current <= end:
        # 找到最大的2^k块
        best_k = 0
        for k in range(width + 1):
            block_size = 1 << k
            if current % block_size == 0 and current + block_size - 1 <= end:
                best_k = k
        
        count += 1
        current += (1 << best_k)
    
    return count


# ======================== 树遍历 ========================

def dfs_traverse_tree(tree, path_conditions=None):
    """DFS遍历决策树，提取所有路径"""
    if path_conditions is None:
        path_conditions = []
    
    if "children" not in tree or len(tree.get("children", [])) == 0:
        yield path_conditions.copy()
        return
    
    feature = tree.get("feature")
    threshold = float(tree.get("threshold", 0))
    
    children = tree.get("children", [])
    if len(children) >= 1:
        left_conditions = path_conditions + [(feature, '<=', threshold)]
        yield from dfs_traverse_tree(children[0], left_conditions)
    
    if len(children) >= 2:
        right_conditions = path_conditions + [(feature, '>', threshold)]
        yield from dfs_traverse_tree(children[1], right_conditions)


def aggregate_path_conditions(path_conditions):
    """聚合路径条件，计算每个特征的范围"""
    feature_ranges = {f: (0, (1 << FEATURE_BITS[f]) - 1) for f in FEATURE_ORDER}
    
    for feature, operator, threshold in path_conditions:
        if feature not in feature_ranges:
            continue
        
        threshold_int = int(threshold)
        current_min, current_max = feature_ranges[feature]
        
        if operator == '<=':
            new_max = min(current_max, threshold_int)
            feature_ranges[feature] = (current_min, new_max)
        elif operator == '>':
            new_min = max(current_min, threshold_int + 1)
            feature_ranges[feature] = (new_min, current_max)
    
    return feature_ranges


# ======================== 规则展开计算 ========================

def calculate_single_path_entries(feature_ranges, match_config):
    """计算单条路径展开后的规则数"""
    total = 1
    
    for feature in FEATURE_ORDER:
        match_type = match_config.get(feature, 1)  # 默认ternary
        min_val, max_val = feature_ranges[feature]
        bit_width = FEATURE_BITS[feature]
        is_wildcard = (min_val == 0 and max_val == (1 << bit_width) - 1)
        
        expansion = 1
        
        if is_wildcard:
            expansion = 1  # 通配符始终是1
        elif match_type == 0:  # EXACT
            range_size = max_val - min_val + 1
            expansion = min(range_size, 100)  # 限制
        elif match_type == 1:  # TERNARY
            expansion = range_to_ternary_count(min_val, max_val, bit_width)
        elif match_type == 2:  # RANGE
            if bit_width == 1:
                expansion = range_to_ternary_count(min_val, max_val, bit_width)
            else:
                expansion = 1  # 硬件Range支持
        elif match_type == 3:  # LPM
            expansion = range_to_lpm_count(min_val, max_val, bit_width)
        
        total *= max(1, expansion)
        
        if total > 100000:
            return 100000  # 封顶
    
    return total


def calculate_rule_expansion(model, match_config):
    """
    计算给定匹配配置下的总规则展开数
    
    参数:
        model: dict - 模型JSON对象
        match_config: dict - {feature: match_type_code}
    
    返回:
        int - 总规则数
    """
    tree = model.get('tree_structure')
    if tree is None:
        return 0
    
    total_entries = 0
    valid_paths = 0
    
    for path_conditions in dfs_traverse_tree(tree):
        feature_ranges = aggregate_path_conditions(path_conditions)
        
        # 检查路径是否有效（所有范围的min <= max）
        valid = all(ranges[0] <= ranges[1] for ranges in feature_ranges.values())
        if not valid:
            continue
        
        valid_paths += 1
        path_entries = calculate_single_path_entries(feature_ranges, match_config)
        total_entries += path_entries
    
    return total_entries


# ======================== 资源预测 ========================

def predict_resource(rule_count, match_config):
    """
    预测资源消耗
    
    简化模型：
    - TCAM占用主要取决于规则数和匹配类型
    - ternary/range使用TCAM，exact/lpm主要使用SRAM
    """
    # 统计匹配类型
    ternary_count = sum(1 for v in match_config.values() if v == 1)
    range_count = sum(1 for v in match_config.values() if v == 2)
    
    # 每条规则的资源消耗（简化估算）
    # Ternary规则约占用更多TCAM
    if ternary_count >= 5:
        tcam_per_rule = 0.008  # KB
    elif range_count >= 2:
        tcam_per_rule = 0.005
    else:
        tcam_per_rule = 0.006
    
    sram_per_rule = 0.002  # KB
    
    tcam_kb = rule_count * tcam_per_rule
    sram_kb = rule_count * sram_per_rule
    
    # Stages估算：每2000条规则大约1个stage
    stages = max(1, (rule_count + 1999) // 2000)
    
    return {
        'tcam_kb': tcam_kb,
        'sram_kb': sram_kb,
        'stages': stages,
        'rule_count': rule_count
    }


def check_feasibility(resource, config):
    """检查资源是否符合约束"""
    return (
        resource['tcam_kb'] <= config['tcam_kb'] and
        resource['sram_kb'] <= config['sram_kb'] and
        resource['stages'] <= config['stages']
    )


def calculate_utilization(resource, config):
    """计算资源利用率（取最大的那个维度）"""
    tcam_util = resource['tcam_kb'] / config['tcam_kb'] if config['tcam_kb'] > 0 else 0
    sram_util = resource['sram_kb'] / config['sram_kb'] if config['sram_kb'] > 0 else 0
    stage_util = resource['stages'] / config['stages'] if config['stages'] > 0 else 0
    
    return max(tcam_util, sram_util, stage_util)
