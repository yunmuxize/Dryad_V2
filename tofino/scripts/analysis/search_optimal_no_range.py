# -*- coding:utf-8 -*-
"""
最优匹配方式搜索器 - 在禁用RANGE匹配条件下寻找最少规则展开的配置
"""

import json
import os
import numpy as np
from itertools import product

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
    'Total length', 'Protocol', 'IPV4 Flags (DF)', 'Time to live',
    'Src Port', 'Dst Port', 'TCP flags (Reset)', 'TCP flags (Syn)'
]

# 允许的匹配类型 (不包含 range)
MATCH_TYPES = ['ternary', 'lpm', 'exact']

def range_to_ternary_count(start, end, width):
    """计算将 [start, end] 范围转换为 ternary 规则需要多少条"""
    if start > end: return 0
    if start == 0 and end == (1 << width) - 1: return 1  # 全通配
    if start == end: return 1
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
    """计算将 [start, end] 范围转换为 LPM 规则需要多少条 (近似)"""
    if start > end: return 0
    if start == 0 and end == (1 << width) - 1: return 1
    if start == end: return 1
    # LPM 比 ternary 复杂，这里做保守估计
    return range_to_ternary_count(start, end, width) * 2

def dfs_paths(node, path_conditions=None):
    if path_conditions is None: path_conditions = []
    if "children" not in node or not node["children"]:
        yield path_conditions
        return
    feat = node["feature"]
    thres = float(node["threshold"])
    yield from dfs_paths(node["children"][0], path_conditions + [(feat, '<=', thres)])
    yield from dfs_paths(node["children"][1], path_conditions + [(feat, '>', thres)])

def get_ranges_for_path(path):
    ranges = {f: [0, (1 << FEATURE_BITS[f]) - 1] for f in FEATURE_ORDER}
    for feat, op, thres in path:
        t_int = int(thres)
        if op == '<=':
            ranges[feat][1] = min(ranges[feat][1], t_int)
        else:
            ranges[feat][0] = max(ranges[feat][0], t_int + 1)
    return ranges

def estimate_rule_expansion(path_ranges, match_config):
    """估算单条路径在给定配置下展开为多少条规则"""
    total = 1
    for feat in FEATURE_ORDER:
        low, high = path_ranges[feat]
        bits = FEATURE_BITS[feat]
        m_type = match_config[feat]
        
        is_full = (low == 0 and high == (1 << bits) - 1)
        
        if m_type == 'ternary':
            if is_full:
                count = 1
            else:
                count = range_to_ternary_count(low, high, bits)
        elif m_type == 'lpm':
            if is_full:
                count = 1
            else:
                count = range_to_lpm_count(low, high, bits)
        elif m_type == 'exact':
            if is_full:
                count = high - low + 1  # 需要枚举所有值
            else:
                count = high - low + 1
        else:
            count = 1
        
        total *= max(1, count)
    return total

def search_optimal_config(tree_root):
    """搜索所有可能的匹配配置，找到规则展开最少的"""
    paths = list(dfs_paths(tree_root))
    print(f"树路径数: {len(paths)}")
    
    # 对于 1-bit 字段，ternary 是唯一合理选择
    # 对于 8-bit 和 16-bit 字段，需要在 ternary/lpm 中选择
    
    # 固定配置: 1-bit 字段必须用 ternary
    fixed_config = {
        'IPV4 Flags (DF)': 'ternary',
        'TCP flags (Reset)': 'ternary',
        'TCP flags (Syn)': 'ternary'
    }
    
    # 可变配置: 其他字段在 ternary/lpm 中选择
    variable_features = ['Total length', 'Protocol', 'Time to live', 'Src Port', 'Dst Port']
    
    best_config = None
    best_count = float('inf')
    
    # 由于 exact 对于 16-bit 字段会爆炸，只考虑 ternary 和 lpm
    options = ['ternary', 'lpm']
    
    for combo in product(options, repeat=len(variable_features)):
        config = fixed_config.copy()
        for i, feat in enumerate(variable_features):
            config[feat] = combo[i]
        
        total_rules = 0
        for path in paths:
            ranges = get_ranges_for_path(path)
            # 检查是否是有效路径
            valid = True
            for feat in FEATURE_ORDER:
                if ranges[feat][0] > ranges[feat][1]:
                    valid = False
                    break
            if not valid:
                continue
            
            expansion = estimate_rule_expansion(ranges, config)
            total_rules += expansion
        
        if total_rules < best_count:
            best_count = total_rules
            best_config = config.copy()
    
    return best_config, best_count

def main():
    model_path = r"C:\Users\86177\OneDrive\Desktop\Dryad_V2\Dryad\tofino\data\iscx_depth_5_model.json"
    with open(model_path, 'r', encoding='utf-8') as f:
        model = json.load(f)
    
    tree_root = model['tree_structure']
    
    print("=" * 70)
    print("深度5模型 - 无RANGE匹配的最优配置搜索")
    print("=" * 70)
    
    best_config, best_count = search_optimal_config(tree_root)
    
    print("\n最优配置 (禁用RANGE):")
    print("-" * 50)
    for feat in FEATURE_ORDER:
        print(f"  {feat:<20}: {best_config[feat]}")
    print("-" * 50)
    print(f"理论最少规则数: {best_count}")

if __name__ == "__main__":
    main()
