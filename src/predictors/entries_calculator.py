# -*- coding:utf-8 -*-
"""
精准规则数计算器
完全复刻 generate_genetic_edt.py 的逻辑，确保计算结果一致
"""

import math

# ==================== 配置 ====================

# 特征位宽配置 (与 generate_genetic_edt.py 保持一致)
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

# 特征在P4表中的顺序
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

# Exact特征的常见值
EXACT_FEATURE_COMMON_VALUES = {
    'Protocol': [6, 17],
    'TCP flags (Syn)': [0, 1],
    'IPV4 Flags (DF)': [0, 1],
    'TCP flags (Reset)': [0, 1]
}

# ==================== 核心分解算法 ====================

def range_to_prefix_decomposition(start, end, width):
    """
    精确地将范围 [start, end] 分解为最小的LPM前缀集
    返回: [(value, prefix_len), ...]
    """
    if start > end:
        return []
    if start == 0 and end == (1 << width) - 1:
        return [(0, 0)]  # 完全通配
    
    prefixes = []
    while start <= end:
        bits_diff = start ^ end
        if bits_diff != 0:
            diff_bit_len = bits_diff.bit_length()
            mask_len = width - diff_bit_len
        else:
            mask_len = width
        
        if start == 0:
            lsb_len = width
        else:
            lsb = start & -start
            lsb_len = lsb.bit_length() - 1
        
        prefix_len = max(mask_len, width - lsb_len)
        
        while prefix_len < width:
            mask = ((1 << (width - prefix_len)) - 1)
            prefix_end = start | mask
            if prefix_end <= end:
                break
            prefix_len += 1
        
        prefixes.append((start, prefix_len))
        
        mask = ((1 << (width - prefix_len)) - 1)
        prefix_end = start | mask
        start = prefix_end + 1
    
    return prefixes

def range_to_ternary_decomposition(start, end, width):
    """
    精确地将范围 [start, end] 分解为最小的三元组(value, mask)集
    """
    if start > end:
        return []
    if start == 0 and end == (1 << width) - 1:
        return [(0, 0)]
    if start == end:
        return [(start, (1 << width) - 1)]
    
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

# ==================== 辅助函数 ====================

def dfs_traverse_tree(tree, path_conditions=None):
    """DFS遍历决策树，提取所有路径"""
    if path_conditions is None:
        path_conditions = []
    
    if "children" not in tree or len(tree["children"]) == 0:
        yield path_conditions.copy()
        return
    
    feature = tree["feature"]
    threshold = float(tree["threshold"])
    
    left_conditions = path_conditions + [(feature, '<=', threshold)]
    yield from dfs_traverse_tree(tree["children"][0], left_conditions)
    
    right_conditions = path_conditions + [(feature, '>', threshold)]
    yield from dfs_traverse_tree(tree["children"][1], right_conditions)

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

# ==================== 核心：计算Entries数量 ====================

def calculate_single_path_entries(feature_ranges, feature_match_types_map):
    """
    计算单条路径展开后的Entries数量
    """
    # 笛卡尔积：总数 = 特征1展开数 * 特征2展开数 * ...
    total_combinations = 1
    
    for feature in FEATURE_ORDER:
        match_type_code = feature_match_types_map.get(feature, 0) # 默认 EXACT(0)
        min_val, max_val = feature_ranges[feature]
        bit_width = FEATURE_BITS[feature]
        is_wildcard = (min_val == 0 and max_val == (1 << bit_width) - 1)
        
        expansion_count = 1
        
        # EXACT (0)
        if match_type_code == 0:
            if is_wildcard:
                # 关键修复：仅展开常见值，而非全部
                if feature in EXACT_FEATURE_COMMON_VALUES:
                    expansion_count = len(EXACT_FEATURE_COMMON_VALUES[feature])
                else:
                    expansion_count = 1
            else:
                # P4中exact match通常不能匹配范围，除非展开
                range_size = max_val - min_val + 1
                if range_size > 100:  # 限制过大展开
                    expansion_count = 100 # 惩罚性估计
                else:
                    expansion_count = range_size
                
        # TERNARY (1)
        elif match_type_code == 1:
            if is_wildcard:
                expansion_count = 1
            else:
                expansion_count = len(range_to_ternary_decomposition(min_val, max_val, bit_width))
        
        # RANGE (2)
        elif match_type_code == 2:
            if bit_width == 1: # 1位字段回退到Ternary
                if is_wildcard:
                    expansion_count = 1
                else:
                    expansion_count = len(range_to_ternary_decomposition(min_val, max_val, bit_width))
            else:
                # 硬件支持Range，占用1个Entry
                expansion_count = 1
        
        # LPM (3)
        elif match_type_code == 3:
            if is_wildcard:
                expansion_count = 1
            else:
                expansion_count = len(range_to_prefix_decomposition(min_val, max_val, bit_width))
        
        total_combinations *= expansion_count
        
        # 如果已经变得非常大，提前截断防止溢出
        if total_combinations > 1_000_000:
            return 1_000_000 # 封顶
            
    return total_combinations

def calculate_entries_count(tree_structure, feature_match_types_list, feature_list):
    """
    计算整棵树展开后的P4 Entries总数
    """
    if tree_structure is None:
        return 0
        
    # 构建特征 -> 匹配类型代码的映射
    feature_match_types_map = {}
    # print("\n[DEBUG] calculate_entries_count 接收到的匹配方式映射:")
    for i, feature in enumerate(feature_list):
        if i < len(feature_match_types_list):
            feature_match_types_map[feature] = feature_match_types_list[i]
        else:
            feature_match_types_map[feature] = 0 # Default EXACT
        # print(f"  '{feature}': {feature_match_types_map[feature]}")
        
    # print("\n[DEBUG] 正在校验 FEATURE_ORDER 与 传入Key 的一致性:")
    # for feat_in_order in FEATURE_ORDER:
    #     if feat_in_order not in feature_match_types_map:
    #         print(f"  [ERROR] FEATURE_ORDER 中的 '{feat_in_order}' 在 map 中未找到！将被默认为 EXACT(0)")
    #         # 尝试模糊匹配查找是否有类似的 Key
    #         for k in feature_match_types_map:
    #             if k.strip() == feat_in_order.strip():
    #                  print(f"   -> 发现类似的 Key: '{k}'，可能是空格或不可见字符差异。")
    #     else:
    #         print(f"  [OK] '{feat_in_order}' -> Map Value: {feature_match_types_map[feat_in_order]}")
            
    total_entries = 0
    
    # 遍历每条逻辑路径
    for path_conditions in dfs_traverse_tree(tree_structure):
        # 1. 计算该路径下每个特征的范围
        feature_ranges = aggregate_path_conditions(path_conditions)
        
        # 2. 计算该路径展开后的数量
        path_entries = calculate_single_path_entries(feature_ranges, feature_match_types_map)
        
        total_entries += path_entries
        
    return total_entries
