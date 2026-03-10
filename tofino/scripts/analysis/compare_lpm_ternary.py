# -*- coding:utf-8 -*-
"""
LPM vs Ternary 规则展开对比分析
分析 Src Port, Dst Port, Time to live 使用 LPM 是否比 Ternary 更优
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

def range_to_ternary_decomposition(start, end, width):
    """Ternary分解：返回 (value, mask) 对列表"""
    if start > end: return []
    if start == 0 and end == (1 << width) - 1: return [(0, 0)]
    if start == end: return [(start, (1 << width) - 1)]
    
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

def range_to_lpm_decomposition(start, end, width):
    """
    LPM分解：将 [start, end] 范围分解为前缀匹配规则
    LPM只能匹配形如 prefix/* 的范围，即从某个值开始到该前缀能覆盖的最大值
    
    例如：[0, 7] 在 4-bit 下可以用 0*** (prefix=0, len=1) 表示
    但 [3, 12] 需要多条LPM规则
    """
    if start > end: return []
    if start == 0 and end == (1 << width) - 1: 
        return [(0, 0)]  # prefix=0, prefix_len=0 表示全通配
    
    result = []
    current = start
    
    while current <= end:
        # 找到从current开始的最大对齐块
        # 块大小必须是2的幂，且current必须是块大小的整数倍
        max_block_size = 1
        while max_block_size <= (end - current + 1):
            # 检查current是否对齐到这个块大小
            if current % (max_block_size * 2) == 0 and current + max_block_size * 2 - 1 <= end:
                max_block_size *= 2
            else:
                break
        
        # 计算前缀长度
        prefix_len = width
        temp = max_block_size
        while temp > 1:
            prefix_len -= 1
            temp //= 2
        
        result.append((current, prefix_len))
        current += max_block_size
    
    return result

def count_ternary(start, end, width):
    return len(range_to_ternary_decomposition(start, end, width))

def count_lpm(start, end, width):
    return len(range_to_lpm_decomposition(start, end, width))

def dfs_paths(node, path_conditions=None):
    if path_conditions is None: path_conditions = []
    if "children" not in node or not node["children"]:
        class_id = int(np.argmax(node["value"]))
        yield path_conditions, class_id
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

def calculate_total_rules(paths, config):
    """计算给定配置下的总规则数"""
    total = 0
    for path, _ in paths:
        ranges = get_ranges_for_path(path)
        
        # 检查有效性
        valid = True
        for f in FEATURE_ORDER:
            if ranges[f][0] > ranges[f][1]:
                valid = False
                break
        if not valid:
            continue
        
        path_rules = 1
        for feat in FEATURE_ORDER:
            low, high = ranges[feat]
            bits = FEATURE_BITS[feat]
            match_type = config.get(feat, 'ternary')
            
            if match_type == 'ternary':
                count = count_ternary(low, high, bits)
            elif match_type == 'lpm':
                count = count_lpm(low, high, bits)
            else:
                count = count_ternary(low, high, bits)
            
            path_rules *= max(1, count)
        
        total += path_rules
    return total

def main():
    model_path = r"C:\Users\86177\OneDrive\Desktop\Dryad_V2\Dryad\tofino\data\iscx_depth_5_model.json"
    with open(model_path, 'r', encoding='utf-8') as f:
        model = json.load(f)
    
    tree_root = model['tree_structure']
    paths = list(dfs_paths(tree_root))
    
    print("=" * 70)
    print("LPM vs Ternary 规则展开对比分析")
    print("=" * 70)
    print(f"有效路径数: {len([p for p, _ in paths if all(get_ranges_for_path(p)[f][0] <= get_ranges_for_path(p)[f][1] for f in FEATURE_ORDER)])}")
    
    # 基准：全Ternary
    base_config = {f: 'ternary' for f in FEATURE_ORDER}
    base_rules = calculate_total_rules(paths, base_config)
    print(f"\n基准配置 (全Ternary): {base_rules} 条规则")
    
    # 测试单个特征换成LPM
    test_features = ['Src Port', 'Dst Port', 'Time to live', 'Total length']
    
    print("\n" + "-" * 70)
    print("单特征替换为 LPM 的对比:")
    print("-" * 70)
    print(f"{'特征名称':<20} | {'Ternary规则数':<15} | {'LPM规则数':<15} | {'变化'}")
    print("-" * 70)
    
    for feat in test_features:
        lpm_config = base_config.copy()
        lpm_config[feat] = 'lpm'
        lpm_rules = calculate_total_rules(paths, lpm_config)
        
        diff = lpm_rules - base_rules
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"{feat:<20} | {base_rules:<15} | {lpm_rules:<15} | {diff_str}")
    
    # 详细分析每条路径的展开情况
    print("\n" + "-" * 70)
    print("各特征在每条路径上的 Ternary vs LPM 展开对比:")
    print("-" * 70)
    
    for feat in test_features:
        print(f"\n{feat} ({FEATURE_BITS[feat]}-bit):")
        bits = FEATURE_BITS[feat]
        
        ternary_counts = []
        lpm_counts = []
        
        for path, class_id in paths:
            ranges = get_ranges_for_path(path)
            if ranges[feat][0] > ranges[feat][1]:
                continue
            
            low, high = ranges[feat]
            t_count = count_ternary(low, high, bits)
            l_count = count_lpm(low, high, bits)
            ternary_counts.append(t_count)
            lpm_counts.append(l_count)
        
        if ternary_counts:
            print(f"  Ternary - 平均: {np.mean(ternary_counts):.2f}, 最大: {max(ternary_counts)}, 总: {sum(ternary_counts)}")
            print(f"  LPM     - 平均: {np.mean(lpm_counts):.2f}, 最大: {max(lpm_counts)}, 总: {sum(lpm_counts)}")
            
            # 显示LPM更优的情况
            better_count = sum(1 for t, l in zip(ternary_counts, lpm_counts) if l < t)
            worse_count = sum(1 for t, l in zip(ternary_counts, lpm_counts) if l > t)
            same_count = sum(1 for t, l in zip(ternary_counts, lpm_counts) if l == t)
            print(f"  比较: LPM更优 {better_count}条, LPM更差 {worse_count}条, 相同 {same_count}条")

if __name__ == "__main__":
    main()
