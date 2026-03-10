# -*- coding:utf-8 -*-
"""
最终核验：在禁用RANGE的限制下，确认全Ternary是否为最优解
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

def range_to_ternary_count(start, end, width):
    if start > end: return 0
    if start == 0 and end == (1 << width) - 1: return 1
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
    if start > end: return 0
    if start == 0 and end == (1 << width) - 1: return 1
    
    count = 0
    current = start
    while current <= end:
        best_k = 0
        for k in range(width + 1):
            block_size = 1 << k
            if current % block_size == 0 and current + block_size - 1 <= end:
                best_k = k
        block_size = 1 << best_k
        count += 1
        current += block_size
    return count

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

def calculate_rules(paths, config):
    total = 0
    for path, _ in paths:
        ranges = get_ranges_for_path(path)
        valid = all(ranges[f][0] <= ranges[f][1] for f in FEATURE_ORDER)
        if not valid:
            continue
        
        path_rules = 1
        for feat in FEATURE_ORDER:
            low, high = ranges[feat]
            bits = FEATURE_BITS[feat]
            m_type = config[feat]
            
            if m_type == 'ternary':
                count = range_to_ternary_count(low, high, bits)
            elif m_type == 'lpm':
                count = range_to_lpm_count(low, high, bits)
            elif m_type == 'exact':
                count = high - low + 1
            else:
                count = 1
            
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
    print("最终核验：禁用RANGE限制下的最优配置搜索")
    print("=" * 70)
    print(f"有效路径数: {len([p for p, _ in paths if all(get_ranges_for_path(p)[f][0] <= get_ranges_for_path(p)[f][1] for f in FEATURE_ORDER)])}")
    
    # 固定配置：1-bit字段必须用ternary（exact会翻倍）
    fixed_1bit = {
        'IPV4 Flags (DF)': 'ternary',
        'TCP flags (Reset)': 'ternary',
        'TCP flags (Syn)': 'ternary'
    }
    
    # 8-bit Protocol: ternary vs lpm (exact会展开为256个值，排除)
    protocol_options = ['ternary', 'lpm']
    
    # 8-bit TTL: ternary vs lpm
    ttl_options = ['ternary', 'lpm']
    
    # 16-bit 字段: ternary vs lpm (exact会展开为65536个值，排除)
    port_length_options = ['ternary', 'lpm']
    
    print("\n搜索空间:")
    print(f"  1-bit字段 (3个): 固定为ternary")
    print(f"  Protocol (8-bit): {len(protocol_options)} 种选择")
    print(f"  TTL (8-bit): {len(ttl_options)} 种选择")
    print(f"  16-bit字段 (3个): 每个 {len(port_length_options)} 种选择")
    total_configs = len(protocol_options) * len(ttl_options) * (len(port_length_options) ** 3)
    print(f"  总配置数: {total_configs}")
    
    best_config = None
    best_count = float('inf')
    all_results = []
    
    for proto_type in protocol_options:
        for ttl_type in ttl_options:
            for total_len_type in port_length_options:
                for src_port_type in port_length_options:
                    for dst_port_type in port_length_options:
                        config = fixed_1bit.copy()
                        config['Protocol'] = proto_type
                        config['Time to live'] = ttl_type
                        config['Total length'] = total_len_type
                        config['Src Port'] = src_port_type
                        config['Dst Port'] = dst_port_type
                        
                        rule_count = calculate_rules(paths, config)
                        
                        all_results.append({
                            'config': config.copy(),
                            'rules': rule_count
                        })
                        
                        if rule_count < best_count:
                            best_count = rule_count
                            best_config = config.copy()
    
    # 统计结果
    unique_counts = sorted(set(r['rules'] for r in all_results))
    print(f"\n搜索结果:")
    print(f"  不同规则数的配置数量:")
    for count in unique_counts:
        num_configs = sum(1 for r in all_results if r['rules'] == count)
        print(f"    {count} 条规则: {num_configs} 种配置")
    
    print(f"\n最优配置 (规则数最少):")
    print("-" * 70)
    for feat in FEATURE_ORDER:
        print(f"  {feat:<20}: {best_config[feat]}")
    print("-" * 70)
    print(f"  最少规则数: {best_count}")
    
    # 验证全Ternary是否是最优解之一
    all_ternary = {f: 'ternary' for f in FEATURE_ORDER}
    all_ternary_count = calculate_rules(paths, all_ternary)
    
    print(f"\n全Ternary配置验证:")
    print(f"  规则数: {all_ternary_count}")
    
    if all_ternary_count == best_count:
        print(f"  结论: 全Ternary 是最优解之一")
    else:
        print(f"  结论: 全Ternary 不是最优解 (差距: {all_ternary_count - best_count})")
    
    # 列出所有最优配置
    optimal_configs = [r for r in all_results if r['rules'] == best_count]
    print(f"\n所有最优配置 (共 {len(optimal_configs)} 种):")
    for i, res in enumerate(optimal_configs[:3], 1):
        print(f"  配置{i}:")
        variable_features = ['Protocol', 'Time to live', 'Total length', 'Src Port', 'Dst Port']
        config_str = ', '.join([f"{f.split()[0]}={res['config'][f]}" for f in variable_features])
        print(f"    {config_str}")
    if len(optimal_configs) > 3:
        print(f"    ... 还有 {len(optimal_configs) - 3} 种配置")

if __name__ == "__main__":
    main()
