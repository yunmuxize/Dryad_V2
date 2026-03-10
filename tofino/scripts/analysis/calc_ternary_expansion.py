# -*- coding:utf-8 -*-
"""
精确规则展开计算器 - 计算禁用RANGE时的实际规则数
"""

import json
import os
import numpy as np

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

# 全部使用 ternary 匹配
MATCH_CONFIG = {f: 'ternary' for f in FEATURE_ORDER}

def range_to_ternary_decomposition(start, end, width):
    """将 [start, end] 范围分解为 (value, mask) 对列表"""
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

def count_exact_expansion(path_ranges):
    """精确计算一条路径使用全 ternary 时展开的规则数"""
    total = 1
    breakdown = {}
    for feat in FEATURE_ORDER:
        low, high = path_ranges[feat]
        bits = FEATURE_BITS[feat]
        
        if low > high:
            return 0, {}  # 无效路径
        
        decomp = range_to_ternary_decomposition(low, high, bits)
        count = len(decomp)
        breakdown[feat] = count
        total *= count
    
    return total, breakdown

def main():
    model_path = r"C:\Users\86177\OneDrive\Desktop\Dryad_V2\Dryad\tofino\data\iscx_depth_5_model.json"
    with open(model_path, 'r', encoding='utf-8') as f:
        model = json.load(f)
    
    tree_root = model['tree_structure']
    
    print("=" * 70)
    print("深度5模型 - 全Ternary配置精确规则展开计算")
    print("=" * 70)
    
    paths = list(dfs_paths(tree_root))
    print(f"树路径数 (有效叶节点): {len(paths)}")
    
    total_rules = 0
    path_details = []
    
    for path, class_id in paths:
        ranges = get_ranges_for_path(path)
        expansion, breakdown = count_exact_expansion(ranges)
        if expansion > 0:
            total_rules += expansion
            path_details.append({
                'class_id': class_id,
                'expansion': expansion,
                'breakdown': breakdown
            })
    
    print(f"\n有效路径数: {len(path_details)}")
    print(f"展开后总规则数: {total_rules}")
    
    # 按展开数排序，显示最大的5条
    path_details.sort(key=lambda x: x['expansion'], reverse=True)
    print("\n展开最多的5条路径:")
    print("-" * 70)
    for i, pd in enumerate(path_details[:5]):
        print(f"  路径 {i+1} (Class {pd['class_id']}): 展开为 {pd['expansion']} 条规则")
        for feat, count in pd['breakdown'].items():
            if count > 1:
                print(f"    - {feat}: {count} 条ternary规则")
    
    # 分析哪些特征贡献了最多的展开
    print("\n特征展开贡献分析:")
    print("-" * 50)
    for feat in FEATURE_ORDER:
        max_count = max(pd['breakdown'].get(feat, 1) for pd in path_details)
        avg_count = sum(pd['breakdown'].get(feat, 1) for pd in path_details) / len(path_details)
        print(f"  {feat:<20}: 最大 {max_count}, 平均 {avg_count:.2f}")

if __name__ == "__main__":
    main()
