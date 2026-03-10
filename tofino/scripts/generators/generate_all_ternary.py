# -*- coding:utf-8 -*-
"""
全Ternary配置 P4规则生成器 - 禁用RANGE的最优解
"""

import json
import os
import re
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

BFRT_PARAM_MAP = {
    'Total length': 'total_len',
    'Protocol': 'protocol',
    'IPV4 Flags (DF)': 'flags_df',
    'Time to live': 'ttl',
    'Src Port': 'src_port',
    'Dst Port': 'dst_port',
    'TCP flags (Reset)': 'flag_rst',
    'TCP flags (Syn)': 'flag_syn'
}

def range_to_ternary_decomposition(start, end, width):
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

def generate_ternary_rules(path_ranges, class_id):
    """生成全ternary匹配的BFRT规则"""
    # 为每个特征生成ternary分解
    feature_decomps = {}
    for feat in FEATURE_ORDER:
        low, high = path_ranges[feat]
        bits = FEATURE_BITS[feat]
        decomp = range_to_ternary_decomposition(low, high, bits)
        if not decomp:
            return []  # 无效路径
        feature_decomps[feat] = decomp
    
    # 笛卡尔积展开
    from itertools import product
    all_combos = list(product(*[feature_decomps[f] for f in FEATURE_ORDER]))
    
    cmds = []
    for combo in all_combos:
        params = []
        for i, feat in enumerate(FEATURE_ORDER):
            v, m = combo[i]
            name = BFRT_PARAM_MAP[feat]
            params.append(f"{v}")
            params.append(f"{m}")
        params.append(f"class_id={class_id}")
        cmd = f"bfrt.genetic_edt.pipe.Ingress.EDT.add_with_SetClass({', '.join(params)})\n"
        cmds.append(cmd)
    
    return cmds

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "data", "iscx_depth_5_model.json")
    
    with open(model_path, 'r', encoding='utf-8') as f:
        model = json.load(f)
    
    tree_root = model['tree_structure']
    
    output_dir = os.path.join(base_dir, "generated", "iscx_depth_5_all_ternary")
    os.makedirs(output_dir, exist_ok=True)
    
    rules_file = os.path.join(output_dir, "genetic_edt_rules.txt")
    p4_file = os.path.join(output_dir, "genetic_edt.p4")
    
    print("=" * 70)
    print("深度5模型 - 全Ternary配置规则生成")
    print("=" * 70)
    
    total_rules = 0
    with open(rules_file, 'w', encoding='utf-8') as f:
        for path, class_id in dfs_paths(tree_root):
            ranges = get_ranges_for_path(path)
            cmds = generate_ternary_rules(ranges, class_id)
            for c in cmds:
                f.write(c)
            total_rules += len(cmds)
    
    print(f"规则总数: {total_rules}")
    print(f"规则文件: {rules_file}")
    
    # 生成P4文件
    template_path = os.path.join(base_dir, "generated", "iscx_depth_5", "genetic_edt.p4")
    if os.path.exists(template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            p4_code = f.read()
        
        # 修改匹配类型为全ternary
        key_replacement = """key = {
            hdr.ipv4.total_len            : ternary;
            hdr.ipv4.protocol             : ternary;
            hdr.ipv4.flags[1:1]           : ternary;
            hdr.ipv4.ttl                  : ternary;
            meta.src_port                 : ternary;
            meta.dst_port                 : ternary;
            meta.tcp_flags[2:2]           : ternary;
            meta.tcp_flags[1:1]           : ternary;
        }"""
        
        # 替换key定义
        p4_code = re.sub(r'key\s*=\s*\{[^}]+\}', key_replacement, p4_code)
        
        # 更新size
        p4_code = re.sub(r'size\s*=\s*\d+;', f'size = {total_rules};', p4_code)
        
        with open(p4_file, 'w', encoding='utf-8') as f:
            f.write(p4_code)
        print(f"P4文件: {p4_file}")
    
    print("\n最优配置 (禁用RANGE):")
    print("-" * 50)
    for feat in FEATURE_ORDER:
        print(f"  {feat:<20}: ternary")
    print("-" * 50)
    print(f"Table Size: {total_rules}")

if __name__ == "__main__":
    main()
