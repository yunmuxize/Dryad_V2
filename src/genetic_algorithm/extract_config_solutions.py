#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从ga_results.json中提取指定配置的所有可行解信息并保存到单独文件
"""

import json
import sys
from pathlib import Path

def extract_config_solutions(json_file_path, config_name, output_file_path=None):
    """提取指定配置的所有可行解"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    config_data = data['results'].get(config_name)
    if not config_data:
        print(f"错误: 找不到配置 {config_name}")
        return None
    
    if 'all_feasible_solutions' not in config_data:
        print(f"错误: 配置 {config_name} 没有all_feasible_solutions数据")
        return None
    
    # 构建输出数据
    output_data = {
        "config_name": config_name,
        "description": config_data.get('description', ''),
        "optimal_solution": {
            "depth": config_data.get('optimal_depth'),
            "f1": config_data.get('f1'),
            "rule_count": config_data.get('rule_count'),
            "tcam_pct": config_data.get('tcam_pct'),
            "sram_pct": config_data.get('sram_pct'),
            "stages": config_data.get('stages'),
            "tcam_util": config_data.get('tcam_util'),
            "sram_util": config_data.get('sram_util'),
            "stage_util": config_data.get('stage_util'),
            "match_config": config_data.get('match_config', {}),
            "match_config_readable": config_data.get('match_config_readable', {})
        },
        "all_feasible_depths": config_data.get('all_feasible_depths', []),
        "all_feasible_solutions": config_data.get('all_feasible_solutions', []),
        "duration_seconds": config_data.get('duration_seconds', 0)
    }
    
    # 确定输出文件路径
    if output_file_path is None:
        output_file_path = f"{config_name.lower()}_all_solutions.json"
    
    # 保存到文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 成功保存配置 {config_name} 的所有可行解信息")
    print(f"  - 可行解数量: {len(output_data['all_feasible_solutions'])}")
    print(f"  - 最优深度: {output_data['optimal_solution']['depth']}")
    print(f"  - 最优F1分数: {output_data['optimal_solution']['f1']*100:.2f}%")
    print(f"  - 输出文件: {output_file_path}")
    
    return output_data

def main():
    json_file = "/mnt/8T/xgr/yaorunze/Dryad/src/ga_results.json"
    
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    else:
        config_name = "Config3"
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = None
    
    extract_config_solutions(json_file, config_name, output_file)

if __name__ == "__main__":
    main()
