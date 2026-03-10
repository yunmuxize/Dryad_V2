#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取Config3所有可行解信息并保存到TXT文件
"""

import json
from pathlib import Path

def extract_config3_solutions(json_file_path):
    """从JSON文件中提取Config3的所有可行解"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    config3_data = data["results"]["Config3"]
    solutions = config3_data["all_feasible_solutions"]
    
    return solutions, config3_data

def format_match_config_readable(match_config_readable):
    """格式化match_config_readable为字符串"""
    items = []
    for key, value in match_config_readable.items():
        items.append(f"{key}:{value}")
    return ", ".join(items)

def save_to_txt(solutions, config3_data, output_file_path):
    """保存到TXT文件，按fitness从高到低排序"""
    solutions_sorted = sorted(solutions, key=lambda x: x['fitness'], reverse=True)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("Config3 所有可行解 (按 Fitness 由高到低排序)\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"最优解深度: {config3_data['optimal_depth']}\n")
        f.write(f"最优F1分数: {config3_data['f1']}\n")
        f.write(f"可行深度范围: {min(config3_data['all_feasible_depths'])} - {max(config3_data['all_feasible_depths'])}\n")
        f.write(f"可行解总数: {len(solutions)}\n\n")
        
        f.write("-" * 100 + "\n")
        f.write(f"{'Rank':<6} {'Depth':<8} {'Fitness':<12} {'F1':<10} {'Rules':<10} {'TCAM%':<12} {'SRAM%':<12} {'Stages':<10} {'Latency':<12}\n")
        f.write("-" * 100 + "\n")
        
        for idx, sol in enumerate(solutions_sorted, 1):
            f.write(f"{idx:<6} {sol['depth']:<8} {sol['fitness']:<12.6f} {sol['f1']:<10.4f} {sol['rule_count']:<10} "
                   f"{sol['tcam_pct']:<12.4f} {sol['sram_pct']:<12.4f} {sol['stages']:<10} {sol['latency_factor']:<12.6f}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("详细解信息\n")
        f.write("=" * 100 + "\n\n")
        
        for idx, sol in enumerate(solutions_sorted, 1):
            f.write(f"解 #{idx}\n")
            f.write("-" * 100 + "\n")
            f.write(f"  Depth:           {sol['depth']}\n")
            f.write(f"  Fitness:         {sol['fitness']:.6f}\n")
            f.write(f"  F1 Score:        {sol['f1']:.4f}\n")
            f.write(f"  Rule Count:      {sol['rule_count']}\n")
            f.write(f"  TCAM Percentage: {sol['tcam_pct']:.4f}%\n")
            f.write(f"  SRAM Percentage: {sol['sram_pct']:.4f}%\n")
            f.write(f"  Stages:          {sol['stages']}\n")
            f.write(f"  TCAM Util:       {sol['tcam_util']:.6f}\n")
            f.write(f"  SRAM Util:       {sol['sram_util']}\n")
            f.write(f"  Stage Util:      {sol['stage_util']}\n")
            f.write(f"  Latency Factor:  {sol['latency_factor']:.6f}\n")
            f.write(f"  Match Config:    {format_match_config_readable(sol['match_config_readable'])}\n")
            f.write("\n")

def main():
    json_file = "/mnt/8T/xgr/yaorunze/Dryad/src/ga_results.json"
    output_file = "/mnt/8T/xgr/yaorunze/Dryad/src/genetic_algorithm/config3_solutions.txt"
    
    print(f"正在从JSON文件中提取Config3信息...")
    solutions, config3_data = extract_config3_solutions(json_file)
    
    print(f"正在保存到文件: {output_file}")
    save_to_txt(solutions, config3_data, output_file)
    
    print(f"✓ 成功保存Config3的所有可行解信息")
    print(f"  - 可行解数量: {len(solutions)}")
    print(f"  - 最优深度: {config3_data['optimal_depth']}")
    print(f"  - 最优F1分数: {config3_data['f1']*100:.2f}%")
    print(f"  - 输出文件: {output_file}")

if __name__ == "__main__":
    main()
