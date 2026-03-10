# -*- coding: utf-8 -*-
"""
遗传算法多核并行主程序 (V2)

全深度探索 + Tofino真实预测模型
"""

import os
import sys
import time
import json
from multiprocessing import Pool, freeze_support

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic_algorithm.ga_config import CONFIGS, DEPTH_RANGE, GA_PARAMS, DEPTH_F1_MAP
from genetic_algorithm.ga_explorer import explore_single_depth


def run_parallel_exploration(config, depth_range, num_cores=None):
    """并行探索所有深度"""
    # 默认使用配置中的核心数
    if num_cores is None:
        num_cores = GA_PARAMS.get('n_jobs', 4)
    
    tasks = [(d, config) for d in depth_range]
    
    # 限制进程数不超过任务数，且不超过物理限制
    actual_cores = min(num_cores, len(tasks))
    print(f"Starting parallel exploration with {actual_cores} processes...")

    
    with Pool(processes=actual_cores) as pool:
        results = pool.map(explore_single_depth, tasks)
    
    return results


def select_optimal_depth(results):
    """
    选择最优深度
    
    策略: 选择Fitness评分最高的可行解（修复前bug：原逻辑错误地选择最大深度）
    
    修复说明:
    - 原逻辑: max(feasible, key=lambda x: x['depth']) - 仅选择最深解，忽略适应度
    - 新逻辑: max(feasible, key=lambda x: x['fitness']) - 选择最高适应度解
    - 原因: 导致所有Config均倾向于选择最大深度(38)，而非最优解
    """
    # 筛选可行解
    feasible = [r for r in results if r.get('feasible', False)]
    
    if not feasible:
        return None
    
    # 修复: 选择最高适应度的解，而非最大深度
    # Fitness = α × F1 + β × TCAM_Util - γ × Latency_Factor
    # 高适应度意味着在精度、资源利用率和延迟之间达到最佳平衡
    return max(feasible, key=lambda x: x['fitness'])


def print_separator(char='=', length=80):
    print(char * length)


def format_result_table(results, config):
    """格式化结果表格"""
    print(f"\n{'Depth':>5} | {'Status':^8} | {'Rules':>8} | {'TCAM%':>7} | {'SRAM%':>7} | {'Stages':>6} | {'T-Util':>7} | {'S-Util':>7} | {'St-Util':>7} | {'F1':>6}")
    print("-" * 100)
    
    for r in sorted(results, key=lambda x: x.get('depth', 0)):
        depth = r.get('depth', 0)
        
        if r.get('feasible', False):
            status = "OK"
            rules = r.get('rule_count', 0)
            tcam = r.get('tcam_pct', 0)
            sram = r.get('sram_pct', 0)
            stages = r.get('stages', 0)
            t_util = r.get('tcam_util', 0) * 100
            s_util = r.get('sram_util', 0) * 100
            st_util = r.get('stage_util', 0) * 100
            f1 = r.get('f1', 0) * 100
            print(f"{depth:>5} | {status:^8} | {rules:>8} | {tcam:>7.2f} | {sram:>7.2f} | {stages:>6} | {t_util:>6.1f}% | {s_util:>6.1f}% | {st_util:>6.1f}% | {f1:>5.2f}%")
        else:
            error = r.get('error', 'Exceeded')[:15]
            print(f"{depth:>5} | {'FAIL':^8} | {'-':>8} | {'-':>7} | {'-':>7} | {'-':>6} | {'-':>7} | {'-':>7} | {'-':>7} | {'-':>6}  ({error})")


def main():
    """主函数"""
    
    # 从配置获取核心数
    num_cores = GA_PARAMS.get('n_jobs', 4)
    
    print_separator()
    print("Genetic Algorithm Multi-Core Optimization System (V2)")
    print(f"Parallel Cores: {num_cores}")
    print(f"Depth Range: {min(DEPTH_RANGE)} to {max(DEPTH_RANGE)} ({len(DEPTH_RANGE)} depths)")
    print_separator()
    
    total_start_time = time.time()
    
    all_results = {}
    config_times = {}
    
    for config in CONFIGS:
        print(f"\n{'='*80}")
        print(f"Processing: {config['name']} - {config.get('description', '')}")
        print(f"Limits: TCAM={config['tcam_pct_max']:.1f}%, SRAM={config['sram_pct']:.1f}%, Stages={config['stages']}")
        print('='*80)
        
        config_start_time = time.time()
        
        print(f"\nLaunching parallel exploration for {len(DEPTH_RANGE)} depths...")
        results = run_parallel_exploration(config, DEPTH_RANGE, num_cores)
        
        config_end_time = time.time()
        config_duration = config_end_time - config_start_time
        config_times[config['name']] = config_duration
        
        # 显示详细结果表格
        format_result_table(results, config)
        
        # 选择最优深度
        optimal = select_optimal_depth(results)
        
        all_results[config['name']] = {
            'optimal': optimal,
            'all_results': results,
            'duration_seconds': config_duration
        }
        
        if optimal:
            print(f"\n>> OPTIMAL: Depth={optimal['depth']}, F1={optimal['f1']*100:.2f}%, Rules={optimal['rule_count']}")
            print(f"   Resource: TCAM={optimal['tcam_pct']:.2f}%, SRAM={optimal['sram_pct']:.2f}%, Stages={optimal['stages']}")
            print(f"   Match Config: {optimal['match_config_readable']}")
        else:
            print(f"\n>> NO FEASIBLE SOLUTION FOUND")
        
        print(f"   Time: {config_duration:.2f}s")
    
    total_duration = time.time() - total_start_time
    
    # 最终汇总
    print("\n")
    print_separator()
    print("EXPERIMENT SUMMARY")
    print_separator()
    
    print(f"\n{'Config':^10} | {'Depth':>6} | {'F1':>7} | {'Rules':>8} | {'TCAM%':>7} | {'SRAM%':>7} | {'Stages':>6} | {'Time':>8}")
    print("-" * 80)
    
    for config_name, result in all_results.items():
        opt = result.get('optimal')
        duration = result.get('duration_seconds', 0)
        
        if opt:
            print(f"{config_name:^10} | {opt['depth']:>6} | {opt['f1']*100:>6.2f}% | {opt['rule_count']:>8} | {opt['tcam_pct']:>7.2f} | {opt['sram_pct']:>7.2f} | {opt['stages']:>6} | {duration:>7.2f}s")
        else:
            print(f"{config_name:^10} | {'-':>6} | {'-':>7} | {'-':>8} | {'-':>7} | {'-':>7} | {'-':>6} | {duration:>7.2f}s")
    
    print(f"\nTotal Time: {total_duration:.2f}s")
    print_separator()
    
    # 保存结果到JSON
    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_file = os.path.join(output_dir, 'ga_results.json')
    
    serializable_results = {}
    for k, v in all_results.items():
        opt = v.get('optimal', {})
        all_results_list = v.get('all_results', [])
        
        # 保存所有可行解的完整信息（包括match_config）
        feasible_solutions = []
        for r in all_results_list:
            if r.get('feasible', False):
                feasible_solutions.append({
                    'depth': r.get('depth'),
                    'fitness': r.get('fitness'),
                    'f1': r.get('f1'),
                    'rule_count': r.get('rule_count'),
                    'tcam_pct': r.get('tcam_pct'),
                    'sram_pct': r.get('sram_pct'),
                    'stages': r.get('stages'),
                    'tcam_util': r.get('tcam_util'),
                    'sram_util': r.get('sram_util'),
                    'stage_util': r.get('stage_util'),
                    'latency_factor': r.get('latency_factor'),
                    'match_config': r.get('match_config', {}),
                    'match_config_readable': r.get('match_config_readable', {})
                })
        
        serializable_results[k] = {
            'optimal_depth': opt.get('depth') if opt else None,
            'f1': opt.get('f1') if opt else None,
            'rule_count': opt.get('rule_count') if opt else None,
            'tcam_pct': opt.get('tcam_pct') if opt else None,
            'sram_pct': opt.get('sram_pct') if opt else None,
            'stages': opt.get('stages') if opt else None,
            'tcam_util': opt.get('tcam_util') if opt else None,
            'sram_util': opt.get('sram_util') if opt else None,
            'stage_util': opt.get('stage_util') if opt else None,
            'match_config': opt.get('match_config', {}) if opt else {},
            'match_config_readable': opt.get('match_config_readable', {}) if opt else {},
            'duration_seconds': v.get('duration_seconds', 0),
            'all_feasible_depths': [r.get('depth') for r in all_results_list if r.get('feasible', False)],
            'all_feasible_solutions': feasible_solutions
        }
    
    output_data = {
        'results': serializable_results,
        'total_duration_seconds': total_duration,
        'config_durations': config_times,
        'depth_range': [min(DEPTH_RANGE), max(DEPTH_RANGE)],
        'num_cores': num_cores
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    return all_results


if __name__ == '__main__':
    freeze_support()
    main()
