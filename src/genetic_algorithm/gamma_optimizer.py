#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config1 Gamma参数优化探索脚本（修正版）

正确逻辑：
1. 使用92核并行探索所有深度（4-38）
2. 每一轮实验改变的是同一config（Config1）的gamma值
3. 确保最终最优解的配置稳定在深度5
"""

import os
import sys
import time
import json
from multiprocessing import Pool, freeze_support
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic_algorithm.ga_config import GA_PARAMS, DEPTH_RANGE, DEPTH_F1_MAP
from genetic_algorithm.ga_explorer import run_ga_for_depth


def run_gamma_experiment(gamma_value, config, depth_range, num_cores=92):
    """
    运行单个gamma值的完整深度探索实验
    
    参数:
        gamma_value: gamma参数值
        config: 配置对象（Config1）
        depth_range: 深度范围（4-38）
        num_cores: 并行核心数（默认92）
    
    返回:
        实验结果，包含所有深度的探索结果
    """
    # 更新config的gamma值
    config = config.copy()
    config['gamma'] = gamma_value
    
    print(f"\n{'='*80}")
    print(f"开始Gamma={gamma_value:.3f}的完整深度探索实验")
    print(f"配置: {config['name']}")
    print(f"Gamma: {gamma_value:.3f}")
    print(f"深度范围: {min(depth_range)}-{max(depth_range)} ({len(depth_range)}个深度)")
    print(f"并行核心数: {num_cores}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # 创建任务列表
    tasks = [(d, config) for d in depth_range]
    
    # 限制进程数
    actual_cores = min(num_cores, len(tasks))
    print(f"启动并行探索，使用{actual_cores}个进程...")
    
    # 并行探索所有深度
    with Pool(processes=actual_cores) as pool:
        results = pool.starmap(run_ga_for_depth, tasks)
    
    elapsed_time = time.time() - start_time
    
    # 筛选可行解
    feasible_results = [r for r in results if r.get('feasible', False)]
    
    # 找到最优解（最高适应度）
    if feasible_results:
        optimal = max(feasible_results, key=lambda x: x['fitness'])
        print(f"\n>> 最优解: Depth={optimal['depth']}, F1={optimal['f1']*100:.2f}%, Rules={optimal['rule_count']}")
        print(f"   适应度: {optimal['fitness']:.6f}")
        print(f"   资源: TCAM={optimal['tcam_pct']:.2f}%, SRAM={optimal['sram_pct']:.2f}%, Stages={optimal['stages']}")
        print(f"   匹配配置: {optimal['match_config_readable']}")
    else:
        optimal = None
        print(f"\n>> 没有可行解！")
    
    print(f"   耗时: {elapsed_time:.2f}秒")
    
    # 返回实验结果
    return {
        'gamma': gamma_value,
        'optimal_depth': optimal['depth'] if optimal else None,
        'optimal_fitness': optimal['fitness'] if optimal else None,
        'optimal_f1': optimal['f1'] if optimal else None,
        'optimal_rule_count': optimal['rule_count'] if optimal else None,
        'optimal_tcam_pct': optimal['tcam_pct'] if optimal else None,
        'optimal_sram_pct': optimal['sram_pct'] if optimal else None,
        'optimal_stages': optimal['stages'] if optimal else None,
        'optimal_match_config': optimal['match_config'] if optimal else {},
        'optimal_match_config_readable': optimal['match_config_readable'] if optimal else {},
        'all_results': results,
        'feasible_count': len(feasible_results),
        'elapsed_time': elapsed_time
    }


def analyze_gamma_experiments(experiment_results, target_depth=5):
    """
    分析所有gamma实验结果，找到最优gamma值
    
    参数:
        experiment_results: 所有gamma实验的结果列表
        target_depth: 目标深度（默认5）
    
    返回:
        最优gamma值及分析报告
    """
    print(f"\n{'='*80}")
    print("Gamma参数优化分析")
    print(f"目标深度: {target_depth}")
    print(f"{'='*80}")
    
    # 统计信息
    total_tests = len(experiment_results)
    feasible_experiments = [e for e in experiment_results if e['optimal_depth'] is not None]
    
    print(f"\n总实验数: {total_tests}")
    print(f"可行实验数: {len(feasible_experiments)}")
    
    if not feasible_experiments:
        print("  警告：没有找到可行解！")
        return None
    
    # 找到最接近目标深度的gamma值
    target_gammas = []
    for exp in feasible_experiments:
        depth_diff = abs(exp['optimal_depth'] - target_depth)
        if depth_diff <= 1:  # 允许±1的误差
            target_gammas.append(exp)
    
    if target_gammas:
        # 在目标深度范围内，选择适应度最高的
        best_in_target = max(target_gammas, key=lambda x: x['optimal_fitness'])
        print(f"\n✓ 找到 {len(target_gammas)} 个gamma值使深度在目标范围")
        
        # 显示目标范围内的gamma值
        print(f"\n目标深度范围内的Gamma值:")
        print(f"{'Gamma':>10} | {'Depth':>6} | {'Fitness':>10} | {'F1':>6} | {'Rules':>8}")
        print("-" * 60)
        for exp in sorted(target_gammas, key=lambda x: x['gamma']):
            print(f"{exp['gamma']:>10.3f} | {exp['optimal_depth']:>6} | {exp['optimal_fitness']:>10.4f} | {exp['optimal_f1']*100:>5.2f}% | {exp['optimal_rule_count']:>8}")
        
        print(f"\n推荐最优Gamma值: {best_in_target['gamma']:.3f}")
        print(f"  最优深度: {best_in_target['optimal_depth']}")
        print(f"  最优适应度: {best_in_target['optimal_fitness']:.6f}")
        print(f"  最优F1分数: {best_in_target['optimal_f1']*100:.2f}%")
        print(f"  最优规则数: {best_in_target['optimal_rule_count']}")
        
        return best_in_target
    else:
        # 没有在目标范围内的解，选择最接近的
        closest = min(feasible_experiments, key=lambda x: abs(x['optimal_depth'] - target_depth))
        print(f"\n✗ 没有gamma值使深度在目标范围")
        print(f"  最接近的Gamma值: {closest['gamma']:.3f}")
        print(f"  最接近深度: {closest['optimal_depth']}")
        print(f"  深度误差: {abs(closest['optimal_depth'] - target_depth)}")
        
        return closest


def generate_report(experiment_results, best_gamma_result, target_depth=5, output_file=None):
    """
    生成优化报告
    
    参数:
        experiment_results: 所有gamma实验的结果列表
        best_gamma_result: 最优gamma实验结果
        target_depth: 目标深度
        output_file: 输出文件路径
    """
    if output_file is None:
        output_file = f"gamma_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # 统计信息
    total_time = sum(e['elapsed_time'] for e in experiment_results)
    total_tests = sum(len(e.get('all_results', [])) for e in experiment_results)
    
    report = {
        'config_name': 'Config1',
        'target_depth': target_depth,
        'experiment_params': {
            'depth_range': [min(DEPTH_RANGE), max(DEPTH_RANGE)],
            'num_cores': 92,
            'gamma_values': [e['gamma'] for e in experiment_results]
        },
        'final_result': {
            'best_gamma': best_gamma_result['gamma'] if best_gamma_result else None,
            'best_depth': best_gamma_result['optimal_depth'] if best_gamma_result else None,
            'best_fitness': best_gamma_result['optimal_fitness'] if best_gamma_result else None,
            'depth_error': abs(best_gamma_result['optimal_depth'] - target_depth) if best_gamma_result and best_gamma_result['optimal_depth'] else None
        },
        'statistics': {
            'total_experiments': len(experiment_results),
            'feasible_experiments': sum(1 for e in experiment_results if e['optimal_depth'] is not None),
            'total_tests': total_tests,
            'total_time': total_time,
            'avg_time_per_experiment': total_time / len(experiment_results) if experiment_results else 0
        },
        'experiments': experiment_results
    }
    
    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print(f"\n{'='*80}")
    print("优化报告摘要")
    print(f"{'='*80}")
    print(f"配置名称: {report['config_name']}")
    print(f"目标深度: {report['target_depth']}")
    print(f"最优gamma: {report['final_result']['best_gamma']:.4f}")
    print(f"实际深度: {report['final_result']['best_depth']}")
    print(f"深度误差: {report['final_result']['depth_error']:.4f}")
    print(f"最优适应度: {report['final_result']['best_fitness']:.6f}")
    print(f"\n统计信息:")
    print(f"  总实验数: {report['statistics']['total_experiments']}")
    print(f"  可行实验数: {report['statistics']['feasible_experiments']}")
    print(f"  总测试数: {report['statistics']['total_tests']}")
    print(f"  总耗时: {report['statistics']['total_time']:.2f}秒")
    print(f"  平均耗时: {report['statistics']['avg_time_per_experiment']:.2f}秒")
    print(f"\n报告已保存: {output_file}")
    
    return report


def binary_search_gamma(config, depth_range, target_depth, gamma_min, gamma_max, 
                      num_cores=92, max_iterations=10, tolerance=0.001):
    """
    使用密集采样探索gamma参数，分析深度变化规律
    
    参数:
        config: 配置对象
        depth_range: 深度范围
        target_depth: 目标深度
        gamma_min: gamma最小值
        gamma_max: gamma最大值
        num_cores: 并行核心数
        max_iterations: 最大迭代次数
        tolerance: 收敛容差
    
    返回:
        最优gamma值及搜索历史
    """
    print(f"\n{'='*80}")
    print("Gamma参数密集采样分析")
    print(f"目标深度: {target_depth}")
    print(f"搜索范围: [{gamma_min}, {gamma_max}]")
    print(f"采样步长: 0.001")
    print(f"{'='*80}")
    
    search_history = []
    
    # 密集采样：从gamma_min到gamma_max，步长0.001
    gamma_values = []
    g = gamma_min
    while g <= gamma_max:
        gamma_values.append(round(g, 6))
        g += 0.001
    
    print(f"\n采样点数: {len(gamma_values)}")
    print(f"Gamma值: {gamma_values}")
    
    # 逐个测试每个gamma值
    for i, gamma in enumerate(gamma_values):
        print(f"\n{'='*80}")
        print(f"测试 {i+1}/{len(gamma_values)}: Gamma={gamma:.6f}")
        print(f"{'='*80}")
        
        result = run_gamma_experiment(gamma, config, depth_range, num_cores)
        search_history.append(result)
        
        # 打印所有深度的适应度，用于分析
        if 'all_results' in result:
            print(f"\n所有深度结果:")
            print(f"{'Depth':>6} | {'Fitness':>10} | {'F1':>6} | {'Rules':>8} | {'TCAM%':>7}")
            print("-" * 55)
            for r in sorted(result['all_results'], key=lambda x: x['depth']):
                if r.get('feasible', False):
                    print(f"{r['depth']:>6} | {r['fitness']:>10.6f} | {r['f1']*100:>5.2f}% | {r['rule_count']:>8} | {r['tcam_pct']:>7.2f}")
    
    # 分析结果
    print(f"\n{'='*80}")
    print("深度变化分析")
    print(f"{'='*80}")
    print(f"\n{'Gamma':>10} | {'Optimal Depth':>13} | {'Fitness':>10} | {'F1':>6} | {'Rules':>8}")
    print("-" * 70)
    
    for result in search_history:
        if result['optimal_depth'] is not None:
            print(f"{result['gamma']:>10.6f} | {result['optimal_depth']:>13} | {result['optimal_fitness']:>10.6f} | {result['optimal_f1']*100:>5.2f}% | {result['optimal_rule_count']:>8}")
    
    # 统计各深度出现的次数
    depth_counts = {}
    for result in search_history:
        depth = result['optimal_depth']
        if depth is not None:
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
    
    print(f"\n深度分布:")
    for depth in sorted(depth_counts.keys()):
        print(f"  Depth={depth}: {depth_counts[depth]} 次")
    
    # 找到最接近目标深度的结果
    closest = min(search_history, 
                 key=lambda x: abs(x['optimal_depth'] - target_depth) if x['optimal_depth'] else float('inf'))
    
    # 检查是否存在目标深度
    has_target_depth = any(r['optimal_depth'] == target_depth for r in search_history)
    
    if has_target_depth:
        print(f"\n✓ 找到目标深度{target_depth}的gamma值！")
        target_results = [r for r in search_history if r['optimal_depth'] == target_depth]
        print(f"  共{len(target_results)}个gamma值使深度={target_depth}")
        for r in target_results:
            print(f"    Gamma={r['gamma']:.6f}, Fitness={r['optimal_fitness']:.6f}")
    else:
        print(f"\n✗ 未找到目标深度{target_depth}的gamma值")
        print(f"  最接近的深度: {closest['optimal_depth']}")
        print(f"  最接近的gamma: {closest['gamma']:.6f}")
        print(f"  深度误差: {abs(closest['optimal_depth'] - target_depth)}")
    
    return closest, search_history


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Config1 Gamma参数优化探索脚本（修正版）')
    parser.add_argument('--mode', type=str, default='grid', choices=['grid', 'binary'],
                       help='搜索模式：grid=网格搜索，binary=二分查找（默认：grid）')
    parser.add_argument('--gamma-values', type=str, default='1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0',
                       help='gamma值列表，逗号分隔（网格搜索模式，默认：1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0）')
    parser.add_argument('--gamma-min', type=float, default=1.4, help='gamma最小值（二分查找模式，默认：1.4）')
    parser.add_argument('--gamma-max', type=float, default=1.5, help='gamma最大值（二分查找模式，默认：1.5）')
    parser.add_argument('--target-depth', type=int, default=5, help='目标深度（默认：5）')
    parser.add_argument('--num-cores', type=int, default=92, help='并行核心数（默认：92）')
    parser.add_argument('--max-iterations', type=int, default=10, help='最大迭代次数（二分查找模式，默认：10）')
    parser.add_argument('--tolerance', type=float, default=0.001, help='收敛容差（二分查找模式，默认：0.001）')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("Config1 Gamma参数优化探索脚本（修正版）")
    print(f"搜索模式: {args.mode}")
    print(f"目标深度: {args.target_depth}")
    print(f"并行核心数: {args.num_cores}")
    print(f"深度范围: {min(DEPTH_RANGE)}-{max(DEPTH_RANGE)} ({len(DEPTH_RANGE)}个深度)")
    print(f"{'='*80}")
    
    # Config1基础配置
    base_config = {
        'name': 'Config1',
        'tcam_pct_max': 5.0,
        'sram_pct': 1.0,
        'stages': 1,
        'alpha': 2.0,
        'beta': 0.05,
        'gamma': 1.5,  # 会被覆盖
        'target_depth': 5,  # 目标深度
        'delta': 0.5  # 目标深度奖励权重
    }
    
    start_time = time.time()
    
    if args.mode == 'binary':
        # 二分查找模式
        print(f"\n二分查找参数:")
        print(f"  搜索范围: [{args.gamma_min}, {args.gamma_max}]")
        print(f"  最大迭代: {args.max_iterations}")
        print(f"  收敛容差: {args.tolerance}")
        
        best_gamma_result, search_history = binary_search_gamma(
            base_config, DEPTH_RANGE, args.target_depth,
            args.gamma_min, args.gamma_max, args.num_cores,
            args.max_iterations, args.tolerance
        )
        
        # 生成报告
        if args.output is None:
            output_file = f"gamma_binary_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            output_file = args.output
        
        report = {
            'config_name': 'Config1',
            'target_depth': args.target_depth,
            'search_mode': 'binary',
            'search_params': {
                'gamma_range': [args.gamma_min, args.gamma_max],
                'num_cores': args.num_cores,
                'max_iterations': args.max_iterations,
                'tolerance': args.tolerance
            },
            'final_result': {
                'best_gamma': best_gamma_result['gamma'],
                'best_depth': best_gamma_result['optimal_depth'],
                'depth_error': abs(best_gamma_result['optimal_depth'] - args.target_depth) if best_gamma_result['optimal_depth'] else None,
                'best_fitness': best_gamma_result['optimal_fitness'],
                'best_f1': best_gamma_result['optimal_f1'],
                'best_rule_count': best_gamma_result['optimal_rule_count']
            },
            'statistics': {
                'total_tests': len(search_history),
                'total_time': time.time() - start_time,
                'avg_time_per_test': (time.time() - start_time) / len(search_history) if search_history else 0
            },
            'search_history': search_history
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 打印摘要
        print(f"\n{'='*80}")
        print("优化报告摘要")
        print(f"{'='*80}")
        print(f"配置名称: {report['config_name']}")
        print(f"目标深度: {report['target_depth']}")
        print(f"最优gamma: {report['final_result']['best_gamma']:.6f}")
        print(f"实际深度: {report['final_result']['best_depth']}")
        print(f"深度误差: {report['final_result']['depth_error']:.6f}")
        print(f"最优适应度: {report['final_result']['best_fitness']:.6f}")
        print(f"\n统计信息:")
        print(f"  总测试数: {report['statistics']['total_tests']}")
        print(f"  总耗时: {report['statistics']['total_time']:.2f}秒")
        print(f"  平均耗时: {report['statistics']['avg_time_per_test']:.2f}秒")
        print(f"\n报告已保存: {output_file}")
        
    else:
        # 网格搜索模式
        gamma_values = [float(g.strip()) for g in args.gamma_values.split(',')]
        
        print(f"\n网格搜索参数:")
        print(f"  Gamma值列表: {gamma_values}")
        
        # 运行所有gamma实验
        experiment_results = []
        
        for gamma_value in gamma_values:
            result = run_gamma_experiment(gamma_value, base_config, DEPTH_RANGE, args.num_cores)
            experiment_results.append(result)
        
        # 分析结果
        best_gamma_result = analyze_gamma_experiments(experiment_results, args.target_depth)
        
        # 生成报告
        report = generate_report(experiment_results, best_gamma_result, args.target_depth, args.output)
    
    total_elapsed_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"优化完成！总耗时: {total_elapsed_time:.2f}秒")
    print(f"{'='*80}")


if __name__ == '__main__':
    freeze_support()
    main()
