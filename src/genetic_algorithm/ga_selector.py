# -*- coding: utf-8 -*-
"""
最优深度选择器 - 基于边际收益递减的选择算法
"""

from .ga_config import SELECTION_PARAMS


def select_optimal_depth(results, config):
    """
    基于边际收益递减的最优深度选择
    
    参数:
        results: List[dict] - 所有深度的探索结果
        config: dict - 资源配置
    
    返回:
        dict - 最优解
    """
    params = SELECTION_PARAMS
    
    # 过滤可行解
    valid = [r for r in results 
             if r.get('feasible', False) and r.get('f1', 0) >= params['min_f1']]
    
    if not valid:
        # 没有可行解，返回最接近可行的
        all_results = [r for r in results if 'fitness' in r]
        if all_results:
            return max(all_results, key=lambda r: r.get('fitness', float('-inf')))
        return None
    
    # 按深度排序
    valid.sort(key=lambda x: x['depth'])
    
    # 边际分析选择
    selected = valid[0]
    
    for i in range(1, len(valid)):
        curr = valid[i]
        prev = valid[i-1]
        
        f1_gain = curr.get('f1', 0) - prev.get('f1', 0)
        util_curr = curr.get('utilization', 0)
        util_prev = prev.get('utilization', 0)
        resource_increase = util_curr - util_prev
        
        # 条件1：边际增益必须超过阈值
        if f1_gain < params['marginal_threshold']:
            break
        
        # 条件2：资源调整后的边际增益
        if resource_increase > 0:
            adjusted_gain = f1_gain / resource_increase
            if adjusted_gain < params['adjusted_threshold']:
                break
        
        # 条件3：利用率在合理范围
        if params['utilization_min'] <= util_curr <= params['utilization_max']:
            selected = curr
        elif util_curr > params['utilization_max']:
            break
    
    # 最终优化：如果当前选择的利用率太低，尝试找更高深度
    if selected.get('utilization', 0) < params['utilization_min']:
        better = [r for r in valid 
                  if params['utilization_min'] <= r.get('utilization', 0) <= params['utilization_max']]
        if better:
            selected = max(better, key=lambda x: x.get('f1', 0))
    
    return selected


def format_result_summary(result):
    """格式化单个结果的摘要"""
    if result is None:
        return "无可行解"
    
    depth = result.get('depth', '?')
    f1 = result.get('f1', 0)
    rule_count = result.get('rule_count', 0)
    utilization = result.get('utilization', 0)
    match_config = result.get('match_config_readable', {})
    
    lines = [
        f"  最优深度: {depth}",
        f"  Macro F1: {f1:.4f}",
        f"  规则数: {rule_count}",
        f"  资源利用率: {utilization:.1%}",
        f"  匹配配置: {match_config}"
    ]
    
    return '\n'.join(lines)
