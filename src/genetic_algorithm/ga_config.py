# -*- coding: utf-8 -*-
"""
遗传算法配置参数 (Server Edition)
适配 AMD EPYC 9274F (24-Core, 48-Thread)
路径配置已更新为相对路径

Fitness = α × F1 + β × TCAM_Util - γ × Latency_Factor
"""

import os
import math

# ======================== 路径配置 ========================
# 动态获取项目根目录 
# 假设当前文件在 Dryad/src/genetic_algorithm/ga_config.py
# Dryad/src/genetic_algorithm -> Dryad/src -> Dryad (Project Root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 模型路径: Dryad/models (注意这里是新的相对路径位置)
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# 如果还有其他JSON模型，也可以放在 Dryad/json_models
JSON_MODELS_DIR = os.path.join(BASE_DIR, 'json_models')

# ======================== 硬件资源配置 ========================
CONFIGS = [
    {
        'name': 'Config1',
        'tcam_pct_max': 5.0,     # Screenshot: 5%
        'sram_pct': 1.0,         # Screenshot: 1%
        'stages': 1,             # Screenshot: 1
        'alpha': 2.0,            # 精度权重
        'beta': 0.05,
        'gamma': 1.5,            # 推算最优值: 基于历史数据插值，使深度从4提升至5
        'description': '极低资源 (边缘/IoT)'
    },
    {
        'name': 'Config2',
        'tcam_pct_max': 30.0,    # Screenshot: 30%
        'sram_pct': 4.0,         # Screenshot: 4%
        'stages': 4,             # Screenshot: 4
        'alpha': 4.2,            # 微调: 提高至4.2，增强对F1指标的激励
        'beta': 0.1,
        'gamma': 0.8,            # 调整: 设置为0.8
        'description': '中低资源 (接入层)'
    },
    {
        'name': 'Config3',
        'tcam_pct_max': 55.0,    # Screenshot: 55%
        'sram_pct': 7.0,         # Screenshot: 7%
        'stages': 8,             # Screenshot: 8
        'alpha': 5.0,            
        'beta': 0.4,             # 提高资源奖励 -> 鼓励利用TCAM
        'gamma': 0.9,            # 精准微调: 回调至0.9，压制深度至10-12范围
        'description': '中高资源 (汇聚层)'
    },
    {
        'name': 'Config4',
        'tcam_pct_max': 80.0,    # Screenshot: 80%
        'sram_pct': 10.0,        # Screenshot: 10%
        'stages': 12,            # Screenshot: 12
        'alpha': 15.0,
        'beta': 0.5,             # 降低资源奖励 -> 避免过度拟合TCAM上限
        'gamma': 0.05,           # 微调: 设置为0.05，极低延迟惩罚
        'description': '高资源 (核心层)'
    }
]

# 延迟因子计算的参考最大规则数
MAX_RULES_REF = 10000

# ======================== 深度与精度映射 ========================
DEPTH_F1_MAP = {
    4: 0.7352,
    5: 0.7763,
    6: 0.8288,
    7: 0.8430,
    8: 0.8528,
    9: 0.8661,
    10: 0.8686,
    11: 0.8745,
    12: 0.8816,
    13: 0.8820,
    14: 0.8822,
    15: 0.8824,
    16: 0.8828,
    17: 0.8832,
    18: 0.8836,
    19: 0.8840,
    20: 0.8843,
    21: 0.8844,
    22: 0.8845,
    23: 0.8845,
    24: 0.8846,
    25: 0.8846,
    26: 0.8847,
    27: 0.8848,
    28: 0.8849,
    29: 0.8849,
    30: 0.8850,
    31: 0.8850,
    32: 0.8851,
    33: 0.8851,
    34: 0.8851,
    35: 0.8851,
    36: 0.8852,
    37: 0.8852,
    38: 0.8852,
}

# ======================== 探索深度范围 ========================
MIN_DEPTH = 4
MAX_DEPTH = 38
DEPTH_RANGE = list(range(MIN_DEPTH, MAX_DEPTH + 1))

# ======================== 特征配置 ========================
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

# 特征顺序（影响规则生成）
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

ALL_FEATURES = FEATURE_ORDER
ONE_BIT_FEATURES = ['IPV4 Flags (DF)', 'TCP flags (Reset)', 'TCP flags (Syn)']

# 匹配类型选项: 0: exact, 1: ternary, 2: range, 3: lpm
MATCH_TYPE_OPTIONS = {
    'Total length': [1, 2, 3],
    'Protocol': [0, 1, 3],
    'Time to live': [1, 2, 3],
    'Src Port': [1, 2, 3],
    'Dst Port': [1, 2, 3],
    'IPV4 Flags (DF)': [0, 1, 3],
    'TCP flags (Reset)': [0, 1, 3],
    'TCP flags (Syn)': [0, 1, 3],
}

MATCH_TYPE_NAMES = {
    0: 'exact',
    1: 'ternary',
    2: 'range',
    3: 'lpm'
}

# ======================== 遗传算法参数 (Server Config) ========================
# 适配 AMD EPYC 9274F (24-Core, 96-Thread)
# 注意：num_workers 或 n_jobs 建议设置为 90-92，预留少量核心给系统
GA_PARAMS = {
    'population_size': 100,    # 增大种群规模 (原50)
    'generations': 100,        # 增加迭代次数 (原80)
    'crossover_rate': 0.85,
    'mutation_rate': 0.15,
    'tournament_size': 3,
    'elite_count': 5,
    'n_jobs': 92               # 并行进程数，接近物理线程数（96核系统）
}

# ======================== 硬件约束 ========================
MAX_LPM_COUNT = 1


# ======================== 适应度计算函数 ========================
def calculate_weighted_fitness(f1, tcam_pct, rule_count, config):
    """
    加权多目标适应度函数
    
    Fitness = α × F1 + β × TCAM_Util - γ × Latency_Factor
    """
    alpha = config.get('alpha', 1.0)
    beta = config.get('beta', 0.1)      # 资源利用率奖励权重
    gamma = config.get('gamma', 1.0)
    tcam_max = config.get('tcam_pct_max', 100.0)
    
    # TCAM利用率（0-1范围）
    tcam_util = tcam_pct / tcam_max if tcam_max > 0 else 0
    
    # 延迟因子 (对数归一化)
    if rule_count > 1:
        latency_factor = math.log(rule_count) / math.log(MAX_RULES_REF)
    else:
        latency_factor = 0
    
    # 加权适应度：精度 + 资源利用率奖励 - 延迟惩罚
    fitness = alpha * f1 + beta * tcam_util - gamma * latency_factor
    
    return fitness, {
        'f1': f1,
        'tcam_util': tcam_util,
        'latency_factor': latency_factor,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma
    }
