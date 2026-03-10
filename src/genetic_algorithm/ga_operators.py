# -*- coding: utf-8 -*-
"""
遗传算子：选择、交叉、变异
包含硬件约束检查
"""

import random
try:
    from .ga_config import ALL_FEATURES, MATCH_TYPE_OPTIONS, MAX_LPM_COUNT
except ImportError:
    from ga_config import ALL_FEATURES, MATCH_TYPE_OPTIONS, MAX_LPM_COUNT


def check_lpm_constraint(chromosome):
    """
    检查 LPM 约束：整表最多允许 MAX_LPM_COUNT 个 LPM 字段
    返回: True 如果满足约束
    """
    lpm_count = 0
    for i, feat in enumerate(ALL_FEATURES):
        options = MATCH_TYPE_OPTIONS[feat]
        gene = chromosome[i] % len(options)
        match_type = options[gene]
        if match_type == 3:  # LPM
            lpm_count += 1
    return lpm_count <= MAX_LPM_COUNT


def repair_lpm_constraint(chromosome):
    """
    修复 LPM 约束：如果 LPM 数量超过限制，随机将部分 LPM 改为其他类型
    """
    result = chromosome.copy()
    
    # 找出所有 LPM 位置
    lpm_positions = []
    for i, feat in enumerate(ALL_FEATURES):
        options = MATCH_TYPE_OPTIONS[feat]
        gene = result[i] % len(options)
        match_type = options[gene]
        if match_type == 3:  # LPM
            lpm_positions.append(i)
    
    # 如果 LPM 数量超过限制，随机保留 MAX_LPM_COUNT 个
    while len(lpm_positions) > MAX_LPM_COUNT:
        # 随机选择一个 LPM 位置进行修改
        pos_to_fix = random.choice(lpm_positions)
        lpm_positions.remove(pos_to_fix)
        
        # 将该位置改为非 LPM 的选项
        feat = ALL_FEATURES[pos_to_fix]
        options = MATCH_TYPE_OPTIONS[feat]
        non_lpm_options = [j for j, opt in enumerate(options) if opt != 3]
        if non_lpm_options:
            result[pos_to_fix] = random.choice(non_lpm_options)
    
    return result


def initialize_population(size):
    """初始化种群"""
    population = []
    for _ in range(size):
        chromosome = []
        for feat in ALL_FEATURES:
            max_val = len(MATCH_TYPE_OPTIONS[feat]) - 1
            chromosome.append(random.randint(0, max_val))
        
        # 修复 LPM 约束
        chromosome = repair_lpm_constraint(chromosome)
        population.append(chromosome)
    return population


def tournament_selection(population, fitness_values, k=3):
    """锦标赛选择"""
    selected_indices = random.sample(range(len(population)), min(k, len(population)))
    best_idx = max(selected_indices, key=lambda i: fitness_values[i])
    return population[best_idx].copy()


def crossover(parent1, parent2, rate=0.8):
    """单点交叉"""
    if random.random() > rate or len(parent1) <= 1:
        return parent1.copy(), parent2.copy()
    
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(chromosome, rate=0.2):
    """随机变异"""
    result = chromosome.copy()
    for i, feat in enumerate(ALL_FEATURES):
        if random.random() < rate:
            max_val = len(MATCH_TYPE_OPTIONS[feat]) - 1
            result[i] = random.randint(0, max_val)
    return result


def create_next_generation(population, fitness_values, params):
    """创建下一代种群"""
    pop_size = params['population_size']
    elite_count = params['elite_count']
    crossover_rate = params['crossover_rate']
    mutation_rate = params['mutation_rate']
    tournament_size = params['tournament_size']
    
    # 精英保留
    elite_indices = sorted(
        range(len(population)),
        key=lambda i: fitness_values[i],
        reverse=True
    )[:elite_count]
    elites = [population[i].copy() for i in elite_indices]
    
    # 生成新种群
    new_population = elites.copy()
    
    while len(new_population) < pop_size:
        # 选择父代
        p1 = tournament_selection(population, fitness_values, tournament_size)
        p2 = tournament_selection(population, fitness_values, tournament_size)
        
        # 交叉
        c1, c2 = crossover(p1, p2, crossover_rate)
        
        # 变异
        c1 = mutate(c1, mutation_rate)
        c2 = mutate(c2, mutation_rate)
        
        # 修复 LPM 约束
        c1 = repair_lpm_constraint(c1)
        c2 = repair_lpm_constraint(c2)
        
        new_population.append(c1)
        if len(new_population) < pop_size:
            new_population.append(c2)
    
    return new_population[:pop_size]
