# -*- coding:utf-8 -*-
"""
分层深度优先遗传算法（Hierarchical Depth-First GA）
策略：从最高深度开始，逐层降低，每层优化匹配方式
"""

import os
import sys
import time
import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimization import *
from sklearn.metrics import classification_report


def calculate_resource_score(individual):
    """
    计算资源消耗综合评分（越低越好）
    
    优先级：TCAM > SRAM > Stages > 规则数
    
    评分公式：
    score = TCAM_weight * TCAM% + SRAM_weight * SRAM% + Stages_weight * Stages + Rules_weight * (Rules/1000)
    
    权重设计基于硬件成本和稀缺性：
    - TCAM：权重100（最昂贵，最稀缺）
    - SRAM：权重10（次要资源）
    - Stages：权重1（影响延迟，但资源占用更重要）
    - 规则数：权重0.1（作为tie-breaker）
    """
    tcam_weight = 100.0
    sram_weight = 10.0
    stages_weight = 1.0
    rules_weight = 0.1
    
    score = (
        tcam_weight * individual.pred_tcam +
        sram_weight * individual.pred_sram +
        stages_weight * individual.pred_stages +
        rules_weight * (individual.p4_rule_size / 1000.0)
    )
    
    return score


def compare_solutions(sol1, sol2):
    """
    比较两个解的优劣（用于排序）
    
    返回：
    - 负数：sol1更优
    - 正数：sol2更优
    - 0：相等
    
    比较规则：
    1. F1更高者优先
    2. F1相同时，资源评分更低者优先
    """
    # 首先比较F1（容差1e-6）
    f1_diff = sol2.fitness - sol1.fitness
    if abs(f1_diff) > 1e-6:
        return f1_diff
    
    # F1相同，比较资源评分
    score1 = calculate_resource_score(sol1)
    score2 = calculate_resource_score(sol2)
    
    return score1 - score2  # 评分越低越好


def sort_solutions_by_priority(solutions):
    """
    按优先级排序解：F1高优先，F1相同时资源消耗低优先
    """
    from functools import cmp_to_key
    return sorted(solutions, key=cmp_to_key(compare_solutions))


def soft_prune(json_model):
    """
    软剪枝：删除所有子节点属于同一类别的非叶节点
    """
    classNameStack = []
    jsonNode = json_model
    jsonNodeStack = []

    while jsonNodeStack or jsonNode:
        while jsonNode:  # 一直往左走，走到最左的节点
            jsonNodeStack.append(jsonNode)
            if "children" in jsonNode:
                jsonNode = jsonNode["children"][0]
            else:
                jsonNode = None

        # 访问当前节点
        currentNode = jsonNodeStack.pop()  # 转到最后一个节点
        if "children" in currentNode and len(currentNode["children"]) > 0:  # 如果该节点有子节点
            currentNode["leafcount"][0] = currentNode["children"][0]["leafcount"][0] + \
                                          currentNode["children"][0]["leafcount"][1]
            currentNode["leafcount"][1] = currentNode["children"][1]["leafcount"][0] + \
                                          currentNode["children"][1]["leafcount"][1]

            classname = np.argmax(currentNode['value'])  # 得到当前节点所属的类别

            # 判断该节点是否可以被软剪枝
            flag = 1
            count = currentNode["leafcount"][0] + currentNode["leafcount"][1]  # 得到当前节点左右子节点叶子数之和
            for childClassName in classNameStack[-count:]:
                if classname != childClassName:
                    flag = 0
                    break
            if flag == 1:
                currentNode["tobedel"] = 1
                del currentNode["children"]
        else:  # 如果该节点没有子节点
            classNameStack.append(np.argmax(currentNode['value']))  # push叶节点所属的类别

        # turn to current node's brother right node
        if jsonNodeStack and jsonNodeStack[-1]["children"][0] is currentNode:
            jsonNode = jsonNodeStack[-1]["children"][1]
        else:
            jsonNode = None

    return json_model

def hierarchical_depth_first_ga(
    original_tree, feature_list, class_names, X_test, y_test,
    max_depth=12, min_depth=6, config=None, 
    generations_per_depth=20, min_feasible_solutions=5
):
    """
    分层深度优先遗传算法
    
    策略:
    1. 从max_depth开始，固定深度，优化匹配方式
    2. 如果找到足够数量的可行解，继续当前深度
    3. 如果N代后无可行解，降低深度
    4. 重复直到min_depth或找到足够解
    
    Args:
        max_depth: 起始深度（默认12）
        min_depth: 最小深度（默认6）
        generations_per_depth: 每个深度探索的代数
        min_feasible_solutions: 每个深度的目标可行解数量
    """
    if config is None:
        config = GAConfig(
            population_size=150,  # 100 → 150，增加种群多样性
            generations=generations_per_depth,
            crossover_rate=0.8,
            mutation_rate=0.3,
            tournament_size=3,
            elite_size=15,  # 10 → 15
            # ISCX数据集的硬件约束
            limit_tcam=4.5,   # 4.5%
            limit_sram=0.9,   # 0.9%
            limit_stages=3.0  # 3.0
        )
    
    print("\n" + "=" * 80)
    print("分层深度优先遗传算法")
    print("=" * 80)
    print(f"深度范围: {max_depth} → {min_depth}")
    print(f"每层代数: {generations_per_depth}")
    print(f"目标可行解数: {min_feasible_solutions}")
    print("=" * 80)
    
    feature_match_options = [
        MatchType.RANGE, MatchType.PREFIX, MatchType.TERNARY, MatchType.EXACT
    ]
    
    rule_converter = P4RuleConverter()
    resource_model = ResourcePredictionModel()
    
    all_feasible_solutions = []
    global_best = None
    depth_results = {}
    
    # 从高到低遍历深度
    for current_depth in range(max_depth, min_depth - 1, -1):
        print(f"\n{'='*80}")
        print(f"🔍 探索深度 {current_depth}")
        print(f"{'='*80}")
        
        # 初始化种群：混合策略（50%最优配置 + 50%随机）
        print(f"初始化种群（深度固定为{current_depth}）...")
        print(f"  策略：50%使用最优配置（规则数最少），50%随机初始化")
        population = []
        
        # 最优配置（规则数≈叶子节点数）
        optimal_config = [
            MatchType.RANGE,   # Total length
            MatchType.TERNARY, # Protocol
            MatchType.TERNARY, # IPV4 Flags (DF)
            MatchType.RANGE,   # Time to live
            MatchType.RANGE,   # Src Port
            MatchType.RANGE,   # Dst Port
            MatchType.TERNARY, # TCP flags (Reset)
            MatchType.TERNARY  # TCP flags (Syn)
        ]
        
        # 使用加权随机选择匹配方式（降低Exact权重，避免规则爆炸）
        # 而不是强制禁止，保留遗传算法的探索能力
        match_weights = {
            MatchType.RANGE: 3,    # 权重3
            MatchType.TERNARY: 3,  # 权重3
            MatchType.PREFIX: 2,   # 权重2
            MatchType.EXACT: 1     # 权重1（降低但不禁止）
        }
        weighted_options = []
        for match_type, weight in match_weights.items():
            weighted_options.extend([match_type] * weight)
        
        for idx in range(config.population_size):
            individual = Individual()
            individual.tree_depth = current_depth  # 固定深度
            
            # 前50%：基于最优配置 + 小变异
            if idx < config.population_size // 2:
                # 复制最优配置
                individual.feature_match_types = optimal_config.copy()
                
                # 随机变异2-3个位置（保持大部分最优结构）
                num_mutations = random.randint(2, 3)
                mutation_indices = random.sample(range(len(feature_list)), num_mutations)
                for mut_idx in mutation_indices:
                    individual.feature_match_types[mut_idx] = random.choice(weighted_options)
                    
            # 后50%：完全随机初始化（保持多样性）
            else:
                individual.feature_match_types = [
                    random.choice(weighted_options)
                    for _ in range(len(feature_list))
                ]
            
            # 强制约束：16-bit字段禁用Exact（避免规则爆炸）
            # Total length(16bit), Src Port(16bit), Dst Port(16bit)
            high_cardinality_features = {
                'Total length': 0,
                'Src Port': 4,
                'Dst Port': 5
            }
            for feat_name, idx_feat in high_cardinality_features.items():
                if idx_feat < len(individual.feature_match_types):
                    if individual.feature_match_types[idx_feat] == MatchType.EXACT:
                        # 替换为Range
                        individual.feature_match_types[idx_feat] = MatchType.RANGE
            
            # 生成树结构（硬剪枝+软剪枝）
            individual.tree_structure = hard_prune(
                copy.deepcopy(original_tree), 0, current_depth
            )
            # 应用软剪枝
            individual.tree_structure = soft_prune(individual.tree_structure)
            individual.tree_structure, individual.rule_statistics = convert_tree_to_match_types(
                individual.tree_structure, individual.feature_match_types, feature_list
            )
            
            population.append(individual)
        
        best_at_depth = None
        feasible_at_depth = []
        no_improvement_count = 0
        
        # 在当前深度进行进化
        for gen in range(generations_per_depth):
            # 评估适应度
            for ind in population:
                if ind.fitness == 0.0:
                    calculate_fitness(ind, X_test, y_test, feature_list,
                                    class_names, config, rule_converter, resource_model)
            
            # 记录可行解（去重）
            for ind in population:
                if ind.is_feasible and ind.fitness > 0:
                    is_dup = False
                    for fs in feasible_at_depth:
                        if fs.feature_match_types == ind.feature_match_types:
                            is_dup = True
                            break
                    if not is_dup:
                        feasible_at_depth.append(copy.deepcopy(ind))
            
            # 选择最佳（使用资源评分）
            population.sort(key=lambda x: x.fitness, reverse=True)
            current_best = population[0]
            
            # 如果有多个F1相同的解，选择资源消耗最低的
            top_fitness = current_best.fitness
            candidates = [ind for ind in population if abs(ind.fitness - top_fitness) < 1e-6]
            if len(candidates) > 1:
                current_best = min(candidates, key=calculate_resource_score)
            
            if best_at_depth is None:
                best_at_depth = copy.deepcopy(current_best)
                no_improvement_count = 0
            else:
                # 比较：F1提升 或 F1相同但资源更优
                if (current_best.fitness > best_at_depth.fitness + 1e-6 or
                    (abs(current_best.fitness - best_at_depth.fitness) < 1e-6 and
                     calculate_resource_score(current_best) < calculate_resource_score(best_at_depth))):
                    best_at_depth = copy.deepcopy(current_best)
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            
            # 打印进度（增强诊断信息）
            if (gen + 1) % 5 == 0 or gen == 0:
                # 统计不可行的原因
                infeasible_count = sum(1 for ind in population if not ind.is_feasible)
                if infeasible_count > 0:
                    # 分析前20个不可行个体的超限原因
                    tcam_exceed = 0
                    sram_exceed = 0
                    stages_exceed = 0
                    
                    # 记录前3个不可行个体的实际预测值
                    sample_predictions = []
                    
                    for ind in population[:20]:
                        if not ind.is_feasible:
                            if ind.pred_tcam > config.limit_tcam:
                                tcam_exceed += 1
                            if ind.pred_sram > config.limit_sram:
                                sram_exceed += 1
                            if ind.pred_stages > config.limit_stages:
                                stages_exceed += 1
                            
                            # 记录前3个的详细预测值
                            if len(sample_predictions) < 3:
                                sample_predictions.append({
                                    'rules': ind.p4_rule_size,
                                    'tcam': ind.pred_tcam,
                                    'sram': ind.pred_sram,
                                    'stages': ind.pred_stages
                                })
                    
                    exceed_info = f"超限个体数(前20个): TCAM={tcam_exceed}个, SRAM={sram_exceed}个, Stages={stages_exceed}个"
                    
                    # 显示实际预测值样本（改进格式）
                    if sample_predictions:
                        sample_str = " | 样本: "
                        for i, sp in enumerate(sample_predictions[:2]):  # 只显示前2个
                            sample_str += f"#{i+1}(规则{sp['rules']},TCAM{sp['tcam']:.1f}%,Stage{sp['stages']:.0f}) "
                        exceed_info += sample_str
                else:
                    exceed_info = ""
                
                print(f"  第{gen+1:2d}代: F1={current_best.fitness:.4f}, "
                      f"可行解={len(feasible_at_depth)}, "
                      f"规则数={current_best.p4_rule_size}, "
                      f"可行={current_best.is_feasible} {exceed_info}")
            
            # 早停：找到可行解后，充分探索当前深度
            if len(feasible_at_depth) >= 1:  # 只要有1个可行解
                if no_improvement_count >= 15:  # 容忍15代无改进（增加从10）
                    print(f"  ✓ 已找到{len(feasible_at_depth)}个可行解且15代无改进，提前结束")
                    break
            # 注释掉：不再因为没有可行解而提前停止，充分探索高深度
            # elif no_improvement_count >= 5:
            #     print(f"  ⚠ 未找到可行解且5代无改进，提前结束")
            #     break
            
            # 创建下一代（只变异匹配方式，深度不变）
            offspring = []
            
            # 精英保留
            offspring.extend([copy.deepcopy(ind) for ind in population[:config.elite_size]])
            
            # 交叉和变异
            while len(offspring) < config.population_size:
                parent1 = tournament_selection(population, config.tournament_size)
                parent2 = tournament_selection(population, config.tournament_size)
                
                # 交叉（只交叉匹配方式）
                child = Individual()
                child.tree_depth = current_depth  # 保持深度不变
                
                crossover_point = random.randint(0, len(feature_list))
                child.feature_match_types = (
                    parent1.feature_match_types[:crossover_point] +
                    parent2.feature_match_types[crossover_point:]
                )
                
                # 变异（只变异匹配方式）
                for i in range(len(child.feature_match_types)):
                    if random.random() < config.mutation_rate:
                        child.feature_match_types[i] = random.choice(feature_match_options)
                
                # 应用高基数字段约束
                high_cardinality_features = {'Total length': 0, 'Src Port': 4, 'Dst Port': 5}
                for feat_name, idx in high_cardinality_features.items():
                    if idx < len(child.feature_match_types):
                        if child.feature_match_types[idx] == MatchType.EXACT:
                            child.feature_match_types[idx] = MatchType.RANGE
                
                # 生成树结构（硬剪枝+软剪枝）
                child.tree_structure = hard_prune(
                    copy.deepcopy(original_tree), 0, current_depth
                )
                # 应用软剪枝
                child.tree_structure = soft_prune(child.tree_structure)
                child.tree_structure, child.rule_statistics = convert_tree_to_match_types(
                    child.tree_structure, child.feature_match_types, feature_list
                )
                
                offspring.append(child)
            
            population = offspring[:config.population_size]
        
        # 深度探索总结（增强诊断）
        # 获取叶子节点数
        leaves_count = 0
        if best_at_depth and best_at_depth.tree_structure:
            leaves_count = get_tree_leaves_count(best_at_depth.tree_structure, 0)
        
        print(f"\n深度{current_depth}探索完成:")
        if best_at_depth:
            print(f"  叶子节点: {leaves_count}")
            print(f"  最优F1: {best_at_depth.fitness:.4f} {'(超限，fitness被置0)' if not best_at_depth.is_feasible and best_at_depth.fitness == 0 else ''}")
            print(f"  可行解数量: {len(feasible_at_depth)}")
            print(f"  最优可行: {'是' if best_at_depth.is_feasible else '否'}")
            if best_at_depth.is_feasible:
                print(f"  最优资源: TCAM={best_at_depth.pred_tcam:.2f}%, "
                      f"SRAM={best_at_depth.pred_sram:.4f}%, Stages={best_at_depth.pred_stages:.1f}")
            else:
                # 即使超限也显示F1和资源，方便学习
                # 需要重新计算真实F1
                true_f1 = evaluate_f1_score(best_at_depth.tree_structure, X_test, y_test, feature_list, class_names)
                print(f"  真实F1: {true_f1:.4f} (未限制时的性能)")
                print(f"  最优个体超限原因:")
                if best_at_depth.pred_tcam > config.limit_tcam:
                    print(f"    TCAM超限: {best_at_depth.pred_tcam:.2f}% > {config.limit_tcam}%")
                if best_at_depth.pred_sram > config.limit_sram:
                    print(f"    SRAM超限: {best_at_depth.pred_sram:.4f}% > {config.limit_sram}%")
                if best_at_depth.pred_stages > config.limit_stages:
                    print(f"    Stages超限: {best_at_depth.pred_stages:.1f} > {config.limit_stages}")
        else:
            print(f"  未找到任何解")
        
        depth_results[current_depth] = {
            'best': best_at_depth,
            'feasible_count': len(feasible_at_depth),
            'feasible_solutions': feasible_at_depth
        }
        
        # 更新全局最优（F1优先，F1相同时资源优先）
        if best_at_depth:
            if global_best is None:
                global_best = copy.deepcopy(best_at_depth)
            else:
                # F1更高，或F1相同但资源更优
                if (best_at_depth.fitness > global_best.fitness + 1e-6 or
                    (abs(best_at_depth.fitness - global_best.fitness) < 1e-6 and
                     calculate_resource_score(best_at_depth) < calculate_resource_score(global_best))):
                    global_best = copy.deepcopy(best_at_depth)
        
        # 合并可行解
        all_feasible_solutions.extend(feasible_at_depth)
        
        # 决策：是否继续降低深度
        if len(feasible_at_depth) >= 1:  # 只要找到1个可行解
            print(f"  ✓ 深度{current_depth}找到{len(feasible_at_depth)}个可行解")
            print(f"  🎯 已在深度{current_depth}找到可行解，终止探索（保证最高F1）")
            # 找到可行解后立即停止，不再降低深度
            break
        else:
            print(f"  ⚠ 深度{current_depth}未找到可行解，继续降低深度")
    
    print("\n" + "=" * 80)
    print("分层搜索完成")
    print("=" * 80)
    
    # 合并所有可行解并使用新的排序策略
    all_feasible_solutions.extend([sol for result in depth_results.values() 
                                   for sol in result['feasible_solutions']])
    
    # 去重
    unique_feasible = []
    for sol in all_feasible_solutions:
        is_dup = False
        for u in unique_feasible:
            if (u.tree_depth == sol.tree_depth and 
                u.feature_match_types == sol.feature_match_types):
                is_dup = True
                break
        if not is_dup:
            unique_feasible.append(sol)
    
    # 使用新的排序策略：F1优先，F1相同时资源消耗低优先
    unique_feasible = sort_solutions_by_priority(unique_feasible)
    
    return global_best, unique_feasible, depth_results


def test_hierarchical_ga():
    """测试分层深度优先GA"""
    # 记录总开始时间
    total_start_time = time.time()
    start_datetime = datetime.datetime.now()
    
    print("=" * 80)
    print("测试分层深度优先遗传算法")
    print("=" * 80)
    print(f"开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 修正路径：脚本现在位于 src/ 目录下，数据和模型在上一级目录
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    
    model_data_path = os.path.join(project_root, "model_data", "iscx")
    json_models_dir = os.path.join(project_root, "json_models")
    
    if not os.path.exists(json_models_dir):
        os.makedirs(json_models_dir)
    
    # 加载ISCX数据集
    print("\n加载ISCX数据集...")
    with open(os.path.join(model_data_path, "data_train_iscx_C.pkl"), "rb") as f:
        train_data = pickle.load(f)
    with open(os.path.join(model_data_path, "data_eval_iscx_C.pkl"), "rb") as f:
        test_data = pickle.load(f)
    
    # 核心修正：按照物理含义精准对齐 ISCX 的 10 维特征到我们的 8 维模型
    # 索引映射逻辑（基于数值观察）：
    # 0:Protocol, 1:IPv4 Flags, 2:TCP Flags(Reset?), 3:TCP Flags(Syn?), 
    # 4:TTL, 5:DataOffset?, 6:Window?, 7:Port1, 8:Port2, 9:Total Length
    
    # 我们需要的顺序: [Total length(9), Protocol(0), IPv4 Flags(1), TTL(4), Port1(7), Port2(8), Flags_R(2), Flags_S(3)]
    target_indices = [9, 0, 1, 4, 7, 8, 2, 3]
    
    x_train = train_data[:, target_indices]
    y_train = train_data[:, -1].astype(int)
    x_test = test_data[:, target_indices]
    y_test = test_data[:, -1].astype(int)
    
    print(f"训练集大小: {x_train.shape} (特征已精准对齐)")
    print(f"测试集大小: {x_test.shape} (特征已精准对齐)")
    
    feature_list = [
        'Total length', 'Protocol', 'IPV4 Flags (DF)', 'Time to live',
        'Src Port', 'Dst Port', 'TCP flags (Reset)', 'TCP flags (Syn)'
    ]
    # ISCX 是 6 分类任务 (0-5)
    class_names = np.array(['0', '1', '2', '3', '4', '5'])
    
    # 训练原始树
    print("\n训练原始决策树...")
    train_start = time.time()
    model = st.DecisionTreeClassifier(max_depth=12, random_state=5)
    model.fit(x_train, y_train)
    json_model = sklearn2json(model, feature_list, class_names)
    train_time = time.time() - train_start
    print(f"训练时间: {train_time:.2f}秒")
    
    # 运行分层GA
    print("\n运行分层深度优先遗传算法...")
    ga_start_time = time.time()
    global_best, unique_feasible, depth_results = hierarchical_depth_first_ga(
        original_tree=json_model,
        feature_list=feature_list,
        class_names=class_names,
        X_test=x_test,
        y_test=y_test,
        max_depth=12,
        min_depth=6,
        generations_per_depth=30,  # 15 → 30，更充分探索
        min_feasible_solutions=3
    )
    
    ga_time = time.time() - ga_start_time
    print(f"\n遗传算法运行时间: {ga_time:.2f}秒 ({ga_time/60:.2f}分钟)")
    
    # 结果汇总
    print("\n" + "=" * 80)
    print("最优解汇总")
    print("=" * 80)
    
    if global_best:
        print("\n全局最优解:")
        print_individual_details(global_best, feature_list)
        
        print(f"\n找到 {len(unique_feasible)} 个独特可行解")
        
        print("\n各深度表现:")
        for depth in sorted(depth_results.keys(), reverse=True):
            result = depth_results[depth]
            best = result['best']
            print(f"  深度{depth:2d}: F1={best.fitness if best else 0:.4f}, "
                  f"可行解={result['feasible_count']}, "
                  f"最优可行={best.is_feasible if best else False}")
        
        print("\n前10个可行解（按F1和资源消耗排序）:")
        for i, sol in enumerate(unique_feasible[:10]):
            resource_score = calculate_resource_score(sol)
            print(f"\n解 {i+1}:")
            print(f"  深度={sol.tree_depth}, F1={sol.fitness:.4f}")
            print(f"  规则数={sol.p4_rule_size}")
            print(f"  资源: TCAM={sol.pred_tcam:.2f}%, SRAM={sol.pred_sram:.4f}%, Stages={sol.pred_stages:.1f}")
            print(f"  资源评分={resource_score:.2f} (越低越好)")
            match_str = ", ".join([MatchType.get_name(mt) for mt in sol.feature_match_types])
            print(f"  匹配: [{match_str}]")
        
        # 保存结果
        print("\n保存结果...")
        result_json = os.path.join(json_models_dir, "hierarchical_ga_result.json")
        save_optimized_tree(global_best, feature_list, result_json)
        
        # 保存可行解
        feasible_json = os.path.join(json_models_dir, "hierarchical_feasible_solutions.json")
        import json
        feasible_data = []
        for i, sol in enumerate(unique_feasible):
            feasible_data.append({
                "rank": i + 1,
                "tree_depth": int(sol.tree_depth),
                "fitness": float(sol.fitness),
                "p4_rule_size": int(sol.p4_rule_size),
                "pred_tcam": float(sol.pred_tcam),
                "pred_sram": float(sol.pred_sram),
                "pred_stages": float(sol.pred_stages),
                "resource_score": float(calculate_resource_score(sol)),
                "feature_match_types": [int(x) for x in sol.feature_match_types],
                "match_type_names": [MatchType.get_name(x) for x in sol.feature_match_types]
            })
        
        with open(feasible_json, 'w', encoding='utf-8') as f:
            json.dump(feasible_data, f, indent=2, ensure_ascii=False)
        print(f"所有可行解已保存到: {feasible_json}")
        
        # 评估 - 使用Macro F1
        print("\n评估最优解（使用Macro F1）:")
        
        # 计算预测结果
        predictions = []
        for data in x_test:
            pred = predict(global_best.tree_structure, feature_list, class_names, data)
            predictions.append(int(pred))
        
        # 使用sklearn的classification_report计算Macro F1
        print("\n分类报告（测试集）:")
        report = classification_report(y_test, predictions, digits=6, output_dict=True)
        print(classification_report(y_test, predictions, digits=6))
        
        macro_f1 = report['macro avg']['f1-score']
        accuracy = report['accuracy']
        
        print("\n" + "="*80)
        print("最终性能指标")
        print("="*80)
        print(f"测试集 Accuracy: {accuracy:.6f}")
        print(f"测试集 Macro F1-Score: {macro_f1:.6f}")
        for cls in class_names:
            if cls in report:
                print(f"Class {cls} F1-Score: {report[cls]['f1-score']:.6f}")
        print("="*80)
    
    # 计算总运行时间
    total_time = time.time() - total_start_time
    end_datetime = datetime.datetime.now()
    
    print("\n" + "=" * 80)
    print("运行时间统计")
    print("=" * 80)
    print(f"开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"训练时间: {train_time:.2f}秒")
    print(f"遗传算法时间: {ga_time:.2f}秒 ({ga_time/60:.2f}分钟)")
    print(f"总运行时间: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    test_hierarchical_ga()
