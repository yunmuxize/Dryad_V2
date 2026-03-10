# -*- coding:utf-8 -*-
import pickle
import json
import copy
import numpy as np
import time
import random
from collections import Counter
from sklearn.tree._tree import TREE_LEAF
import sklearn.tree as st
import graphviz
import os
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
# 引入真实脚本依赖
from predictors.entries_calculator import calculate_entries_count
from predictors.platform_predictor import TofinoPlatformPredictor
# os.environ["PATH"] += os.pathsep + 'D:/Graphviz/bin'

# ==================== 遗传算法相关数据结构和枚举 ====================

# 匹配方式枚举 (修正为：Exact=0, Ternary=1, Range=2, LPM/Prefix=3)
class MatchType:
    EXACT = 0      # 精准匹配
    TERNARY = 1    # 三元匹配
    RANGE = 2      # 范围匹配
    PREFIX = 3     # 最长前缀匹配 (LPM)
    
    @staticmethod
    def get_name(match_type):
        names = {
            MatchType.EXACT: "exact",
            MatchType.TERNARY: "ternary",
            MatchType.RANGE: "range", 
            MatchType.PREFIX: "prefix"
        }
        return names.get(match_type, "unknown")

# 个体（解决方案）类
@dataclass
class Individual:
    tree_structure: Dict[str, Any] = None      # 裁剪后的树结构
    feature_match_types: List[int] = None       # 每个特征对应的匹配方式
    tree_depth: int = 0                        # 修剪深度 (1-8)
    fitness: float = 0.0                       # 适应度分数
    p4_rule_size: int = 0                      # 转换后的 P4 规则数
    pred_sram: float = 0.0                     # 预测 SRAM
    pred_tcam: float = 0.0                     # 预测 TCAM
    pred_stages: float = 0.0                   # 预测 Stages
    is_feasible: bool = True                   # 可行性标记
    rule_statistics: Dict[str, Any] = None     # 规则统计信息
    
    def __post_init__(self):
        if self.feature_match_types is None:
            self.feature_match_types = []
        if self.rule_statistics is None:
            self.rule_statistics = {}

# 遗传算法配置类
@dataclass
class GAConfig:
    population_size: int = 50
    generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    tournament_size: int = 3
    elite_size: int = 5
    
    # 硬约束参数 - 经过用户指定后的初始值
    limit_tcam: float = 2.5    # TCAM 使用率上限 (%)
    limit_sram: float = 1.0    # SRAM 使用率上限 (%)
    limit_stages: float = 4.0  # Stage 数量上限

# ==================== 外部依赖占位接口 ====================

class P4RuleConverter:
    """输入树结构和匹配方式，输出转换后的 p4_rule_size"""
    def calculate_p4_rules(self, tree_structure, feature_match_types, feature_list):
        # 调用 entries_calculator.py 中的真实逻辑
        return calculate_entries_count(tree_structure, feature_match_types, feature_list)

class ResourcePredictionModel:
    """输入 21 维特征向量，输出 (pred_sram, pred_tcam, pred_stages)"""
    def __init__(self):
        # 初始化真实的 Tofino 预测器
        self.predictor = TofinoPlatformPredictor()

    def predict(self, feature_vector):
        """
        注意：feature_vector 是 21 维特征向量
        feature_match_types 是前 8 位
        p4_rule_size 是第 17 位 (索引16)
        """
        match_types = feature_vector[0:8].astype(int).tolist()
        size = int(feature_vector[16])
        
        # 调用 platform_predictor.py 返回预测结果
        res = self.predictor.predict(match_types, size)
        
        return res['sram_percent'], res['tcam_percent'], res['stages']

# ==================== 代码逻辑修改 ====================

# ==================== 原有函数保持不变 ====================

# 将sklearn模型转化为json模型
def sklearn2json(model, feature_list, class_names, node_index=0):
    json_model = {}
    if model.tree_.children_left[node_index] == -1:  # 叶子节点
        count_labels = zip(model.tree_.value[node_index, 0], class_names)
        json_model['value'] = [count for count, label in count_labels]
    else:  # 非叶节点
        count_labels = zip(model.tree_.value[node_index, 0], class_names)
        json_model['value'] = [count for count, label in count_labels]
        feature = feature_list[model.tree_.feature[node_index]]
        threshold = model.tree_.threshold[node_index]
        json_model['name'] = '{} <= {}'.format(feature, threshold)
        json_model['feature'] = '{}'.format(feature)
        json_model['threshold'] = '{}'.format(threshold)
        left_index = model.tree_.children_right[node_index]
        right_index = model.tree_.children_left[node_index]
        json_model['children'] = [sklearn2json(model, feature_list, class_names, right_index),
                                  sklearn2json(model, feature_list, class_names, left_index)]
    return json_model


# 将sklearn模型和json模型进行同步
def pruned_sklearn_model(sklearn_model, index, json_model):
    if "children" not in json_model:
        sklearn_model.children_left[index] = TREE_LEAF
        sklearn_model.children_right[index] = TREE_LEAF
    else:
        pruned_sklearn_model(sklearn_model, sklearn_model.children_left[index], json_model["children"][0])
        pruned_sklearn_model(sklearn_model, sklearn_model.children_right[index], json_model["children"][1])


# 决策树可视化
def draw_file(model, feature_list, class_names, pdf_file):
    dot_data = st.export_graphviz(
        model,
        out_file=None,
        feature_names=feature_list,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
        impurity=False,
    )
    graph = graphviz.Source(dot_data)
    graph.render(pdf_file)  # 在同级目录下生成tree.pdf文件
    print("The tree has been drawn in " + pdf_file + '.pdf')


# 计算树的叶节点数
def get_tree_leaves_count(json_model, count):
    if "children" not in json_model:
        return 1
    children = json_model["children"]
    for child in children:
        count += get_tree_leaves_count(child, 0)
    return count


# 计算树的最大深度以及节点数（叶节点+非叶节点）
def get_tree_max_depth_and_nodes_count(json_model):
    nodes_count = 0
    max_depth = 0
    stack1 = [json_model]  # 从根节点0开始
    stack2 = [0]  # 根节点的深度为0
    while len(stack1) > 0:
        json_model = stack1.pop()  # pop保证每个节点只会被访问一次
        depth = stack2.pop()
        if depth > max_depth:
            max_depth = depth
        nodes_count += 1
        if "children" in json_model:  # 是非叶节点
            children = json_model["children"]
            for child in children:
                stack1.append(child)  # 将孩子存入，并且深度加1
                stack2.append(depth + 1)
    return max_depth, nodes_count


# 输出模型结构
def output_model_structure(json_model):
    max_depth, nodes_count = get_tree_max_depth_and_nodes_count(json_model)
    leaves_count = get_tree_leaves_count(json_model, 0)
    rules = leaves_count  # 规则数 = 叶子节点数 = 从根到叶子的路径数
    print('The true depth of the tree =', max_depth)
    print('The number of leaves =', leaves_count)
    print('The number of all nodes =', nodes_count)
    print('The number of rules =', rules)


# 计算TP、TN、FP、FN
def get_node_confusion_matrix(json_model):
    value = json_model['value']
    if value[0] >= value[1]:
        class_name = 0
    else:
        class_name = 1
    TP = class_name * max(value)
    TN = (1 - class_name) * max(value)
    FP = class_name * min(value)
    FN = (1 - class_name) * min(value)
    return TP, TN, FP, FN


# 计算叶节点的混淆矩阵指标之和
def get_leaves_confusion_matrix(json_model, TP=0, TN=0, FP=0, FN=0):
    if "children" not in json_model:  # 叶节点
        return get_node_confusion_matrix(json_model)
    children = json_model["children"]
    for child in children:
        TP_, TN_, FP_, FN_ = get_leaves_confusion_matrix(child)
        TP += TP_
        TN += TN_
        FP += FP_
        FN += FN_
    return TP, TN, FP, FN


# 输出精度的评估指标
def output_metrics(TP, TN, FP, FN):
    print('TP =', TP)
    print('TN =', TN)
    print('FP =', FP)
    print('FN =', FN)
    print('%d/%d' % (TP+TN, TP + TN + FP + FN))
    print('Accuracy =', format((TP + TN) / (TP + TN + FP + FN), '.6f'))
    print('Precision score =', format(TP / (TP + FP), '.6f'))
    print('Recall score =', format(TP / (TP + FN), '.6f'))
    print('F1 score =', format(2 * TP / (TP + FP + TP + FN), '.6f'))


# 得到数据对应叶节点的value
def classify(json_model, feature_list, data):
    if "children" not in json_model:
        return json_model["value"]  # 到达叶子节点，完成测试

    feature = json_model["feature"]
    threshold = float(json_model["threshold"])
    feature_value = data[feature_list.index(feature)]
    if float(feature_value) <= threshold:
        child = json_model["children"][0]
        value = classify(child, feature_list, data)
    else:
        child = json_model["children"][1]
        value = classify(child, feature_list, data)

    return value


# 得到数据对应的class
def predict(json_model, feature_list, class_names, data):
    value = classify(json_model, feature_list, data)
    class_names_index = value.index(max(value))
    predict_result = class_names[class_names_index]
    return predict_result


# 输出测试精度
def output_testing_metrics(json_model, X, Y, feature_list, class_names):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for index, data in enumerate(X):
        predict_result = predict(json_model, feature_list, class_names, data)
        if predict_result == '1' and str(Y[index]) == '1':
            TP += 1
        if predict_result == '1' and str(Y[index]) == '0':
            FP += 1
        if predict_result == '0' and str(Y[index]) == '1':
            FN += 1
        if predict_result == '0' and str(Y[index]) == '0':
            TN += 1

    output_metrics(TP, TN, FP, FN)


# 得到节点所属的类别
def get_node_class_name(json_model):
    value = json_model['value']
    if value[0] >= value[1]:
        class_name = 0
    else:
        class_name = 1
    return class_name


# 得到叶节点所属的类别list
def get_leaves_class_name(json_model):
    stack = [json_model]  # 从根节点0开始
    class_name_list = []  # 记录每个叶节点的class
    while len(stack) > 0:
        json_model = stack.pop()  # pop保证每个节点只会被访问一次
        if "children" in json_model:  # 非叶节点
            children = json_model["children"]
            for child in children:
                stack.append(child)  # 将孩子存入
        else:  # 叶节点
            class_name_list.append(int(np.argmax(json_model['value'])))  # 转换为Python int类型

    return class_name_list


# 判断是否可以进行软剪枝
def can_be_simplified(json_model):
    class_name = int(np.argmax(json_model['value']))  # 得到节点所属的类别，转换为Python int类型
    class_name_list = get_leaves_class_name(json_model)  # 得到叶节点所属的类别list  子树节点数
    flag = 1  # 判断是否可以进行软剪枝，1为可以，0为不可以
    for i_class_name in class_name_list:  # 叶节点数
        if i_class_name != class_name:  # class不属于同一类
            flag = 0
            break
    return flag


# 加载数据
def load_data(ROOT_PATH):
    with open(f"{ROOT_PATH}/x_train.pkl", "rb") as tf:
        x_train = pickle.load(tf)
    with open(f"{ROOT_PATH}/y_train.pkl", "rb") as tf:
        y_train = pickle.load(tf)
    with open(f"{ROOT_PATH}/x_test.pkl", "rb") as tf:
        x_test = pickle.load(tf)
    with open(f"{ROOT_PATH}/y_test.pkl", "rb") as tf:
        y_test = pickle.load(tf)
    print('Size of x_train = %d x %d' % (len(x_train), len(x_train[0])))
    print('Size of y_train = %d x 1' % len(y_train))
    print('Size of x_test = %d x %d' % (len(x_test), len(x_test[0])))
    print('Size of y_test = %d x 1' % len(y_test))
    print(Counter(y_test))
    print(Counter(y_train))
    return x_train, y_train, x_test, y_test


# 修改版本
def hard_prune(json_model, now_depth, limit_depth):  # O(n) n:总结点数
    jsonNode = json_model
    jsonNodeQueue = []
    depthQueue = []
    jsonNodeQueue.append(jsonNode)
    depthQueue.append(0)

    while jsonNodeQueue:
        jsonNode = jsonNodeQueue.pop(0)
        depth = depthQueue.pop(0)
        jsonNode["tobedel"] = 0
        if "leafcount" in jsonNode:
            jsonNode["leafcount"][0] = 0
            jsonNode["leafcount"][1] = 0
        else:
            jsonNode["leafcount"] = []
            jsonNode["leafcount"].append(0)
            jsonNode["leafcount"].append(0)

        if "children" not in jsonNode:  # 叶节点
            jsonNode["leafcount"][0] = 1
        else:  # 非叶节点
            left_child = jsonNode["children"][0]
            right_child = jsonNode["children"][1]
            if depth == limit_depth:  # 找到要剪枝的部分，将其删除，删除后即为叶子节点
                del jsonNode["children"]
                jsonNode["leafcount"][0] = 1
            else:
                jsonNodeQueue.append(left_child)
                depthQueue.append(depth + 1)
                jsonNodeQueue.append(right_child)
                depthQueue.append(depth + 1)

    return json_model


# 软剪枝
def soft_prune(json_model):
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

            classname = int(np.argmax(currentNode['value']))  # 得到当前节点所属的类别，转换为Python int类型

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
            classNameStack.append(int(np.argmax(currentNode['value'])))  # push叶节点所属的类别，转换为Python int类型

        # turn to current node's brother right node
        if jsonNodeStack and jsonNodeStack[-1]["children"][0] is currentNode:
            jsonNode = jsonNodeStack[-1]["children"][1]
        else:
            jsonNode = None

    return json_model

# ==================== 遗传算法核心功能 ====================

# 评估树结构的F1 Score
def evaluate_f1_score(tree_structure, X_test, y_test, feature_list, class_names):
    """
    评估树结构的F1 Score
    """
    if tree_structure is None:
        return 0.0
    
    TP = 0  # True Positive
    TN = 0  # True Negative  
    FP = 0  # False Positive
    FN = 0  # False Negative
    
    for i, data in enumerate(X_test):
        try:
            predicted = predict(tree_structure, feature_list, class_names, data)
            actual = str(y_test[i])
            
            if predicted == '1' and actual == '1':
                TP += 1
            elif predicted == '0' and actual == '0':
                TN += 1
            elif predicted == '1' and actual == '0':
                FP += 1
            elif predicted == '0' and actual == '1':
                FN += 1
        except Exception as e:
            # 如果预测失败，跳过该样本
            continue
    
    # 计算F1 Score
    if TP + FP == 0:
        precision = 0.0
    else:
        precision = TP / (TP + FP)
    
    if TP + FN == 0:
        recall = 0.0
    else:
        recall = TP / (TP + FN)
    
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f1_score


# 计算匹配方式的复杂度
def calculate_match_complexity(feature_match_types):
    """
    计算匹配方式的复杂度（越简单的匹配方式得分越高）
    """
    complexity_scores = {
        MatchType.EXACT: 1.0,      # 精准匹配最简单
        MatchType.RANGE: 0.8,      # 范围匹配
        MatchType.PREFIX: 0.6,     # 前缀匹配
        MatchType.TERNARY: 0.4     # 三元匹配最复杂
    }
    
    if not feature_match_types:
        return 0.0
    
    total_score = sum(complexity_scores.get(match_type, 0.5) for match_type in feature_match_types)
    return total_score / len(feature_match_types)


# 计算个体的适应度
def calculate_fitness(individual, X_test, y_test, feature_list, class_names, config, 
                      rule_converter: P4RuleConverter, resource_model: ResourcePredictionModel):
    """
    重构后的数据驱动适应度逻辑
    1. 计算 P4 规则数
    2. 构建 21 维特征并预测资源
    3. 硬约束判定：超标则 fitness=0
    4. 满足约束则返回 F1-Score
    """
    if individual.tree_structure is None:
        individual.is_feasible = False
        return 0.0
    
    # 步骤 1: 获取 P4 规则数
    individual.p4_rule_size = rule_converter.calculate_p4_rules(
        individual.tree_structure, individual.feature_match_types, feature_list
    )
    
    # 步骤 2: 构建 21 维特征向量
    # [0-7] 匹配方式
    m_types = list(individual.feature_match_types)
    # [8-15] 固定位宽
    bit_widths = [16, 8, 8, 8, 16, 16, 8, 8]
    # [16] P4 规则大小
    size_feat = [float(individual.p4_rule_size)]
    # [17-20] 匹配方式计数 (必须与模型训练的顺序一致: Exact=0, Ternary=1, Range=2, LPM=3)
    counts = [individual.feature_match_types.count(i) for i in range(4)]
    
    feature_vector = np.array(m_types + bit_widths + size_feat + counts)
    
    # 步骤 3: 调用模型预测
    sram, tcam, stages = resource_model.predict(feature_vector)
    individual.pred_sram = sram
    individual.pred_tcam = tcam
    individual.pred_stages = stages
    
    # 步骤 4: 执行硬惩罚逻辑 (资源限制 + 硬件约束)
    
    # 4.1 硬件约束检查
    hardware_feasible = True
    
    # 约束 1: 一个表最多出现一次 LPM (编码为 3)
    lpm_count = individual.feature_match_types.count(MatchType.PREFIX)
    if lpm_count > 1:
        hardware_feasible = False
        
    # 约束 2: 1-bit 特征不允许使用 RANGE (编码为 2)
    one_bit_features = ['IPV4 Flags (DF)', 'TCP flags (Reset)', 'TCP flags (Syn)']
    for i, feature in enumerate(feature_list):
        if feature in one_bit_features:
            if individual.feature_match_types[i] == MatchType.RANGE:
                hardware_feasible = False
                break
    
    # 4.2 资源限制检查
    resource_feasible = (sram <= config.limit_sram and 
                         tcam <= config.limit_tcam and 
                         stages <= config.limit_stages)
    
    if not hardware_feasible or not resource_feasible:
        individual.is_feasible = False
        individual.fitness = 0.0
    else:
        individual.is_feasible = True
        # 计算 F1-Score 作为适应度
        f1 = evaluate_f1_score(individual.tree_structure, X_test, y_test, feature_list, class_names)
        individual.fitness = f1
        
    return individual.fitness


# 计算树中的规则数
def count_rules_in_tree(tree_structure):
    """
    计算树中的规则数（等于叶子节点数，即从根到叶子的路径数）
    """
    if tree_structure is None:
        return 0
    
    leaves_count = get_tree_leaves_count(tree_structure, 0)
    return leaves_count


# 根据目标规则数进行裁剪
def prune_by_rule_count(tree_structure, target_rule_count):
    """
    根据目标规则数进行裁剪
    """
    if tree_structure is None:
        return tree_structure
    
    current_rules = count_rules_in_tree(tree_structure)
    if current_rules <= target_rule_count:
        return tree_structure
    
    # 简单的策略：逐步增加深度限制直到规则数满足要求
    max_depth, _ = get_tree_max_depth_and_nodes_count(tree_structure)
    
    for depth in range(max_depth, 0, -1):
        test_tree = copy.deepcopy(tree_structure)
        test_tree = hard_prune(test_tree, 0, depth)
        test_rules = count_rules_in_tree(test_tree)
        
        if test_rules <= target_rule_count:
            return test_tree
    
    return tree_structure


# 将决策树的分割条件转换为指定的匹配方式
def convert_tree_to_match_types(tree_structure, feature_match_types, feature_list):
    """
    将决策树的分割条件转换为指定的匹配方式
    返回转换后的树和规则统计信息
    每条规则 = 从根节点到叶子节点的一条完整路径
    全局约束：每个特征在所有规则中只能使用一种匹配方式
    """
    if tree_structure is None:
        return None, {}
    
    converted_tree = copy.deepcopy(tree_structure)
    rule_statistics = {
        "rules_per_feature": {feature: 0 for feature in feature_list},
        "rules_per_match_type": {MatchType.get_name(mt): 0 for mt in [MatchType.RANGE, MatchType.PREFIX, MatchType.TERNARY, MatchType.EXACT]},
        "rule_details": [],
        "path_rules": [],  # 存储从根到叶子的完整路径规则
        "feature_match_consistency": {}  # 记录每个特征的匹配方式一致性
    }
    
    def convert_node(node, match_type_name=None):
        if "children" in node and "feature" in node:
            feature_name = node["feature"]
            threshold = float(node["threshold"])
            
            # 获取该特征对应的匹配方式
            feature_index = feature_list.index(feature_name) if feature_name in feature_list else 0
            match_type = feature_match_types[feature_index] if feature_index < len(feature_match_types) else MatchType.EXACT
            match_type_name = MatchType.get_name(match_type)
            
            # 记录特征-匹配方式对应关系
            if feature_name not in rule_statistics["feature_match_consistency"]:
                rule_statistics["feature_match_consistency"][feature_name] = match_type_name
            else:
                # 验证一致性
                if rule_statistics["feature_match_consistency"][feature_name] != match_type_name:
                    print(f"警告: 特征 {feature_name} 在不同位置使用了不同的匹配方式!")
            
            # 根据匹配类型转换分割条件
            node["match_type"] = match_type_name
            
            if match_type == MatchType.RANGE:
                node["range"] = [0, threshold]
            elif match_type == MatchType.PREFIX:
                node["prefix"] = str(int(threshold))
            elif match_type == MatchType.TERNARY:
                node["ternary"] = format(int(threshold), 'b')  # 转换为二进制
            elif match_type == MatchType.EXACT:
                node["exact"] = threshold
            
            # 递归处理子节点
            for child in node["children"]:
                convert_node(child, match_type_name)
    
    def extract_path_rules(node, current_path=[], rule_id=0):
        """
        提取从根节点到叶子节点的完整路径规则
        每个特征在所有规则中只能使用一种匹配方式
        """
        if "children" not in node:
            # 叶子节点，创建一条完整规则
            rule = {
                "rule_id": rule_id,
                "path": current_path.copy(),
                "leaf_value": node.get("value", []),
                "prediction": int(np.argmax(node.get("value", [0, 0])))  # 转换为Python int类型
            }
            rule_statistics["path_rules"].append(rule)
            
            # 统计路径中使用的特征（每个特征只统计一次）
            used_features = set()
            for condition in current_path:
                feature_name = condition["feature"]
                if feature_name not in used_features:
                    rule_statistics["rules_per_feature"][feature_name] += 1
                    rule_statistics["rules_per_match_type"][condition["match_type_name"]] += 1
                    used_features.add(feature_name)
            
            return rule_id + 1
        else:
            # 非叶子节点，继续构建路径
            feature_name = node["feature"]
            threshold = float(node["threshold"])
            match_type_name = node.get("match_type", "exact")
            
            # 左子树路径（<= threshold）
            left_condition = {
                "feature": feature_name,
                "threshold": threshold,
                "operator": "<=",
                "match_type_name": match_type_name
            }
            left_path = current_path + [left_condition]
            rule_id = extract_path_rules(node["children"][0], left_path, rule_id)
            
            # 右子树路径（> threshold）
            right_condition = {
                "feature": feature_name,
                "threshold": threshold,
                "operator": ">",
                "match_type_name": match_type_name
            }
            right_path = current_path + [right_condition]
            rule_id = extract_path_rules(node["children"][1], right_path, rule_id)
            
            return rule_id
    
    # 首先转换匹配方式
    convert_node(converted_tree)
    
    # 然后提取路径规则
    extract_path_rules(converted_tree)
    
    return converted_tree, rule_statistics


# 初始化种群
def initialize_population(population_size, original_tree, feature_match_options, feature_list, max_depth=10):
    """
    初始化种群 (改进版 - 深度偏置初始化)
    - 深度范围扩展到 [1, max_depth]
    - 使用深度偏置分布，鼓励深层树
    """
    population = []
    
    # 深度偏置分布：更高深度获得更高权重
    # 例如：depth 8-10 获得更多采样
    depth_weights = [1.0] * max_depth
    for i in range(max_depth):
        if i >= max_depth // 2:  # 后半部分深度权重更高
            depth_weights[i] = 2.0
        if i >= int(max_depth * 0.8):  # 最深20%权重更高
            depth_weights[i] = 3.0
    
    for _ in range(population_size):
        individual = Individual()
        
        # 1. 带偏置的深度随机生成 (1 到 max_depth)
        individual.tree_depth = random.choices(
            range(1, max_depth + 1), 
            weights=depth_weights,
            k=1
        )[0]
        
        # 2. 随机生成特征匹配方式
        individual.feature_match_types = [
            random.choice(feature_match_options) for _ in range(len(feature_list))
        ]
        
        # 3. 硬剪枝
        individual.tree_structure = hard_prune(
            copy.deepcopy(original_tree), 0, individual.tree_depth
        )
        
        # 4. 转换匹配方式
        individual.tree_structure, individual.rule_statistics = convert_tree_to_match_types(
            individual.tree_structure, individual.feature_match_types, feature_list
        )
        
        population.append(individual)
    
    return population


# 锦标赛选择
def tournament_selection(population, tournament_size):
    """
    锦标赛选择
    """
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda x: x.fitness)


# 轮盘赌选择
def roulette_wheel_selection(population):
    """
    轮盘赌选择
    """
    total_fitness = sum(individual.fitness for individual in population)
    if total_fitness == 0:
        return random.choice(population)
    
    r = random.uniform(0, total_fitness)
    cumulative_fitness = 0
    
    for individual in population:
        cumulative_fitness += individual.fitness
        if cumulative_fitness >= r:
            return individual
    
    return population[-1]


# 交叉操作
def crossover(parent1, parent2, feature_list, original_tree, max_depth=10):
    """
    交叉操作（单点交叉）
    """
    child = Individual()
    
    # 1. 深度交叉：随机选择一个父代的深度
    child.tree_depth = random.choice([parent1.tree_depth, parent2.tree_depth])
    
    # 2. 特征匹配方式交叉：单点交叉
    crossover_point = random.randint(0, len(feature_list))
    child.feature_match_types = (
        parent1.feature_match_types[:crossover_point] +
        parent2.feature_match_types[crossover_point:]
    )
    
    # 3. 生成对应的树结构
    child.tree_structure = hard_prune(
        copy.deepcopy(original_tree), 0, child.tree_depth
    )
    
    # 4. 转换匹配方式
    child.tree_structure, child.rule_statistics = convert_tree_to_match_types(
        child.tree_structure, child.feature_match_types, feature_list
    )
    
    return child


# 变异操作
def mutation(individual, feature_match_options, feature_list, original_tree, mutation_rate=0.2, max_depth=10):
    """
    变异操作 (改进版 - 提高变异率和探索范围)
    """
    mutated = copy.deepcopy(individual)
    
    # 1. 深度变异 (概率提高，范围扩大)
    if random.random() < mutation_rate * 1.5:  # 深度变异概率提高
        # 在当前深度 ± 2 范围内变异，鼓励大跳跃
        delta = random.choice([-2, -1, 1, 2])
        new_depth = max(1, min(max_depth, mutated.tree_depth + delta))
        
        # 增加深度偏置：更倾向于增加深度
        if delta > 0 and random.random() < 0.7:  # 70%概率接受深度增加
            new_depth = mutated.tree_depth + abs(delta)
            new_depth = min(max_depth, new_depth)
        
        if new_depth != mutated.tree_depth:
            mutated.tree_depth = new_depth
            mutated.tree_structure = hard_prune(
                copy.deepcopy(original_tree), 0, mutated.tree_depth
            )
    
    # 2. 特征匹配方式变异 (提高变异率)
    for i in range(len(mutated.feature_match_types)):
        if random.random() < mutation_rate:
            mutated.feature_match_types[i] = random.choice(feature_match_options)
    
    # 3. 更新树结构和规则统计
    if mutated.tree_structure:
        mutated.tree_structure, mutated.rule_statistics = convert_tree_to_match_types(
            mutated.tree_structure, mutated.feature_match_types, feature_list
        )
    
    # 清除旧的适应度信息
    mutated.fitness = 0.0
    mutated.p4_rule_size = 0
    mutated.pred_sram = 0.0
    mutated.pred_tcam = 0.0
    mutated.pred_stages = 0.0
    mutated.is_feasible = True
    
    return mutated


# 更新种群
def update_population(population, offspring, elite_size):
    """
    更新种群（精英保留策略）
    """
    # 按适应度排序
    population.sort(key=lambda x: x.fitness, reverse=True)
    
    # 保留精英个体
    new_population = population[:elite_size]
    
    # 添加子代
    new_population.extend(offspring)
    
    # 如果种群大小不够，用随机个体填充
    while len(new_population) < len(population):
        new_population.append(random.choice(population))
    
    return new_population[:len(population)]


# 遗传算法主流程
def genetic_algorithm_optimization(original_tree, num_stages, stage_rule_counts, 
                                 feature_match_options, feature_list, class_names,
                                 X_test, y_test, config=None, constraints=None, max_depth=10):
    """
    遗传算法主流程（改进版 - 支持可行解记录）
    
    参数:
    - original_tree: 原始完整的决策树
    - num_stages: stage数量
    - stage_rule_counts: 每个stage的规则数限制
    - feature_match_options: 每个特征可用的匹配方式列表
    - feature_list: 特征名称列表
    - class_names: 类别名称列表
    - X_test, y_test: 测试数据
    - config: 遗传算法配置
    - constraints: 约束条件字典
    - max_depth: 最大深度上限（默认10）
    
    返回:
    - best_individual: 最优个体
    - feasible_solutions: 所有可行解列表
    """
    if config is None:
        config = GAConfig()
    
    print("=" * 60)
    print("开始遗传算法优化")
    print(f"种群大小: {config.population_size}")
    print(f"迭代次数: {config.generations}")
    print(f"最大深度: {max_depth}")
    print(f"特征数量: {len(feature_list)}")
    
    print("约束条件:")
    if constraints:
        if constraints.get('max_rules'):
            print(f"  最大规则数: {constraints['max_rules']}")
        if constraints.get('min_rules'):
            print(f"  最小规则数: {constraints['min_rules']}")
        if constraints.get('max_depth'):
            print(f"  最大深度: {constraints['max_depth']}")
    else:
        print("  资源约束:")
        print(f"    TCAM ≤ {config.limit_tcam}%")
        print(f"    SRAM ≤ {config.limit_sram}%")
        print(f"    Stages ≤ {config.limit_stages}")
    
    print("=" * 60)
    
    # 初始化预测器逻辑
    rule_converter = P4RuleConverter()
    resource_model = ResourcePredictionModel()

    # 1. 初始化种群（传入max_depth）
    print("初始化种群...")
    population = initialize_population(
        config.population_size, original_tree, 
        feature_match_options, feature_list, max_depth
    )
    
    # 记录最佳个体和所有可行解
    best_individual = None
    best_fitness_history = []
    feasible_solutions = []  # 记录所有可行解
    
    for generation in range(config.generations):
        print(f"\n第 {generation + 1} 代:")
        
        # 2. 评估适应度
        for individual in population:
            if individual.fitness == 0.0:  # 未评估过
                calculate_fitness(individual, X_test, y_test, feature_list, 
                                class_names, config, rule_converter, resource_model)
        
        # 记录所有可行解（去重）
        for ind in population:
            if ind.is_feasible and ind.fitness > 0:
                # 检查是否已存在相同配置
                is_duplicate = False
                for fs in feasible_solutions:
                    if (fs.tree_depth == ind.tree_depth and 
                        fs.feature_match_types == ind.feature_match_types):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    feasible_solutions.append(copy.deepcopy(ind))
        
        # 3. 选择最佳个体
        population.sort(key=lambda x: x.fitness, reverse=True)
        current_best = population[0]
        
        if best_individual is None or current_best.fitness > best_individual.fitness:
            best_individual = copy.deepcopy(current_best)
        
        best_fitness_history.append(best_individual.fitness)
        
        # 打印当代最佳
        print(f"  当前最佳适应度: {current_best.fitness:.4f}")
        print(f"  全局最佳适应度: {best_individual.fitness:.4f}")
        print(f"  最佳个体深度: {best_individual.tree_depth}")
        print(f"  最佳个体 P4 规则数: {best_individual.p4_rule_size}")
        print(f"  最佳个体预测资源: SRAM={best_individual.pred_sram:.2f}%, TCAM={best_individual.pred_tcam:.2f}%, Stages={best_individual.pred_stages:.1f}")
        print(f"  累计可行解数量: {len(feasible_solutions)}")
        
        # 4. 创建下一代
        offspring = []
        
        # 精英选择
        elite_count = config.elite_size
        offspring.extend([copy.deepcopy(ind) for ind in population[:elite_count]])
        
        # 交叉和变异
        while len(offspring) < config.population_size:
            # 选择父代
            parent1 = tournament_selection(population, config.tournament_size)
            parent2 = tournament_selection(population, config.tournament_size)
            
            # 交叉
            if random.random() < config.crossover_rate:
                child = crossover(parent1, parent2, feature_list, original_tree, max_depth)
            else:
                child = copy.deepcopy(parent1)
            
            # 变异（提高变异率）
            child = mutation(child, feature_match_options, feature_list, 
                           original_tree, mutation_rate=0.25, max_depth=max_depth)
            
            offspring.append(child)
        
        population = offspring[:config.population_size]
        
        #  每10代检查种群多样性
        if (generation + 1) % 10 == 0:
            unique_fitness = len(set(ind.fitness for ind in population))
            print(f"\n第 {generation + 1} 代详细信息:")
            print(f"  平均适应度: {np.mean([ind.fitness for ind in population]):.4f}")
            print(f"  适应度标准差: {np.std([ind.fitness for ind in population]):.4f}")
            print(f"  适应度范围: {min(ind.fitness for ind in population):.4f} - {max(ind.fitness for ind in population):.4f}")
            print(f"  独特适应度数量: {unique_fitness}/{config.population_size}")
            
            if unique_fitness < config.population_size * 0.1:
                print(f"  警告: 种群多样性不足，增加变异强度")
    
    print("\n" + "=" * 60)
    print("遗传算法优化完成")
    print("=" * 60)
    
    # 按适应度排序可行解
    feasible_solutions.sort(key=lambda x: x.fitness, reverse=True)
    
    return best_individual, best_fitness_history, feasible_solutions

    
    # 记录最佳个体
    best_individual = None
    best_fitness_history = []
    
    for generation in range(config.generations):
        print(f"\n第 {generation + 1} 代:")
        
        # 2. 评估适应度
        for individual in population:
            individual.fitness = calculate_fitness(
                individual, X_test, y_test, 
                feature_list, class_names, config, 
                rule_converter, resource_model
            )
        
        # 3. 找到当前最佳个体
        current_best = max(population, key=lambda x: x.fitness)
        if best_individual is None or current_best.fitness > best_individual.fitness:
            best_individual = current_best
        
        best_fitness_history.append(best_individual.fitness)
        
        print(f"  当前最佳适应度: {current_best.fitness:.4f}")
        print(f"  全局最佳适应度: {best_individual.fitness:.4f}")
        print(f"  最佳个体 P4 规则数: {best_individual.p4_rule_size}")
        print(f"  最佳个体预测资源: SRAM={best_individual.pred_sram:.2f}%, TCAM={best_individual.pred_tcam:.2f}%, Stages={best_individual.pred_stages:.1f}")
        
        # 4. 选择、交叉、变异
        offspring = []
        
        # 生成子代
        for _ in range(config.population_size // 2):
            # 选择父代
            parent1 = tournament_selection(population, config.tournament_size)
            parent2 = tournament_selection(population, config.tournament_size)
            
            # 交叉
            child1, child2 = crossover(parent1, parent2, config.crossover_rate)
            
            # 变异
            child1 = mutation(child1, config.mutation_rate, feature_match_options, feature_list, original_tree)
            child2 = mutation(child2, config.mutation_rate, feature_match_options, feature_list, original_tree)
            
            offspring.extend([child1, child2])
        
        # 5. 更新种群
        population = update_population(population, offspring, config.elite_size)
        
        # 每10代输出一次详细信息
        if (generation + 1) % 10 == 0:
            fitnesses = [ind.fitness for ind in population]
            print(f"\n第 {generation + 1} 代详细信息:")
            print(f"  平均适应度: {np.mean(fitnesses):.4f}")
            print(f"  适应度标准差: {np.std(fitnesses):.4f}")
            print(f"  适应度范围: {min(fitnesses):.4f} - {max(fitnesses):.4f}")
            
            # 检查种群多样性
            unique_fitnesses = len(set([round(f, 4) for f in fitnesses]))
            print(f"  独特适应度数量: {unique_fitnesses}/{len(population)}")
            
            # 如果多样性不足，增加变异
            if unique_fitnesses < len(population) * 0.3:  # 如果独特适应度少于30%
                print("  警告: 种群多样性不足，增加变异强度")
                # 可以在这里增加额外的变异操作
    
    print("\n" + "=" * 60)
    print("遗传算法优化完成")
    print("=" * 60)
    
    return best_individual, best_fitness_history


# 打印个体的详细信息
def print_individual_details(individual, feature_list):
    """
    打印个体的详细信息
    """
    print("\n最优个体详细信息:")
    print(f"适应度: {individual.fitness:.4f}")
    print(f"P4 规则数: {individual.p4_rule_size}")
    print(f"修剪深度: {individual.tree_depth}")
    
    print("\n特征匹配方式:")
    for i, (feature, match_type) in enumerate(zip(feature_list, individual.feature_match_types)):
        print(f"  {feature}: {MatchType.get_name(match_type)}")
    
    print("\n树结构统计:")
    if individual.tree_structure:
        max_depth, nodes_count = get_tree_max_depth_and_nodes_count(individual.tree_structure)
        leaves_count = get_tree_leaves_count(individual.tree_structure, 0)
        print(f"  最大深度: {max_depth}")
        print(f"  节点总数: {nodes_count}")
        print(f"  叶子节点数: {leaves_count}")
        print(f"  规则数: {leaves_count}")  # 规则数 = 叶子节点数


# 保存优化后的树结构
def save_optimized_tree(individual, feature_list, filename):
    """
    保存优化后的树模型和配置到JSON文件（带异常处理）
    """
    save_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fitness": float(individual.fitness),
        "tree_depth": int(individual.tree_depth),
        "p4_rule_size": int(individual.p4_rule_size),
        "pred_sram": float(individual.pred_sram),
        "pred_tcam": float(individual.pred_tcam),
        "pred_stages": float(individual.pred_stages),
        "is_feasible": bool(individual.is_feasible),
        "feature_match_types": [int(x) for x in individual.feature_match_types],
        "feature_names": feature_list,
        "tree_structure": individual.tree_structure,
        "rule_statistics": individual.rule_statistics
    }
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        print(f"优化结果已保存到: {filename}")
    except PermissionError:
        # 文件被占用，尝试添加时间戳后缀
        alt_filename = filename.replace('.json', f'_{int(time.time())}.json')
        print(f"警告：无法写入 {filename}（文件被占用）")
        print(f"尝试保存到备用文件: {alt_filename}")
        try:
            with open(alt_filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            print(f"优化结果已保存到: {alt_filename}")
        except Exception as e:
            print(f"错误：无法保存优化结果 - {e}")
    print(f"包含 {len(feature_list)} 个特征的匹配方式信息")
    print(f"P4 规则数: {individual.p4_rule_size}")
    print(f"修剪深度: {individual.tree_depth}")
    print(f"适应度: {individual.fitness:.4f}")

# 加载优化后的树结构
def load_optimized_tree(filename):
    """
    加载优化后的树结构
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            save_data = json.load(f)
        
        print(f"成功加载优化结果: {filename}")
        return save_data
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filename}")
        return None
    except json.JSONDecodeError:
        print(f"错误: 文件 {filename} 格式不正确")
        return None


# 导出特征匹配方式摘要到单独文件
def export_feature_match_summary(filename, output_filename=None):
    """
    导出特征匹配方式摘要到单独文件
    """
    save_data = load_optimized_tree(filename)
    if save_data is None:
        return
    
    if output_filename is None:
        output_filename = filename.replace('.json', '_feature_summary.txt')
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("特征匹配方式摘要\n")
        f.write("=" * 50 + "\n\n")
        
        # 优化信息
        opt_info = save_data.get("optimization_info", {})
        f.write(f"优化时间: {opt_info.get('optimization_timestamp', 'N/A')}\n")
        f.write(f"适应度: {opt_info.get('fitness', 'N/A'):.4f}\n")
        f.write(f"P4 规则数: {opt_info.get('p4_rule_size', 'N/A')}\n")
        f.write(f"修剪深度: {opt_info.get('tree_depth', 'N/A')}\n\n")
        
        # 特征匹配方式
        f.write("特征匹配方式:\n")
        f.write("-" * 30 + "\n")
        feature_mapping = save_data.get("feature_match_mapping", {})
        for feature, match_info in feature_mapping.items():
            f.write(f"{feature:25s}: {match_info.get('match_type_name', 'N/A')}\n")
        
        # 统计信息
        stats = save_data.get("statistics", {})
        f.write(f"\n统计信息:\n")
        f.write(f"总特征数: {stats.get('total_features', 'N/A')}\n")
        f.write(f"总Stage数: {stats.get('total_stages', 'N/A')}\n")
        
        # 匹配方式分布
        distribution = stats.get("match_type_distribution", {})
        f.write(f"\n匹配方式分布:\n")
        for match_name, count in distribution.items():
            f.write(f"{match_name:10s}: {count} 个特征\n")
        
        # 规则统计信息
        rule_stats = save_data.get("rule_statistics", {})
        if rule_stats:
            f.write(f"\n规则统计信息:\n")
            f.write(f"每个特征的规则数:\n")
            rules_per_feature = rule_stats.get("rules_per_feature", {})
            for feature, count in rules_per_feature.items():
                if count > 0:
                    f.write(f"  {feature:25s}: {count} 条规则\n")
            
            f.write(f"\n每个匹配方式的规则数:\n")
            rules_per_match_type = rule_stats.get("rules_per_match_type", {})
            for match_type, count in rules_per_match_type.items():
                if count > 0:
                    f.write(f"  {match_type:10s}: {count} 条规则\n")
            
            f.write(f"\n路径规则详细信息:\n")
            path_rules = rule_stats.get("path_rules", [])
            for rule in path_rules:
                f.write(f"  规则 {rule['rule_id']:2d}: 预测={rule['prediction']} (叶子值: {rule['leaf_value']})\n")
                for i, condition in enumerate(rule['path']):
                    indent = "    " + "  " * i
                    f.write(f"{indent}{condition['feature']:20s} {condition['operator']:2s} {condition['threshold']:8.2f} ({condition['match_type_name']})\n")
                f.write("\n")
    
    print(f"特征匹配方式摘要已保存到: {output_filename}")


# 验证规则数量是否正确
def validate_rule_count(tree_structure, rule_statistics):
    """
    验证规则数量是否正确
    """
    if tree_structure is None or not rule_statistics:
        return False
    
    # 计算叶子节点数（应该等于路径规则数）
    leaves_count = get_tree_leaves_count(tree_structure, 0)
    path_rules_count = len(rule_statistics.get("path_rules", []))
    
    print(f"\n规则数量验证:")
    print(f"  叶子节点数: {leaves_count}")
    print(f"  路径规则数: {path_rules_count}")
    print(f"  是否匹配: {'✓' if leaves_count == path_rules_count else '✗'}")
    
    return leaves_count == path_rules_count


# 生成详细的规则分配报告
def generate_rule_allocation_report(filename, output_filename=None):
    """
    生成详细的规则分配报告
    """
    save_data = load_optimized_tree(filename)
    if save_data is None:
        return
    
    if output_filename is None:
        output_filename = filename.replace('.json', '_rule_allocation_report.txt')
    
    rule_stats = save_data.get("rule_statistics", {})
    if not rule_stats:
        print("没有找到规则统计信息")
        return
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("规则分配详细报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 基本信息
        opt_info = save_data.get("optimization_info", {})
        f.write(f"优化时间: {opt_info.get('optimization_timestamp', 'N/A')}\n")
        f.write(f"适应度: {opt_info.get('fitness', 'N/A'):.4f}\n")
        f.write(f"P4 总规则数: {opt_info.get('p4_rule_size', 'N/A')}\n")
        f.write(f"修剪深度: {opt_info.get('tree_depth', 'N/A')}\n\n")
        
        # 1. 每个特征的使用统计
        f.write("1. 每个特征的使用统计:\n")
        f.write("-" * 50 + "\n")
        rules_per_feature = rule_stats.get("rules_per_feature", {})
        total_rules = sum(rules_per_feature.values())
        
        for feature, count in sorted(rules_per_feature.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                percentage = (count / total_rules * 100) if total_rules > 0 else 0
                f.write(f"  {feature:25s}: {count:3d} 次 ({percentage:5.1f}%)\n")
        
        f.write(f"\n  总计: {total_rules} 次特征使用\n\n")
        
        # 2. 每个匹配方式的使用统计
        f.write("2. 每个匹配方式的使用统计:\n")
        f.write("-" * 50 + "\n")
        rules_per_match_type = rule_stats.get("rules_per_match_type", {})
        
        for match_type, count in sorted(rules_per_match_type.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                percentage = (count / total_rules * 100) if total_rules > 0 else 0
                f.write(f"  {match_type:10s}: {count:3d} 次 ({percentage:5.1f}%)\n")
        
        # 3. 特征-匹配方式矩阵
        f.write("\n3. 特征-匹配方式使用矩阵:\n")
        f.write("-" * 60 + "\n")
        feature_list = save_data.get("feature_list", [])
        feature_match_mapping = save_data.get("feature_match_mapping", {})
        
        f.write(f"{'特征名称':25s} {'匹配方式':10s} {'使用次数':8s} {'百分比':8s}\n")
        f.write("-" * 60 + "\n")
        for feature in feature_list:
            match_info = feature_match_mapping.get(feature, {})
            match_name = match_info.get("match_type_name", "N/A")
            use_count = rules_per_feature.get(feature, 0)
            percentage = (use_count / total_rules * 100) if total_rules > 0 else 0
            f.write(f"{feature:25s} {match_name:10s} {use_count:8d} {percentage:7.1f}%\n")
        
        # 4. 所有路径规则详情
        f.write("\n4. 所有路径规则详情:\n")
        f.write("-" * 50 + "\n")
        path_rules = rule_stats.get("path_rules", [])
        
        for rule in path_rules:
            f.write(f"规则 {rule['rule_id']:2d}: 预测={rule['prediction']} (叶子值: {rule['leaf_value']})\n")
            for i, condition in enumerate(rule['path']):
                indent = "  " + "  " * i
                f.write(f"{indent}{condition['feature']:20s} {condition['operator']:2s} {condition['threshold']:8.2f} ({condition['match_type_name']})\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"规则分配报告已保存到: {output_filename}")


# 遗传算法优化示例
def example_genetic_algorithm_optimization(max_rules=None, min_rules=None, max_depth_limit=None, num_stages=None, rules_per_stage=None, feature_match_options=None):    
    print("\n" + "=" * 80)
    print("遗传算法优化开始")
    print("=" * 80)

    # 获取脚本所在目录，用于构建绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_data_path = os.path.join(base_dir, "model_data")
    json_models_dir = os.path.join(base_dir, "json_models")
    if not os.path.exists(json_models_dir):
        os.makedirs(json_models_dir)

    # 1. 加载数据
    print("1. 加载数据...")
    feature_list = [
        'Total length', 'Protocol', 'IPV4 Flags (DF)', 'Time to live',
        'Src Port', 'Dst Port', 'TCP flags (Reset)', 'TCP flags (Syn)'
    ]
    class_names = np.array(['0', '1'])
    x_train, y_train, x_test, y_test = load_data(ROOT_PATH=model_data_path)
    
    # 2. 训练原始模型
    print("\n2. 训练原始决策树...")
    max_depth = 12
    model = st.DecisionTreeClassifier(max_depth=max_depth, random_state=5)
    model.fit(x_train, y_train)
    
    # 3. 转换为JSON模型
    print("\n3. 转换为JSON模型...")
    json_model = sklearn2json(model, feature_list, class_names)
    json_model = hard_prune(json_model, 0, max_depth)
    
    print("原始模型结构:")
    output_model_structure(json_model)
    
    # 4. 设置遗传算法参数
    print("\n4. 设置遗传算法参数...")
    
    # 遗传算法配置
    config = GAConfig(
        population_size=50,  # 恢复种群大小
        generations=10,      # 恢复迭代次数
        crossover_rate=0.8,
        mutation_rate=0.2,    
        tournament_size=3,
        elite_size=2          
    )
    
    # 5. 运行遗传算法优化
    print("\n5. 运行遗传算法优化...")
    best_individual, fitness_history = genetic_algorithm_optimization(
        original_tree=json_model,
        num_stages=None,
        stage_rule_counts=None,
        feature_match_options=feature_match_options,
        feature_list=feature_list,
        class_names=class_names,
        X_test=x_test,
        y_test=y_test,
        config=config,
        constraints=None
    )
    
    # 6. 输出结果
    print("\n6. 优化结果:")
    print_individual_details(best_individual, feature_list)
    
    # 7. 评估优化后的模型
    print("\n7. 评估优化后的模型:")
    print("优化后的测试精度:")
    output_testing_metrics(best_individual.tree_structure, x_test, y_test, feature_list, class_names)
    
    # 8. 保存结果
    print("\n8. 保存优化结果...")
    result_json = os.path.join(json_models_dir, 'genetic_optimized_tree.json')
    save_optimized_tree(best_individual, feature_list, result_json)
    
    # 8.1 验证规则数量
    print("\n8.1 验证规则数量:")
    validate_rule_count(best_individual.tree_structure, best_individual.rule_statistics)
    
    # 8.2 导出特征匹配方式摘要
    print("\n8.2 导出特征匹配方式摘要:")
    export_feature_match_summary(result_json)

    # 8.3 生成详细规则分配报告:
    print("\n8.3 生成详细规则分配报告:")
    generate_rule_allocation_report(result_json)
    
    # 9. 绘制适应度曲线
    print("\n9. 适应度进化历史:")
    for i, fitness in enumerate(fitness_history):
        print(f"  第 {i+1} 代: {fitness:.4f}")
    
    print("\n" + "=" * 80)
    print("遗传算法优化完成")
    print("=" * 80)
    
    return best_individual, fitness_history


if __name__ == '__main__':
    num_stages=12  # stage数量
    rules_per_stage=100  # 每个stage的规则数限制
    max_rules = num_stages * rules_per_stage  # 最大规则数限制
    min_rules=50  # 最小规则数限制
    max_depth_limit=num_stages  # 最大深度限制
    feature_match_options=[
        MatchType.RANGE, MatchType.PREFIX, MatchType.TERNARY, MatchType.EXACT
    ]  # 每个特征可用的匹配方式

    best_individual_constrained, fitness_history_constrained = example_genetic_algorithm_optimization(
        max_rules=max_rules,
        min_rules=min_rules,
        max_depth_limit=max_depth_limit,
        num_stages=num_stages,
        rules_per_stage=rules_per_stage,
        feature_match_options=feature_match_options
    )
