# -*- coding: utf-8 -*-
"""
DPDK 内存预测示例
使用训练好的模型进行预测
"""

import numpy as np
import joblib
import os


def load_model():
    """加载训练好的模型"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    
    model = joblib.load(os.path.join(models_dir, "memory_model_v2.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler_v2.pkl"))
    feature_names = joblib.load(os.path.join(models_dir, "feature_names_v2.pkl"))
    
    return model, scaler, feature_names


def extract_features_from_filename(filename):
    """从文件名提取特征
    
    例如: 2_2_1_1_0_0_1_0_3026.p4 或 2_2_1_1_0_0_1_0_3026
    返回: match_types=[2,2,1,1,0,0,1,0], table_size=3026
    """
    # 移除 .p4 后缀（如果有）
    filename = filename.replace('.p4', '')
    
    # 分割
    parts = filename.split('_')
    
    if len(parts) >= 9:
        match_types = [int(x) for x in parts[:8]]
        table_size = int(parts[8])
    else:
        raise ValueError(f"文件名格式错误: {filename}")
    
    return match_types, table_size


def create_feature_vector(match_types, table_size, exact_count, wildcard_count):
    """创建特征向量（与训练时一致）"""
    
    # 基础特征
    normalized_size = (table_size - 65) / (8177 - 65)
    features = match_types + [normalized_size, exact_count, wildcard_count]
    
    # 派生特征（必须与训练时完全一致）
    match_types_arr = np.array(match_types)
    exact_match_count = np.sum(match_types_arr == 0)
    ternary_match_count = np.sum(match_types_arr == 1)
    wildcard_match_count = np.sum(match_types_arr == 2)
    
    match_complexity = exact_count + wildcard_count * 2
    wildcard_ratio = wildcard_count / (exact_count + wildcard_count) if (exact_count + wildcard_count) > 0 else 0
    size_complexity_interaction = normalized_size * match_complexity
    match_type_complexity = exact_match_count * 1 + ternary_match_count * 1.5 + wildcard_match_count * 2
    total_matches = exact_match_count + ternary_match_count + wildcard_match_count
    wildcard_match_ratio = wildcard_match_count / total_matches if total_matches > 0 else 0
    
    features.extend([
        match_complexity,
        wildcard_ratio,
        size_complexity_interaction,
        match_type_complexity,
        exact_match_count,
        ternary_match_count,
        wildcard_match_count,
        wildcard_match_ratio
    ])
    
    return np.array([features])


def predict_memory(filename, exact_count=None, wildcard_count=None):
    """预测内存使用量
    
    Args:
        filename: P4 文件名，如 "2_2_1_1_0_0_1_0_3026" 或 "2_2_1_1_0_0_1_0_3026.p4"
        exact_count: Exact 匹配数量（如果不提供，从数据集查找）
        wildcard_count: Wildcard 匹配数量（如果不提供，从数据集查找）
    """
    # 提取匹配方式和表大小
    match_types, table_size = extract_features_from_filename(filename)
    
    # 如果没有提供 exact_count 和 wildcard_count，尝试从数据集查找
    if exact_count is None or wildcard_count is None:
        import pandas as pd
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, 'dataset.csv')
        df = pd.read_csv(csv_path)
        
        # 查找对应的记录
        p4_filename = filename if filename.endswith('.p4') else f"{filename}.p4"
        row = df[df['p4_file'] == p4_filename]
        
        if len(row) > 0:
            exact_count = int(row.iloc[0]['exact_count'])
            wildcard_count = int(row.iloc[0]['wildcard_count'])
            actual_memory = int(row.iloc[0]['table_memory_estimate_bytes'])
            print(f"[OK] 从数据集找到记录:")
            print(f"  - exact_count: {exact_count}")
            print(f"  - wildcard_count: {wildcard_count}")
            print(f"  - 实际内存: {actual_memory} bytes ({actual_memory/1024:.2f} KB)")
        else:
            raise ValueError(f"数据集中未找到 {p4_filename}，请手动提供 exact_count 和 wildcard_count")
    else:
        actual_memory = None
    
    # 加载模型
    print("\n加载模型...")
    model, scaler, feature_names = load_model()
    
    # 创建特征向量
    X = create_feature_vector(match_types, table_size, exact_count, wildcard_count)
    X_scaled = scaler.transform(X)
    
    # 预测
    pred_memory = model.predict(X_scaled)[0]
    
    # 输出结果
    print("\n" + "=" * 80)
    print("  预测结果")
    print("=" * 80)
    print(f"\n输入特征:")
    print(f"  - 文件名: {filename}")
    print(f"  - 匹配方式: {match_types}")
    print(f"  - Table Size: {table_size}")
    print(f"  - Exact Count: {exact_count}")
    print(f"  - Wildcard Count: {wildcard_count}")
    
    print(f"\n预测内存使用量:")
    print(f"  - {pred_memory:.2f} bytes")
    print(f"  - {pred_memory/1024:.2f} KB")
    print(f"  - {pred_memory/(1024*1024):.4f} MB")
    
    if actual_memory is not None:
        error = abs(pred_memory - actual_memory)
        error_percent = (error / actual_memory) * 100
        print(f"\n预测准确性:")
        print(f"  - 实际值: {actual_memory} bytes")
        print(f"  - 预测值: {pred_memory:.2f} bytes")
        print(f"  - 绝对误差: {error:.2f} bytes")
        print(f"  - 相对误差: {error_percent:.4f}%")
        
        if error_percent < 1:
            print(f"  - 评价: 优秀！")
        elif error_percent < 5:
            print(f"  - 评价: 良好")
        elif error_percent < 10:
            print(f"  - 评价: 一般")
        else:
            print(f"  - 评价: 需改进")
    
    print("=" * 80)
    
    return pred_memory


def main():
    """主程序"""
    print("=" * 80)
    print("  DPDK 内存预测工具")
    print("=" * 80)
    
    # 示例 1: 从数据集查找
    print("\n[示例 1] 预测: 2_2_1_1_0_0_1_0_3026")
    predict_memory("2_2_1_1_0_0_1_0_3026")
    
    # 示例 2: 手动指定参数
    print("\n\n[示例 2] 预测: 1_2_1_1_2_2_0_0_2341 (手动指定参数)")
    predict_memory("1_2_1_1_2_2_0_0_2341", exact_count=2, wildcard_count=6)
    
    # 示例 3: 自定义预测（不在数据集中）
    print("\n\n[示例 3] 自定义预测: 0_0_0_0_0_0_0_0_5000")
    try:
        predict_memory("0_0_0_0_0_0_0_0_5000", exact_count=8, wildcard_count=0)
    except ValueError as e:
        print(f"注意: {e}")
        print("使用默认参数进行预测...")
        predict_memory("0_0_0_0_0_0_0_0_5000", exact_count=8, wildcard_count=0)


if __name__ == "__main__":
    main()
