import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 配置
CSV_FILE = 'ml_features.csv'
FEATURE_BITS = [16, 8, 1, 8, 16, 16, 1, 1]
FEATURE_NAMES = ['total_len', 'protocol', 'flags', 'ttl', 'src_port', 'dst_port', 'tcp_flags_2', 'tcp_flags_1']

def analyze_data():
    print("=== 1. 数据加载与基础统计 ===")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {CSV_FILE}")
        return

    print(f"总样本数: {len(df)}")
    print(f"内存 (RSS) 统计:\n{df['memory'].describe()}")
    
    # 检查缺失值
    if df.isnull().values.any():
        print("警告: 数据集中存在缺失值，将被丢弃。")
        df = df.dropna()

    print("\n=== 2. 寻找基准 RSS (Base_RSS) ===")
    # 策略: Size < 20
    small_size_df = df[df['size'] < 20]
    print(f"Size < 20 的样本数: {len(small_size_df)}")
    
    if len(small_size_df) > 0:
        print(f"小样本 RSS 均值: {small_size_df['memory'].mean():.4f}")
        print(f"小样本 RSS 中位数: {small_size_df['memory'].median():.4f}")
        print(f"小样本 RSS 最小值: {small_size_df['memory'].min():.4f}")
        print(f"小样本 RSS 最大值: {small_size_df['memory'].max():.4f}")
        print(f"小样本 RSS 标准差: {small_size_df['memory'].std():.4f}")
        
        # 进一步筛选：寻找匹配复杂度最低的
        # 假设 0 (Exact) 是最简单的。计算每行的匹配复杂度总和 (0=Exact, 1=LPM, 2=Ternary, 3=Range)
        match_cols = FEATURE_NAMES
        # 计算每行的复杂度分数 (简单求和)
        small_size_df = small_size_df.copy()
        small_size_df['complexity'] = small_size_df[match_cols].sum(axis=1)
        
        min_complexity = small_size_df['complexity'].min()
        base_candidates = small_size_df[small_size_df['complexity'] == min_complexity]
        
        print(f"\n在 Size < 20 中，最低复杂度分数 (sum of match types) 为: {min_complexity}")
        print(f"符合最低复杂度的样本数: {len(base_candidates)}")
        
        base_rss = base_candidates['memory'].mean()
        print(f"建议 Base_RSS (最低复杂度小样本均值): {base_rss:.4f}")
    else:
        print("未找到 Size < 20 的样本，无法按策略计算 Base_RSS")
        base_rss = df['memory'].min() # Fallback

    print("\n=== 3. 特征相关性分析 (是否需要 FEATURE_BITS) ===")
    # 构建基础特征矩阵
    X = df[FEATURE_NAMES + ['size']].copy()
    y = df['memory']
    
    # 计算 bit 相关的特征
    # 想法：不同的匹配类型对内存的影响可能与字段的位宽有关
    # 例如：16位的 Range 匹配可能比 1位的 Range 匹配消耗更多内存
    
    # 特征 A: 加权复杂度 (Match Type * Bits)
    # Match Type: 0=Exact, 1=LPM, 2=Ternary, 3=Range
    weighted_complexity = np.zeros(len(df))
    total_bits_used = np.zeros(len(df))
    
    for i, col in enumerate(FEATURE_NAMES):
        # 每一列的值 (0,1,2,3) * 该列的位宽
        weighted_complexity += df[col] * FEATURE_BITS[i]
        # 统计非 Exact 匹配的总位宽 (假设 Exact (0) 不增加额外开销，或者开销不同)
        # 这里简单统计所有参与匹配的位宽（其实所有都在参与，只是方式不同）
        # 让我们统计 "非精确匹配的位宽总和"
        is_not_exact = (df[col] > 0).astype(int)
        total_bits_used += is_not_exact * FEATURE_BITS[i]

    X['weighted_complexity'] = weighted_complexity
    X['non_exact_bits'] = total_bits_used
    
    # 计算相关系数
    X['target_memory'] = y
    corr = X.corr()['target_memory'].sort_values(ascending=False)
    print("\n各特征与 Memory 的皮尔逊相关系数:")
    print(corr)
    
    print("\n分析结论:")
    if abs(corr['weighted_complexity']) > abs(corr['size']) or abs(corr['non_exact_bits']) > 0.1:
        print("-> 建议加入位宽相关特征 (FEATURE_BITS 加权)")
    else:
        print("-> 位宽特征相关性较低，可能不需要加入")

if __name__ == "__main__":
    analyze_data()
