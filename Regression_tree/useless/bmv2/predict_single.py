# -*- coding: utf-8 -*-
"""
单样本预测脚本
使用训练好的模型预测CPU和Memory
"""

import sys
import os
from dryad_predictor_enhanced import DryadPredictorEnhanced

def predict_from_input(input_str):
    """
    从输入字符串进行预测
    
    Args:
        input_str: 输入字符串，格式为 "3,2,2,2,2,2,3,3,1235"
                   前8个是matching_pattern，第9个是size
    """
    # 解析输入
    try:
        values = [int(x.strip()) for x in input_str.split(',')]
    except ValueError as e:
        print(f"[ERROR] 输入格式错误: {e}")
        print("请输入9个整数，用逗号分隔，格式：matching_pattern(8个),size(1个)")
        return None
    
    if len(values) != 9:
        print(f"[ERROR] 输入数量错误: 需要9个值，实际得到{len(values)}个")
        print("格式：前8个是matching_pattern，第9个是size")
        return None
    
    matching_pattern = values[:8]
    size = values[8]
    
    print("=" * 60)
    print("Dryad BMv2 Resource Prediction")
    print("=" * 60)
    print(f"Input Features:")
    print(f"  Matching Pattern: {matching_pattern}")
    print(f"  Size: {size}")
    print("=" * 60)
    
    # 创建预测器并加载模型
    try:
        predictor = DryadPredictorEnhanced()
        if not predictor.load_models():
            print("[ERROR] 模型加载失败！请先训练模型。")
            return None
        
        # 进行预测
        result = predictor.predict(matching_pattern, size=size)
        
        # 显示结果
        print("\n" + "=" * 60)
        print("Prediction Results")
        print("=" * 60)
        print(f"CPU Usage:    {result['cpu']:.2f}")
        print(f"Memory Usage:  {result['memory']:.2f} MB")
        print("=" * 60)
        
        # 显示特征解释
        print("\nFeature Explanation:")
        print("  Matching Pattern Types:")
        print("    0 = Exact Match")
        print("    1 = LPM (Longest Prefix Match)")
        print("    2 = Ternary Match")
        print("    3 = Range Match")
        print(f"  Size: Packet size ({size} bytes)")
        
        # 分析特征
        range_count = matching_pattern.count(3)
        ternary_count = matching_pattern.count(2)
        lpm_count = matching_pattern.count(1)
        exact_count = matching_pattern.count(0)
        
        print(f"\nPattern Analysis:")
        print(f"  Range matches:    {range_count}")
        print(f"  Ternary matches:   {ternary_count}")
        print(f"  LPM matches:      {lpm_count}")
        print(f"  Exact matches:    {exact_count}")
        print(f"  Total complexity: {range_count * 4 + ternary_count * 3 + lpm_count * 2 + exact_count * 1}")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 从命令行参数读取
        input_str = sys.argv[1]
    else:
        # 使用默认输入
        input_str = "3,2,2,2,2,2,3,3,1235"
        print("使用默认输入（未提供命令行参数）")
    
    predict_from_input(input_str)

