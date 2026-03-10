import joblib
import os
import numpy as np
import pandas as pd

def analyze_results():
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    model_path = os.path.join(models_dir, 'bmv2_model.pkl')
    
    if not os.path.exists(model_path):
        print("错误：找不到模型文件")
        return

    model = joblib.load(model_path)
    feature_names = [
        'total_len', 'protocol', 'flags', 'ttl', 
        'src_port', 'dst_port', 'tcp_flags_2', 'tcp_flags_1', 
        'size'
    ]
    
    print("=== 特征重要性分析 ===")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print(f"{'排名':<5} {'特征名称':<15} {'重要性':<10}")
        print("-" * 35)
        for f in range(len(feature_names)):
            idx = indices[f]
            print(f"{f+1:<5} {feature_names[idx]:<15} {importances[idx]:.6f}")
            
        # 计算匹配方式特征的总重要性
        match_features = [f for f in feature_names if f != 'size']
        match_importance = sum(importances[feature_names.index(f)] for f in match_features)
        print("-" * 35)
        print(f"Size 特征重要性: {importances[feature_names.index('size')]:.6f}")
        print(f"匹配方式特征总重要性: {match_importance:.6f}")
    else:
        print("该模型不支持特征重要性分析")

if __name__ == "__main__":
    analyze_results()
