# -*- coding: utf-8 -*-
"""
Tofino资源预测器
直接加载 Regression_tree 模型，不依赖 unified_predictor 的路径
"""

import os
import numpy as np
import joblib


class TofinoPredictor:
    """Tofino资源预测器"""
    
    # 硬件容量
    TCAM_CAPACITY = 288  # Tiles
    SRAM_CAPACITY = 960  # Tiles
    
    def __init__(self):
        self.tcam_model = None
        self.sram_model = None
        self.stages_model = None
        self.scaler = None
        self.available = False
        self._load_models()
    
    def _load_models(self):
        """加载预训练模型"""
        # 路径结构:
        # genetic_algorithm (当前) -> src -> Dryad (项目根目录) -> models
        current_dir = os.path.dirname(os.path.abspath(__file__))  # genetic_algorithm
        src_dir = os.path.dirname(current_dir)                     # src
        dryad_root = os.path.dirname(src_dir)                      # Dryad (即 Dryad/Dryad)
        
        # 模型目录现在位于 Dryad/Dryad/models
        model_dir = os.path.join(dryad_root, 'models')
        
        if not os.path.exists(model_dir):
            print(f"[TofinoPredictor] 模型目录不存在: {model_dir}")
            # 尝试向上查找 (兼容测试环境)
            alt_dir = os.path.normpath(os.path.join(current_dir, '../../../../Regression_tree/tofino/models'))
            if os.path.exists(alt_dir):
                print(f"[TofinoPredictor] 使用备用模型目录: {alt_dir}")
                model_dir = alt_dir
            else:
                return
        
        try:
            self.tcam_model = joblib.load(os.path.join(model_dir, 'tcam_model_v2.pkl'))
            self.sram_model = joblib.load(os.path.join(model_dir, 'sram_model_v2.pkl'))
            self.stages_model = joblib.load(os.path.join(model_dir, 'stages_model_v2.pkl'))
            self.scaler = joblib.load(os.path.join(model_dir, 'scaler_v2.pkl'))
            
            # 检查是否是集成模型
            self.tcam_is_ensemble = isinstance(self.tcam_model, list)
            self.sram_is_ensemble = isinstance(self.sram_model, list)
            self.stages_is_ensemble = isinstance(self.stages_model, list)
            
            self.available = True
        except Exception as e:
            print(f"[TofinoPredictor] 模型加载失败: {e}")
            self.available = False
    
    def _get_prediction(self, model, X_scaled, is_ensemble):
        """获取预测值（支持集成模型）"""
        if not is_ensemble:
            return model.predict(X_scaled)
        
        # 集成模型：加权平均
        models_list = model
        weights = []
        for name, info in models_list:
            mae = info.get('cv_mae', 1.0)
            weights.append(1.0 / (mae + 1e-8))
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        final_pred = np.zeros(X_scaled.shape[0])
        for i, (name, info) in enumerate(models_list):
            estimator = info['model']
            pred = estimator.predict(X_scaled)
            final_pred += weights[i] * pred
        
        return final_pred
    
    def predict(self, match_config, rule_count):
        """
        预测资源消耗
        
        参数:
            match_config: dict - {feature_name: match_type_code}
                          或 list - [8个匹配类型代码]
            rule_count: int - P4规则数
        
        返回:
            dict - {'tcam_pct': float, 'sram_pct': float, 'stages': int, ...}
        """
        # 构建匹配类型数组
        if isinstance(match_config, dict):
            feature_order = [
                'Total length', 'Protocol', 'IPV4 Flags (DF)', 'Time to live',
                'Src Port', 'Dst Port', 'TCP flags (Reset)', 'TCP flags (Syn)'
            ]
            match_types = [match_config.get(f, 1) for f in feature_order]
        else:
            match_types = list(match_config)
        
        if not self.available:
            return self._simple_estimate(match_types, rule_count)
        
        try:
            # 构建21维特征向量
            m_types = np.array(match_types, dtype=float)
            bit_widths = np.array([16, 8, 8, 8, 16, 16, 8, 8], dtype=float)
            
            # 标准化 size (与 unified_predictor 一致)
            norm_size = (rule_count - 256) / (20000 - 256)
            
            # 匹配类型计数
            counts = [
                np.sum(m_types == 0),  # exact
                np.sum(m_types == 1),  # ternary
                np.sum(m_types == 2),  # range
                np.sum(m_types == 3),  # lpm
            ]
            
            X = np.concatenate([m_types, bit_widths, [norm_size], counts]).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # 预测
            tcam_tiles = self._get_prediction(self.tcam_model, X_scaled, self.tcam_is_ensemble)[0]
            sram_tiles = self._get_prediction(self.sram_model, X_scaled, self.sram_is_ensemble)[0]
            stages_pred = self._get_prediction(self.stages_model, X_scaled, self.stages_is_ensemble)[0]
            
            # 转换为百分比
            tcam_pct = (tcam_tiles / self.TCAM_CAPACITY) * 100
            sram_pct = (sram_tiles / self.SRAM_CAPACITY) * 100
            stages = max(1, int(round(stages_pred)))
            
            return {
                'tcam_pct': max(0, tcam_pct),
                'sram_pct': max(0, sram_pct),
                'stages': stages,
                'tcam_tiles': max(0, tcam_tiles),
                'sram_tiles': max(0, sram_tiles),
                'rule_count': rule_count
            }
        except Exception as e:
            print(f"[TofinoPredictor] 预测失败: {e}")
            return self._simple_estimate(match_types, rule_count)
    
    def _simple_estimate(self, match_types, rule_count):
        """简化估算（备用）"""
        ternary_count = sum(1 for m in match_types if m == 1)
        range_count = sum(1 for m in match_types if m == 2)
        
        if ternary_count >= 5:
            tcam_per_rule = 0.01
        elif range_count >= 2:
            tcam_per_rule = 0.008
        else:
            tcam_per_rule = 0.006
        
        tcam_pct = rule_count * tcam_per_rule
        sram_pct = rule_count * 0.002
        stages = max(1, (rule_count + 999) // 1000)
        
        return {
            'tcam_pct': tcam_pct,
            'sram_pct': sram_pct,
            'stages': stages,
            'rule_count': rule_count
        }


# 全局单例
_predictor = None

def get_predictor():
    """获取预测器单例"""
    global _predictor
    if _predictor is None:
        _predictor = TofinoPredictor()
    return _predictor


def predict_resource(match_config, rule_count):
    """便捷函数：预测资源消耗"""
    return get_predictor().predict(match_config, rule_count)


if __name__ == '__main__':
    # 测试
    print("Testing TofinoPredictor...")
    predictor = TofinoPredictor()
    
    match_config = {
        'Total length': 2,      # range
        'Protocol': 1,          # ternary
        'IPV4 Flags (DF)': 1,   # ternary
        'Time to live': 2,      # range
        'Src Port': 2,          # range
        'Dst Port': 2,          # range
        'TCP flags (Reset)': 1, # ternary
        'TCP flags (Syn)': 1    # ternary
    }
    
    print("\nPredictions for various rule counts:")
    print(f"{'Rules':>8} | {'TCAM%':>8} | {'SRAM%':>8} | {'Stages':>6}")
    print("-" * 40)
    for rules in [100, 500, 1000, 2000, 5000, 10000]:
        result = predictor.predict(match_config, rules)
        print(f"{rules:>8} | {result['tcam_pct']:>8.2f} | {result['sram_pct']:>8.2f} | {result['stages']:>6}")
