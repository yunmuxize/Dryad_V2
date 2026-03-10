import numpy as np
import pandas as pd
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

class UnifiedPredictor:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Paths
        self.tofino_model_dir = os.path.join(self.base_dir, 'tofino', 'models')
        self.bmv2_model_dir = os.path.join(self.base_dir, 'bmv2', 'models')
        
        # Load Tofino Models
        print("Loading Tofino models...")
        try:
            self.tofino_tcam_model = joblib.load(os.path.join(self.tofino_model_dir, 'tcam_model_v2.pkl'))
            self.tofino_sram_model = joblib.load(os.path.join(self.tofino_model_dir, 'sram_model_v2.pkl'))
            self.tofino_scaler = joblib.load(os.path.join(self.tofino_model_dir, 'scaler_v2.pkl'))
            # Check if models are lists (WeightedEnsemble)
            self.tofino_tcam_is_ensemble = isinstance(self.tofino_tcam_model, list)
            self.tofino_sram_is_ensemble = isinstance(self.tofino_sram_model, list)
            
            # If ensemble, we need weights. But wait, dryad_predictor_v2.py saves 'performance_v2.pkl' 
            # which contains weights? No, looking at the code:
            # if best_name == 'WeightedEnsemble':
            #     self.tcam_model = best_info['model']  # This is the list of (name, info) tuples
            #     self.tcam_weights = best_info['weights']
            # But joblib.dump saves self.tcam_model. 
            # If it was 'WeightedEnsemble', self.tcam_model is a list of (name, info) tuples.
            # Wait, in dryad_predictor_v2.py:
            # models_to_test['WeightedEnsemble'] = { 'model': sorted_models, ... }
            # sorted_models is a list of (name, info) tuples.
            # info['model'] is the actual model object.
            # So if I load it, I get a list of (name, info) tuples.
            # AND I need the weights.
            # The weights are NOT saved in 'tcam_model_v2.pkl' if it just dumps self.tcam_model (which is the list).
            # Wait, let's re-read dryad_predictor_v2.py lines 317-320:
            # if best_name == 'WeightedEnsemble':
            #     self.tcam_model = best_info['model'] 
            #     self.tcam_weights = best_info['weights']
            # And save_models (lines 606):
            # joblib.dump(self.tcam_model, ...)
            # It does NOT save self.tcam_weights separately in a named file, unless it's inside performance?
            # No, performance_v2.pkl saves self.performance dict.
            # This might be a bug in the Tofino script or I missed something.
            # Let's assume for now it's a single model or I can re-derive weights (1/MAE).
            # Actually, if self.tcam_model is the list of sorted_models, each item is (name, info).
            # info contains 'cv_mae'. I can re-calculate weights: weight = 1.0 / (info['cv_mae'] + 1e-6).
            
        except Exception as e:
            print(f"Error loading Tofino models: {e}")
            self.tofino_tcam_model = None
            
        # Load BMv2 Models
        print("Loading BMv2 models...")
        try:
            self.bmv2_model = joblib.load(os.path.join(self.bmv2_model_dir, 'bmv2_model.pkl'))
            self.bmv2_scaler = joblib.load(os.path.join(self.bmv2_model_dir, 'scaler.pkl'))
            self.bmv2_base_rss = joblib.load(os.path.join(self.bmv2_model_dir, 'base_rss.pkl'))
        except Exception as e:
            print(f"Error loading BMv2 models: {e}")
            self.bmv2_model = None

    def _get_tofino_prediction(self, model, X_scaled, is_ensemble):
        if not is_ensemble:
            return model.predict(X_scaled)
        
        # It is an ensemble (list of (name, info))
        # Re-calculate weights
        models_list = model
        weights = []
        for name, info in models_list:
            # Note: in the script, weight = 1.0 / (info['cv_mae'] + 1e-6/1e-8)
            # I'll use a small epsilon.
            mae = info.get('cv_mae', 1.0) # Fallback
            weights.append(1.0 / (mae + 1e-8))
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        final_pred = np.zeros(X_scaled.shape[0])
        for i, (name, info) in enumerate(models_list):
            estimator = info['model']
            pred = estimator.predict(X_scaled)
            final_pred += weights[i] * pred
            
        return final_pred

    def predict(self, match_types, size):
        """
        Predict all metrics.
        
        Args:
            match_types (list or np.array): 8 integers representing match types.
                                          0=Exact, 1=Ternary, 2=Range, 3=LPM (Check this mapping!)
            size (int): Table size.
            
        Returns:
            dict: Dictionary containing all predictions.
        """
        # Ensure inputs are correct format
        match_types = np.array(match_types).flatten()
        if len(match_types) != 8:
            raise ValueError("match_types must have length 8")
            
        # --- 1. Tofino Prediction ---
        # Feature Engineering for Tofino
        # Normalize size
        normalized_size = (size - 256) / (8192 - 256)
        
        # Base features (8 match types + normalized size)
        tofino_base = np.append(match_types, normalized_size).reshape(1, -1)
        
        # Enhanced features
        # match_types is (8,)
        # We need to do counts
        range_counts = np.sum(match_types == 2)
        ternary_counts = np.sum(match_types == 1)
        exact_counts = np.sum(match_types == 0)
        lpm_counts = np.sum(match_types == 3)
        
        complexity_scores = range_counts * 3 + ternary_counts * 2 + exact_counts * 1 + lpm_counts * 2.5
        range_size_interaction = range_counts * normalized_size
        ternary_size_interaction = ternary_counts * normalized_size
        total_special_matches = range_counts + ternary_counts + lpm_counts
        range_ratio = range_counts / 8
        ternary_ratio = ternary_counts / 8
        exact_ratio = exact_counts / 8
        
        derived_features = np.array([
            range_counts, ternary_counts, exact_counts, lpm_counts,
            complexity_scores,
            range_size_interaction, ternary_size_interaction,
            total_special_matches,
            range_ratio, ternary_ratio, exact_ratio
        ]).reshape(1, -1)
        
        tofino_X = np.column_stack([tofino_base, derived_features])
        tofino_X_scaled = self.tofino_scaler.transform(tofino_X)
        
        tcam_pred = self._get_tofino_prediction(self.tofino_tcam_model, tofino_X_scaled, self.tofino_tcam_is_ensemble)[0]
        sram_pred = self._get_tofino_prediction(self.tofino_sram_model, tofino_X_scaled, self.tofino_sram_is_ensemble)[0]
        
        # --- 2. BMv2 Prediction ---
        # Input: 8 match types + RAW size
        bmv2_base = np.append(match_types, size).reshape(1, -1)
        bmv2_X_scaled = self.bmv2_scaler.transform(bmv2_base)
        
        # Predict (Log space)
        bmv2_log_pred = self.bmv2_model.predict(bmv2_X_scaled)[0]
        # Inverse transform (expm1) + Base RSS
        bmv2_rss_pred = np.expm1(bmv2_log_pred) + self.bmv2_base_rss
        
        # --- 3. eBPF Prediction ---
        # Logic from report
        # match_types: 0=Exact, 1=Ternary, 2=Range, 3=LPM
        
        has_ternary = ternary_counts > 0
        has_lpm = lpm_counts > 0
        
        # Map Memory (This is the incremental memory usage for the map)
        if has_ternary:
            ebpf_map_memory = 58.0 * size
        elif has_lpm:
            ebpf_map_memory = 0.0
        else:
            ebpf_map_memory = 13.0 * size
            
        # Instructions
        if has_ternary:
            ebpf_instructions = 1077
        elif has_lpm:
            ebpf_instructions = 994
        else:
            ebpf_instructions = 992
            
        # JIT Size
        if has_ternary:
            ebpf_jit_size = 2919
        elif has_lpm:
            ebpf_jit_size = 2453
        else:
            ebpf_jit_size = 2454
            
        return {
            "Tofino": {
                "TCAM (%)": round(tcam_pred, 4),
                "SRAM (%)": round(sram_pred, 4)
            },
            "BMv2": {
                "Total RSS (MB)": round(bmv2_rss_pred / 1024 / 1024, 2),
                "Base RSS (MB)": round(self.bmv2_base_rss / 1024 / 1024, 2),
                "Delta RSS (KB)": round(np.expm1(bmv2_log_pred) / 1024, 2)
            },
            "eBPF": {
                "Map Memory (Bytes)": int(ebpf_map_memory),
                "Instructions": int(ebpf_instructions),
                "JIT Size (Bytes)": int(ebpf_jit_size)
            }
        }

def main():
    predictor = UnifiedPredictor()
    
    print("\n" + "="*50)
    print("Dryad Unified Resource Predictor")

        
        # Load Tofino Models
        print("Loading Tofino models...")
        try:
            self.tofino_tcam_model = joblib.load(os.path.join(self.tofino_model_dir, 'tcam_model_v2.pkl'))
            self.tofino_sram_model = joblib.load(os.path.join(self.tofino_model_dir, 'sram_model_v2.pkl'))
            self.tofino_scaler = joblib.load(os.path.join(self.tofino_model_dir, 'scaler_v2.pkl'))
            # Check if models are lists (WeightedEnsemble)
            self.tofino_tcam_is_ensemble = isinstance(self.tofino_tcam_model, list)
            self.tofino_sram_is_ensemble = isinstance(self.tofino_sram_model, list)
            
            # If ensemble, we need weights. But wait, dryad_predictor_v2.py saves 'performance_v2.pkl' 
            # which contains weights? No, looking at the code:
            # if best_name == 'WeightedEnsemble':
            #     self.tcam_model = best_info['model']  # This is the list of (name, info) tuples
            #     self.tcam_weights = best_info['weights']
            # But joblib.dump saves self.tcam_model. 
            # If it was 'WeightedEnsemble', self.tcam_model is a list of (name, info) tuples.
            # Wait, in dryad_predictor_v2.py:
            # models_to_test['WeightedEnsemble'] = { 'model': sorted_models, ... }
            # sorted_models is a list of (name, info) tuples.
            # info['model'] is the actual model object.
            # So if I load it, I get a list of (name, info) tuples.
            # AND I need the weights.
            # The weights are NOT saved in 'tcam_model_v2.pkl' if it just dumps self.tcam_model (which is the list).
            # Wait, let's re-read dryad_predictor_v2.py lines 317-320:
            # if best_name == 'WeightedEnsemble':
            #     self.tcam_model = best_info['model'] 
            #     self.tcam_weights = best_info['weights']
            # And save_models (lines 606):
            # joblib.dump(self.tcam_model, ...)
            # It does NOT save self.tcam_weights separately in a named file, unless it's inside performance?
            # No, performance_v2.pkl saves self.performance dict.
            # This might be a bug in the Tofino script or I missed something.
            # Let's assume for now it's a single model or I can re-derive weights (1/MAE).
            # Actually, if self.tcam_model is the list of sorted_models, each item is (name, info).
            # info contains 'cv_mae'. I can re-calculate weights: weight = 1.0 / (info['cv_mae'] + 1e-6).
            
        except Exception as e:
            print(f"Error loading Tofino models: {e}")
            self.tofino_tcam_model = None
            
        # Load BMv2 Models
        print("Loading BMv2 models...")
        try:
            self.bmv2_model = joblib.load(os.path.join(self.bmv2_model_dir, 'bmv2_model.pkl'))
            self.bmv2_scaler = joblib.load(os.path.join(self.bmv2_model_dir, 'scaler.pkl'))
            self.bmv2_base_rss = joblib.load(os.path.join(self.bmv2_model_dir, 'base_rss.pkl'))
        except Exception as e:
            print(f"Error loading BMv2 models: {e}")
            self.bmv2_model = None

    def _get_tofino_prediction(self, model, X_scaled, is_ensemble):
        if not is_ensemble:
            return model.predict(X_scaled)
        
        # It is an ensemble (list of (name, info))
        # Re-calculate weights
        models_list = model
        weights = []
        for name, info in models_list:
            # Note: in the script, weight = 1.0 / (info['cv_mae'] + 1e-6/1e-8)
            # I'll use a small epsilon.
            mae = info.get('cv_mae', 1.0) # Fallback
            weights.append(1.0 / (mae + 1e-8))
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        final_pred = np.zeros(X_scaled.shape[0])
        for i, (name, info) in enumerate(models_list):
            estimator = info['model']
            pred = estimator.predict(X_scaled)
            final_pred += weights[i] * pred
            
        return final_pred

    def predict(self, match_types, size):
        """
        Predict all metrics.
        
        Args:
            match_types (list or np.array): 8 integers representing match types.
                                          0=Exact, 1=Ternary, 2=Range, 3=LPM (Check this mapping!)
            size (int): Table size.
            
        Returns:
            dict: Dictionary containing all predictions.
        """
        # Ensure inputs are correct format
        match_types = np.array(match_types).flatten()
        if len(match_types) != 8:
            raise ValueError("match_types must have length 8")
            
        # --- 1. Tofino Prediction ---
        # Feature Engineering for Tofino
        # Normalize size
        normalized_size = (size - 256) / (8192 - 256)
        
        # Base features (8 match types + normalized size)
        tofino_base = np.append(match_types, normalized_size).reshape(1, -1)
        
        # Enhanced features
        # match_types is (8,)
        # We need to do counts
        range_counts = np.sum(match_types == 2)
        ternary_counts = np.sum(match_types == 1)
        exact_counts = np.sum(match_types == 0)
        lpm_counts = np.sum(match_types == 3)
        
        complexity_scores = range_counts * 3 + ternary_counts * 2 + exact_counts * 1 + lpm_counts * 2.5
        range_size_interaction = range_counts * normalized_size
        ternary_size_interaction = ternary_counts * normalized_size
        total_special_matches = range_counts + ternary_counts + lpm_counts
        range_ratio = range_counts / 8
        ternary_ratio = ternary_counts / 8
        exact_ratio = exact_counts / 8
        
        derived_features = np.array([
            range_counts, ternary_counts, exact_counts, lpm_counts,
            complexity_scores,
            range_size_interaction, ternary_size_interaction,
            total_special_matches,
            range_ratio, ternary_ratio, exact_ratio
        ]).reshape(1, -1)
        
        tofino_X = np.column_stack([tofino_base, derived_features])
        tofino_X_scaled = self.tofino_scaler.transform(tofino_X)
        
        tcam_pred = self._get_tofino_prediction(self.tofino_tcam_model, tofino_X_scaled, self.tofino_tcam_is_ensemble)[0]
        sram_pred = self._get_tofino_prediction(self.tofino_sram_model, tofino_X_scaled, self.tofino_sram_is_ensemble)[0]
        
        # --- 2. BMv2 Prediction ---
        # Input: 8 match types + RAW size
        bmv2_base = np.append(match_types, size).reshape(1, -1)
        bmv2_X_scaled = self.bmv2_scaler.transform(bmv2_base)
        
        # Predict (Log space)
        bmv2_log_pred = self.bmv2_model.predict(bmv2_X_scaled)[0]
        # Inverse transform (expm1) + Base RSS
        bmv2_rss_pred = np.expm1(bmv2_log_pred) + self.bmv2_base_rss
        
        # --- 3. eBPF Prediction ---
        # Logic from report
        # match_types: 0=Exact, 1=Ternary, 2=Range, 3=LPM
        
        has_ternary = ternary_counts > 0
        has_lpm = lpm_counts > 0
        
        # Map Memory (This is the incremental memory usage for the map)
        if has_ternary:
            ebpf_map_memory = 58.0 * size
        elif has_lpm:
            ebpf_map_memory = 0.0
        else:
            ebpf_map_memory = 13.0 * size
            
        # Instructions
        if has_ternary:
            ebpf_instructions = 1077
        elif has_lpm:
            ebpf_instructions = 994
        else:
            ebpf_instructions = 992
            
        # JIT Size
        if has_ternary:
            ebpf_jit_size = 2919
        elif has_lpm:
            ebpf_jit_size = 2453
        else:
            ebpf_jit_size = 2454
            
        return {
            "Tofino": {
                "TCAM (%)": round(tcam_pred, 4),
                "SRAM (%)": round(sram_pred, 4)
            },
            "BMv2": {
                "Total RSS (MB)": round(bmv2_rss_pred / 1024 / 1024, 2),
                "Base RSS (MB)": round(self.bmv2_base_rss / 1024 / 1024, 2),
                "Delta RSS (KB)": round(np.expm1(bmv2_log_pred) / 1024, 2)
            },
            "eBPF": {
                "Map Memory (Bytes)": int(ebpf_map_memory),
                "Instructions": int(ebpf_instructions),
                "JIT Size (Bytes)": int(ebpf_jit_size)
            }
        }

def print_results(results, match_types, size):
    """格式化打印预测结果"""
    print("\n" + "=" * 80)
    print(f"  输入参数: Match Types = {match_types}, Size = {size}")
    print("=" * 80)
    
    # Tofino
    print("\n【Tofino 预测】")
    print(f"  TCAM 使用率: {results['Tofino']['TCAM (%)']:.4f}%")
    print(f"  SRAM 使用率: {results['Tofino']['SRAM (%)']:.4f}%")
    
    # BMv2
    print("\n【BMv2 预测】")
    print(f"  总 RSS: {results['BMv2']['Total RSS (MB)']:.2f} MB")
    print(f"  基础 RSS: {results['BMv2']['Base RSS (MB)']:.2f} MB")
    print(f"  增量 RSS: {results['BMv2']['Delta RSS (KB)']:.2f} KB")
    
    # eBPF
    print("\n【eBPF 预测】")
    print(f"  Map 内存: {results['eBPF']['Map Memory (Bytes)']:.0f} Bytes")
    print(f"  预估指令数: {results['eBPF']['Instructions']:.0f}")
    print(f"  JIT 大小: {results['eBPF']['JIT Size (Bytes)']:.0f} Bytes")
    print("=" * 80)

def main():
    """主函数 - 支持命令行参数、交互式输入和示例模式"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Dryad P4 资源预测系统 - 统一预测器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  1. 命令行模式:
     python unified_predictor.py -m 2 1 1 1 0 0 1 0 -s 4024
     
  2. 交互式模式:
     python unified_predictor.py -i
     
  3. 示例模式 (默认):
     python unified_predictor.py

Match Types 编码:
  0 = Exact (精确匹配)
  1 = Ternary (三态匹配)
  2 = Range (范围匹配)
  3 = LPM (最长前缀匹配)
        """
    )
    
    parser.add_argument('-m', '--match-types', nargs=8, type=int, metavar='M',
                        help='8个匹配类型 (0=Exact, 1=Ternary, 2=Range, 3=LPM)')
    parser.add_argument('-s', '--size', type=int, metavar='SIZE',
                        help='表大小 (256-8192)')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='交互式输入模式')
    parser.add_argument('-e', '--examples', action='store_true',
                        help='运行示例预测')
    
    args = parser.parse_args()
    
    # 加载预测器
    print("正在加载预测模型...")
    predictor = UnifiedPredictor()
    print("模型加载完成！\n")
    
    # 模式 1: 命令行参数模式
    if args.match_types and args.size:
        results = predictor.predict(args.match_types, args.size)
        print_results(results, args.match_types, args.size)
    
    # 模式 2: 交互式模式
    elif args.interactive:
        print("=" * 80)
        print("  交互式预测模式")
        print("=" * 80)
        while True:
            try:
                print("\n请输入预测参数 (输入 'q' 退出):")
                
                # 输入 match types
                match_input = input("  Match Types (8个数字，空格分隔): ").strip()
                if match_input.lower() == 'q':
                    print("退出程序。")
                    break
                
                match_types = [int(x) for x in match_input.split()]
                if len(match_types) != 8:
                    print("  ❌ 错误: 必须输入8个匹配类型！")
                    continue
                
                if not all(0 <= m <= 3 for m in match_types):
                    print("  ❌ 错误: 匹配类型必须在 0-3 之间！")
                    continue
                
                # 输入 size
                size_input = input("  Size (256-8192): ").strip()
                if size_input.lower() == 'q':
                    print("退出程序。")
                    break
                
                size = int(size_input)
                if not (256 <= size <= 8192):
                    print("  ❌ 错误: Size 必须在 256-8192 之间！")
                    continue
                
                # 执行预测
                results = predictor.predict(match_types, size)
                print_results(results, match_types, size)
                
            except ValueError as e:
                print(f"  ❌ 输入错误: {e}")
            except KeyboardInterrupt:
                print("\n\n程序被中断。")
                break
            except Exception as e:
                print(f"  ❌ 预测错误: {e}")
    
    # 模式 3: 示例模式 (默认)
    else:
        print("=" * 80)
        print("  示例预测模式")
        print("=" * 80)
        print("\n提示: 使用 -h 查看所有使用方式\n")
        
        examples = [
            {
                'name': '用户测试案例',
                'match': [2, 1, 1, 1, 0, 0, 1, 0],
                'size': 4024,
                'note': '1个Range, 4个Ternary, 3个Exact'
            },
            {
                'name': '全精确匹配',
                'match': [0, 0, 0, 0, 0, 0, 0, 0],
                'size': 4096,
                'note': '8个Exact'
            },
            {
                'name': '单个三态匹配',
                'match': [1, 0, 0, 0, 0, 0, 0, 0],
                'size': 1024,
                'note': '1个Ternary, 7个Exact'
            },
            {
                'name': '单个LPM匹配',
                'match': [3, 0, 0, 0, 0, 0, 0, 0],
                'size': 8192,
                'note': '1个LPM, 7个Exact'
            },
            {
                'name': '混合匹配',
                'match': [2, 2, 1, 1, 0, 0, 1, 1],
                'size': 3768,
                'note': '2个Range, 3个Ternary, 3个Exact'
            }
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"\n{'─' * 80}")
            print(f"示例 {i}: {example['name']} ({example['note']})")
            print(f"{'─' * 80}")
            results = predictor.predict(example['match'], example['size'])
            print_results(results, example['match'], example['size'])
        
        print("\n" + "=" * 80)
        print("  提示: 使用以下命令进行自定义预测:")
        print("  python unified_predictor.py -m 2 1 1 1 0 0 1 0 -s 4024")
        print("  或使用交互模式: python unified_predictor.py -i")
        print("=" * 80)

if __name__ == "__main__":
    main()
