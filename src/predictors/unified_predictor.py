import numpy as np
import pandas as pd
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

class UnifiedPredictor:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Paths: src/ -> Dryad/ -> Dryad_V2/Dryad/ (project root where Regression_tree is)
        # Need to go up TWO levels from src/
        src_parent = os.path.dirname(self.base_dir)  # .../Dryad/Dryad
        project_root = os.path.dirname(src_parent)    # .../Dryad_V2/Dryad
        
        self.tofino_model_dir = os.path.join(project_root, 'Regression_tree', 'tofino', 'models')
        self.bmv2_model_dir = os.path.join(project_root, 'Regression_tree', 'bmv2', 'models')
        
        # Load Tofino Models
        print("Loading Tofino models...")
        try:
            self.tofino_tcam_model = joblib.load(os.path.join(self.tofino_model_dir, 'tcam_model_v2.pkl'))
            self.tofino_sram_model = joblib.load(os.path.join(self.tofino_model_dir, 'sram_model_v2.pkl'))
            self.tofino_stages_model = joblib.load(os.path.join(self.tofino_model_dir, 'stages_model_v2.pkl'))
            self.tofino_scaler = joblib.load(os.path.join(self.tofino_model_dir, 'scaler_v2.pkl'))
            
            self.tofino_tcam_is_ensemble = isinstance(self.tofino_tcam_model, list)
            self.tofino_sram_is_ensemble = isinstance(self.tofino_sram_model, list)
            self.tofino_stages_is_ensemble = isinstance(self.tofino_stages_model, list)
            self.tofino_available = True
        except Exception as e:
            print(f"Error loading Tofino models: {e}")
            self.tofino_tcam_model = None
            self.tofino_sram_model = None
            self.tofino_stages_model = None
            self.tofino_scaler = None
            self.tofino_tcam_is_ensemble = False
            self.tofino_sram_is_ensemble = False
            self.tofino_stages_is_ensemble = False
            self.tofino_available = False
            
        try:
            self.bmv2_model = joblib.load(os.path.join(self.bmv2_model_dir, 'bmv2_model.pkl'))
            self.bmv2_scaler = joblib.load(os.path.join(self.bmv2_model_dir, 'scaler.pkl'))
            # 使用固定的基准值 26.00 MB (转换为字节)
            self.bmv2_base_rss_mb = 26.00
            self.bmv2_available = True
        except Exception as e:
            print(f"Error loading BMv2 models: {e}")
            self.bmv2_model = None
            self.bmv2_scaler = None
            self.bmv2_base_rss_mb = 26.00
            self.bmv2_available = False
        
        # Hardware Capacities (Tofino)
        self.TCAM_CAPACITY = 288
        self.SRAM_CAPACITY = 960

    def _get_tofino_prediction(self, model, X_scaled, is_ensemble):
        if not is_ensemble:
            return model.predict(X_scaled)
        
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

    def predict(self, match_types, size):
        """
        Predict all metrics.
        
        Args:
            match_types (list or np.array): 8 integers representing match types.
                                          0=Exact, 1=Ternary, 2=Range, 3=LPM
            size (int): Table size.
            
        Returns:
            dict: Dictionary containing all predictions.
        """
        match_types = np.array(match_types).flatten()
        if len(match_types) != 8:
            raise ValueError("match_types must have length 8")
        
        # Common calculations (needed for eBPF prediction below)
        m_types = np.array(match_types, dtype=float)
        counts = [np.sum(m_types == 0), np.sum(m_types == 1), np.sum(m_types == 2), np.sum(m_types == 3)]
            
        # --- 1. Tofino Prediction ---
        # Check if Tofino models are available
        if not self.tofino_available or self.tofino_scaler is None:
            # Return default/fallback values when models are not loaded
            tcam_tiles = 0.0
            sram_tiles = 0.0
            tcam_percent = 0.0
            sram_percent = 0.0
            stages_final = 1
        else:
            BIT_WIDTHS = [16, 8, 8, 8, 16, 16, 8, 8]
            bits = np.array(BIT_WIDTHS, dtype=float)
            
            norm_size = (size - 256) / (20000 - 256)
            
            tofino_X = np.concatenate([m_types, bits, [norm_size], counts]).reshape(1, -1)
            tofino_X_scaled = self.tofino_scaler.transform(tofino_X)
            
            tofino_X_stages = np.concatenate([m_types, bits, [norm_size], counts]).reshape(1, -1)
            tofino_X_stages_scaled = self.tofino_scaler.transform(tofino_X_stages)
            
            tcam_tiles = self._get_tofino_prediction(self.tofino_tcam_model, tofino_X_scaled, self.tofino_tcam_is_ensemble)[0]
            sram_tiles = self._get_tofino_prediction(self.tofino_sram_model, tofino_X_scaled, self.tofino_sram_is_ensemble)[0]
            stages_pred = self._get_tofino_prediction(self.tofino_stages_model, tofino_X_stages_scaled, self.tofino_stages_is_ensemble)[0]
            
            tcam_percent = (tcam_tiles / self.TCAM_CAPACITY) * 100
            sram_percent = (sram_tiles / self.SRAM_CAPACITY) * 100
            stages_final = int(round(stages_pred))

        # --- 2. BMv2 Prediction ---
        if not self.bmv2_available or self.bmv2_scaler is None:
            bmv2_total_rss_mb = self.bmv2_base_rss_mb
            bmv2_delta_rss_kb = 0.0
        else:
            bmv2_base = np.append(match_types, size).reshape(1, -1)
            bmv2_X_scaled = self.bmv2_scaler.transform(bmv2_base)
            
            bmv2_log_pred = self.bmv2_model.predict(bmv2_X_scaled)[0]
            bmv2_delta_rss_kb = np.expm1(bmv2_log_pred)
            bmv2_total_rss_mb = self.bmv2_base_rss_mb + (bmv2_delta_rss_kb / 1024)
        
        # --- 3. eBPF Prediction ---
        has_ternary = counts[1] > 0
        has_lpm = counts[3] > 0
        
        if has_ternary:
            ebpf_map_memory = 58.0 * size
            ebpf_instructions = 1077
            ebpf_jit_size = 2919
        elif has_lpm:
            ebpf_map_memory = 0.0
            ebpf_instructions = 994
            ebpf_jit_size = 2453
        else:
            ebpf_map_memory = 13.0 * size
            ebpf_instructions = 992
            ebpf_jit_size = 2454
            
        return {
            "Tofino": {
                "TCAM (Tiles)": round(tcam_tiles, 2),
                "SRAM (Tiles)": round(sram_tiles, 2),
                "TCAM (%)": round(tcam_percent, 4),
                "SRAM (%)": round(sram_percent, 4),
                "Stages": stages_final
            },
            "BMv2": {
                "Total RSS (MB)": round(bmv2_total_rss_mb, 2),
                "Base RSS (MB)": self.bmv2_base_rss_mb,
                "Delta RSS (KB)": round(bmv2_delta_rss_kb, 2)
            },
            "eBPF": {
                "Map Memory (Bytes)": int(ebpf_map_memory),
                "Estimated Instructions": int(ebpf_instructions),
                "JIT Size (Bytes)": int(ebpf_jit_size)
            }
        }

def print_results(results, match_types, size):
    """格式化打印预测结果"""
    print("\n" + "=" * 80)
    print(f"  输入参数: Match Types = {match_types}, Size = {size}")
    print("=" * 80)
    
    print("\n【Tofino 预测】")
    print(f"  TCAM 使用: {results['Tofino']['TCAM (Tiles)']} Tiles ({results['Tofino']['TCAM (%)']:.4f}%)")
    print(f"  SRAM 使用: {results['Tofino']['SRAM (Tiles)']} Tiles ({results['Tofino']['SRAM (%)']:.4f}%)")
    print(f"  Stages 数: {results['Tofino']['Stages']} (整数预测值)")
    
    print("\n【BMv2 预测】")
    print(f"  总 RSS: {results['BMv2']['Total RSS (MB)']:.2f} MB")
    print(f"  基础 RSS: {results['BMv2']['Base RSS (MB)']:.2f} MB")
    print(f"  增量 RSS: {results['BMv2']['Delta RSS (KB)']:.2f} KB")
    
    print("\n【eBPF 预测】")
    print(f"  Map 内存: {results['eBPF']['Map Memory (Bytes)']:.0f} Bytes")
    print(f"  预估指令数: {results['eBPF']['Estimated Instructions']:.0f}")
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
    
    args = parser.parse_args()
    
    # 加载预测器
    print("正在加载预测模型...")
    predictor = UnifiedPredictor()
    print("模型加载完成!\n")
    
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
                
                match_input = input("  Match Types (8个数字，空格分隔): ").strip()
                if match_input.lower() == 'q':
                    print("退出程序。")
                    break
                
                match_types = [int(x) for x in match_input.split()]
                if len(match_types) != 8:
                    print("  错误: 必须输入8个匹配类型!")
                    continue
                
                if not all(0 <= m <= 3 for m in match_types):
                    print("  错误: 匹配类型必须在 0-3 之间!")
                    continue
                
                size_input = input("  Size (256-8192): ").strip()
                if size_input.lower() == 'q':
                    print("退出程序。")
                    break
                
                size = int(size_input)
                if not (256 <= size <= 8192):
                    print("  错误: Size 必须在 256-8192 之间!")
                    continue
                
                results = predictor.predict(match_types, size)
                print_results(results, match_types, size)
                
            except ValueError as e:
                print(f"  输入错误: {e}")
            except KeyboardInterrupt:
                print("\n\n程序被中断。")
                break
            except Exception as e:
                print(f"  预测错误: {e}")
    
    # 模式 3: 示例模式 (默认)
    else:
        print("=" * 80)
        print("  示例预测模式")
        print("=" * 80)
        print("\n提示: 使用 -h 查看所有使用方式\n")
        
        examples = [
            {'name': '用户测试案例', 'match': [2, 1, 1, 1, 0, 0, 1, 0], 'size': 4024, 'note': '1个Range, 4个Ternary, 3个Exact'},
            {'name': '全精确匹配', 'match': [0, 0, 0, 0, 0, 0, 0, 0], 'size': 4096, 'note': '8个Exact'},
            {'name': '单个三态匹配', 'match': [1, 0, 0, 0, 0, 0, 0, 0], 'size': 1024, 'note': '1个Ternary, 7个Exact'},
            {'name': '单个LPM匹配', 'match': [3, 0, 0, 0, 0, 0, 0, 0], 'size': 8192, 'note': '1个LPM, 7个Exact'},
            {'name': '混合匹配', 'match': [2, 2, 1, 1, 0, 0, 1, 1], 'size': 3768, 'note': '2个Range, 3个Ternary, 3个Exact'}
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
