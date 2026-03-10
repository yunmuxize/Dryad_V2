import os

def repair_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # 替换硬编码路径的通用逻辑
    # 针对 tofino 文件夹下的脚本
    if 'tofino\\scripts' in filepath:
        # 如果是 tofino/scripts/... 下的脚本，目标 base_dir 通常是 tofino 目录
        dynamic_tofino_base = 'os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))'
        
        content = content.replace(r'r"C:\Users\86177\OneDrive\Desktop\Dryad_V2\Dryad\tofino"', dynamic_tofino_base)
        content = content.replace(r'"C:\Users\86177\OneDrive\Desktop\Dryad_V2\Dryad\tofino"', dynamic_tofino_base)
        
        # 针对包含 \data 的情况
        content = content.replace(r'r"C:\Users\86177\OneDrive\Desktop\Dryad_V2\Dryad\tofino\data"', f'os.path.join({dynamic_tofino_base}, "data")')
        content = content.replace(r'"C:\Users\86177\OneDrive\Desktop\Dryad_V2\Dryad\tofino\data"', f'os.path.join({dynamic_tofino_base}, "data")')

    # 针对 Dryad/useless 等文件夹下的脚本
    if 'Dryad\\useless' in filepath:
        dynamic_dryad_base = 'os.path.dirname(os.path.dirname(os.path.abspath(__file__)))'
        content = content.replace(r'r"C:\Users\86177\OneDrive\Desktop\Dryad_V2\Dryad\Dryad"', dynamic_dryad_base)
        content = content.replace(r'"C:\Users\86177\OneDrive\Desktop\Dryad_V2\Dryad\Dryad"', dynamic_dryad_base)

    # 写回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    root_dir = r"d:\Desktop\Dryad_V2\Dryad"
    target_pattern = "C:\\Users\\86177"
    
    count = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.py', '.md')):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        if target_pattern in f.read():
                            print(f"Repairing: {filepath}")
                            repair_file(filepath)
                            count += 1
                except Exception as e:
                    print(f"Error checking {filepath}: {e}")
    
    print(f"Finished. Repaired {count} files.")

if __name__ == "__main__":
    main()
