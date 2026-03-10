# -*- coding: utf-8 -*-
"""
检查dryad_predictor_enhanced.py的进程状态
"""

import subprocess
import os
from pathlib import Path
from datetime import datetime

def check_dryad_processes():
    """检查dryad_predictor_enhanced.py相关进程"""
    print("=" * 70)
    print("dryad_predictor_enhanced.py 进程检查")
    print("=" * 70)
    
    # 使用PowerShell检查进程
    ps_script = '''
    Get-Process python -ErrorAction SilentlyContinue | 
    Where-Object {$_.Path -like "*linc_env*"} | 
    ForEach-Object { 
        $cmd = (Get-WmiObject Win32_Process -Filter "ProcessId = $($_.Id)").CommandLine;
        if ($cmd -like "*dryad_predictor_enhanced*" -or $cmd -like "*joblib*") {
            $runtime = (Get-Date) - $_.StartTime;
            Write-Output "$($_.Id)|$($_.CPU)|$([math]::Round($_.WorkingSet64/1MB, 2))|$($_.StartTime)|$($runtime.ToString('hh\\:mm\\:ss'))|$cmd"
        }
    }
    '''
    
    try:
        result = subprocess.run(
            ['powershell', '-Command', ps_script],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        if result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            main_process = None
            worker_processes = []
            
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split('|')
                if len(parts) >= 6:
                    pid = parts[0]
                    cpu = float(parts[1])
                    memory = parts[2]
                    start_time = parts[3]
                    runtime = parts[4]
                    cmd = parts[5]
                    
                    if 'dryad_predictor_enhanced.py' in cmd and 'joblib' not in cmd:
                        main_process = {
                            'pid': pid,
                            'cpu': cpu,
                            'memory': memory,
                            'start_time': start_time,
                            'runtime': runtime,
                            'cmd': cmd
                        }
                    elif 'joblib' in cmd or cpu > 100:
                        worker_processes.append({
                            'pid': pid,
                            'cpu': cpu,
                            'memory': memory,
                            'start_time': start_time,
                            'runtime': runtime,
                            'cmd': cmd[:100] + '...' if len(cmd) > 100 else cmd
                        })
            
            print("\n1. 主进程 (dryad_predictor_enhanced.py):")
            if main_process:
                print(f"   PID: {main_process['pid']}")
                print(f"   CPU时间: {main_process['cpu']:.2f} 秒")
                print(f"   内存: {main_process['memory']} MB")
                print(f"   启动时间: {main_process['start_time']}")
                print(f"   运行时长: {main_process['runtime']}")
                print(f"   状态: [运行中]")
            else:
                print("   [未找到] 主进程可能已结束或未启动")
            
            print(f"\n2. GridSearchCV Worker进程 (共 {len(worker_processes)} 个):")
            if worker_processes:
                for i, worker in enumerate(worker_processes[:5], 1):  # 只显示前5个
                    print(f"   Worker {i}:")
                    print(f"     PID: {worker['pid']}")
                    print(f"     CPU时间: {worker['cpu']:.2f} 秒")
                    print(f"     内存: {worker['memory']} MB")
                    print(f"     运行时长: {worker['runtime']}")
                if len(worker_processes) > 5:
                    print(f"   ... 还有 {len(worker_processes) - 5} 个worker进程")
            else:
                print("   [未找到] 没有worker进程")
            
            # 检查日志文件
            print("\n3. 日志文件状态:")
            logs_dir = Path(__file__).parent.parent / "logs"
            log_file = logs_dir / "dryad_predictor_20251107_213615.log"
            
            if log_file.exists():
                stat = log_file.stat()
                size_kb = stat.st_size / 1024
                mtime = datetime.fromtimestamp(stat.st_mtime)
                time_diff = datetime.now() - mtime
                
                print(f"   文件: {log_file.name}")
                print(f"   大小: {size_kb:.2f} KB")
                print(f"   最后修改: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   距离现在: {str(time_diff).split('.')[0]}")
                
                # 读取最后几行
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        print(f"\n   最后5行日志:")
                        for line in lines[-5:]:
                            print(f"     {line.rstrip()}")
                except UnicodeDecodeError:
                    # 尝试其他编码
                    try:
                        with open(log_file, 'r', encoding='gbk') as f:
                            lines = f.readlines()
                            print(f"\n   最后5行日志:")
                            for line in lines[-5:]:
                                print(f"     {line.rstrip()}")
                    except:
                        print(f"\n   [无法读取] 日志文件编码问题")
                
                # 判断状态
                if main_process and time_diff.total_seconds() > 300:  # 5分钟没有更新
                    print(f"\n   [警告] 日志已 {int(time_diff.total_seconds()/60)} 分钟未更新")
                    print(f"   可能原因:")
                    print(f"     1. GridSearchCV正在执行，但verbose输出未刷新到日志")
                    print(f"     2. 训练可能卡住或遇到问题")
                    print(f"     3. 日志重定向可能有问题")
                elif main_process:
                    print(f"\n   [正常] 进程正在运行，日志可能正在更新")
            else:
                print("   [未找到] 日志文件不存在")
            
            # 总结
            print("\n" + "=" * 70)
            print("总结:")
            if main_process:
                print(f"  [OK] 主进程正在运行 (PID: {main_process['pid']})")
                print(f"  [OK] 有 {len(worker_processes)} 个worker进程在执行GridSearchCV")
                if time_diff.total_seconds() > 300:
                    print(f"  [警告] 日志已 {int(time_diff.total_seconds()/60)} 分钟未更新")
                    print(f"  [建议] 检查进程是否真的在工作（CPU使用率）")
                else:
                    print(f"  [OK] 日志最近有更新")
            else:
                print("  [未运行] 主进程未运行，训练可能已完成或已中断")
        
        else:
            print("\n[未找到] 没有找到dryad_predictor_enhanced.py相关进程")
            print("可能原因:")
            print("  1. 训练已完成")
            print("  2. 训练已中断")
            print("  3. 进程名称不匹配")
    
    except Exception as e:
        print(f"\n[错误] 检查进程时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_dryad_processes()

