@echo off
REM 启动优化的后台训练脚本

echo ============================================================
echo 启动优化训练（后台模式）
echo 目标: Memory R^2 >= 0.90, CPU准确率提升
echo ============================================================
echo.

cd /d "%~dp0"

REM 创建日志目录
if not exist logs mkdir logs

REM 获取时间戳
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set timestamp=%datetime:~0,8%_%datetime:~8,6%

REM 启动后台训练
echo 正在启动训练进程...
set LOG_FILE=logs\training_optimized_%timestamp%.log
echo 日志文件将保存到: %LOG_FILE%
echo.

start /B "" "C:\Users\86177\anaconda3\envs\linc_env\python.exe" train_optimized.py > "%LOG_FILE%" 2>&1

echo.
echo ============================================================
echo 训练已在后台启动！
echo ============================================================
echo 日志文件路径: %CD%\%LOG_FILE%
echo.
echo 可以使用以下命令查看进度:
echo   python monitor_training.py
echo   或直接查看日志文件: %LOG_FILE%
echo.
echo 查看最新日志内容:
type "%LOG_FILE%"
echo.
pause

