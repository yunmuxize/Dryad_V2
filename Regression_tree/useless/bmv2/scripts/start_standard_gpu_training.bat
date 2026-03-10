@echo off
REM 启动标准模式 + GPU训练（后台运行）

cd /d "%~dp0\.."

REM 创建logs目录
if not exist "logs" mkdir logs

REM 生成时间戳
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set timestamp=%datetime:~0,8%_%datetime:~8,6%

REM 启动后台训练，重定向日志
echo ============================================================
echo 启动标准模式 + GPU训练
echo ============================================================
echo 日志文件: logs\training_standard_gpu_%timestamp%.log
echo ============================================================
echo.

start /B C:\Users\86177\anaconda3\envs\linc_env\python.exe scripts\train_standard_gpu.py > logs\training_standard_gpu_%timestamp%.log 2>&1

echo 训练已在后台启动！
echo.
echo 查看日志: type logs\training_standard_gpu_%timestamp%.log
echo 或使用: python scripts\check_dryad_process.py
echo.
pause




