@echo off
REM Windows批处理脚本 - 后台启动训练
REM 使用方法: start_training.bat [--extensive]

echo ============================================================
echo Dryad BMv2 Training - Background Mode
echo ============================================================

REM 设置Python路径（如果需要）
REM set PYTHONPATH=%CD%

REM 创建输出目录
if not exist logs mkdir logs
if not exist plots mkdir plots
if not exist models mkdir models

REM 启动训练（后台运行）
REM 使用start命令在后台窗口运行
start "Dryad Training" /MIN python train_background.py %*

echo Training started in background window.
echo Check logs/ directory for training progress.
echo Use monitor_training.py to monitor the process.
pause

