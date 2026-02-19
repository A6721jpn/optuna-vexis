@echo off
set PYTHONPATH=%~dp0src;%PYTHONPATH%
"C:\Users\aokuni\AppData\Local\miniforge3\envs\fcad\python.exe" -m proto4_claude.runner --config optimizer_config.yaml --limits proto4_limitations.yaml --max-trials 10 --verbose > output_log.txt 2>&1
if errorlevel 1 type output_log.txt
