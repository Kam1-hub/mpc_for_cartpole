@echo off
cd /d "%~dp0"
title Cart-Pole MPC Simulator
echo =========================================
echo   Starting Cart-Pole MPC Simulation...
echo   (Using uv environment)
echo =========================================
uv run main.py
echo.
pause
