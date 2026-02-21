@echo off
title Vision-Based Terrain Classification System

REM Move to the folder where this .bat file is located
cd /d "%~dp0"

REM Activate virtual environment
call .venv\Scripts\activate

REM Show Python version (optional but helpful)
python --version

REM Run the main application
python main.py

REM Keep window open after exit or error
pause
