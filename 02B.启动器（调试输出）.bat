@echo off
call venv\Scripts\activate.bat
if defined VIRTUAL_ENV (
start python launcher.py
) else (
echo Python venv cannot be activated. 
echo Please check the project folder and your python installation. Only english character is allowed in their paths.
)
pause