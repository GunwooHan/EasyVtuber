@echo off
echo Activating Python Virtual Environment...
call venv\Scripts\activate.bat
if defined VIRTUAL_ENV (
start python launcher.py
) else (
echo Python venv cannot be activated. (.\venv\Scripts\activate.bat missing)
echo Please check the project folder and your python installation. Only english character is allowed in their paths.
)
pause