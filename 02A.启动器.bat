@echo off
echo Activating Python Virtual Environment...
call venv\Scripts\activate.bat
if defined VIRTUAL_ENV (
start pythonw launcher.py
) else (
echo Missing .\venv\Scripts\activate.bat, Python venv cannot be activated. 
echo Please check the project folder and your python installation. Only english character is allowed in their paths.
)