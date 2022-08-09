@echo off
echo Activating Python Virtual Environment...
call bin\Scripts\activate.bat
if defined VIRTUAL_ENV (
start python launcher.py
) else (
echo Missing .\bin\Scripts\activate.bat, Python venv cannot be activated.
echo Please check the project folder. Only english character is allowed in their paths.
)
pause