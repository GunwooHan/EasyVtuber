@echo off
python --version
echo Initializing Python Virtual Environment (.\venv)...
python -m venv venv
echo Activating Python Virtual Environment...
call venv\Scripts\activate.bat
if defined VIRTUAL_ENV (
pip install -r .\requirements.txt
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
) else (
echo Python venv cannot be activated. (.\venv\Scripts\activate.bat missing)
echo Please check the project folder and your python installation. Only english character is allowed in their paths.
)
pause