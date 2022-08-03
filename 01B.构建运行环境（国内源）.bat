python --version
python -m venv venv
call venv\Scripts\activate.bat
if defined VIRTUAL_ENV (
pip install -r .\requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
) else (
echo Python venv cannot be activated. 
echo Please check the project folder and your python installation. Only english character is allowed in their paths.
)
pause