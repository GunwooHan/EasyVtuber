@echo off
echo Activating Python Virtual Environment...
call bin\Scripts\activate.bat
if defined VIRTUAL_ENV (
pip install wheel -i https://mirror.sjtu.edu.cn/pypi/web/simple
pip install -r .\requirements.txt -i https://mirror.sjtu.edu.cn/pypi/web/simple
pip install torch --extra-index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu113
) else (
echo Missing .\bin\Scripts\activate.bat, Python venv cannot be activated.
echo Please check the project folder. Only english character is allowed in their paths.
)
pause