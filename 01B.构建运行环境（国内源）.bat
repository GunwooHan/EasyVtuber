@echo off
cd bin
Scripts\pip install -r ..\requirements.txt -i https://mirror.sjtu.edu.cn/pypi/web/simple --no-warn-script-location
Scripts\pip install torch --extra-index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu113 --no-warn-script-location
pause