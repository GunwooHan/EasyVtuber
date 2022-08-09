@echo off
cd bin
pip install wheel -i https://mirror.sjtu.edu.cn/pypi/web/simple
pip install -r .\requirements.txt -i https://mirror.sjtu.edu.cn/pypi/web/simple
pip install torch --extra-index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu113
pause