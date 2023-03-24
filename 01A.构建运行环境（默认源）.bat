@echo off
cd bin
python.exe -m pip install -r ..\requirements.txt --no-warn-script-location
python.exe -m pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 --no-warn-script-location
pause
