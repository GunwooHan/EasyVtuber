python --version
python -m venv venv
call venv\Scripts\activate.bat
pip install -r .\requirements.txt
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
pause