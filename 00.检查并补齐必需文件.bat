@echo off
echo 1. Check Python Install
where pip>nul
if %ERRORLEVEL% NEQ 0 (
echo Python or pip not found, use link 00A.
) else (
python --version
pip --version
python -c "import sys; print('OK! Python version verified' if sys.version_info[0]>=3 and sys.version_info[1]>7 else 'Python version less than 3.8, use link 00A to get the latest Python3.')"
)
echo.
echo 2. Pretrained data
if exist pretrained\combiner.pt (
echo OK! Pretrained data found.
) else (
echo Pretrained data not found, use link 00B and extract it to pretrained/..
)
echo.
echo 3. UnityCapture
reg query HKLM\SOFTWARE\Classes /s /f UnityCaptureFilter32bit.dll>nul
if errorlevel 1 (
echo UnityCapture not found, use link 00C if you need RGBA output.
) else (
echo OK! UnityCapture found.
)
echo.
echo.
echo Make sure everything is OK before you run 01A or 01B.
pause