@echo off
echo 1. Check Python Install
python --version
pip --version
echo If pip is not found, or Python version less than 3.8, use link 00A.
echo.
echo 2. Pretrained data
if exist pretrained\combiner.pt (
echo Pretrained data found.
) else (
echo Pretrained data not found, use link 00B and extract it to pretrained/..
)
echo.
echo 3. UnityCapture
reg query HKLM\SOFTWARE\Classes /s /f UnityCaptureFilter32bit.dll>"%tmp%\null"
if errorlevel 1 (
echo UnityCapture not found, use link 00C if you need RGBA output.
) else (
echo UnityCapture found.
)
echo.
echo.
echo Make sure everything is OK before you run 01A or 01B.
pause