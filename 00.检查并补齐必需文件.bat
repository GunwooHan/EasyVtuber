@echo off
echo 1. Pretrained data
if exist data\models\standard_float\editor.pt (
echo OK! Pretrained data found.
) else (
echo Pretrained data not found, use link 00B and extract it to data/models..
)
echo.
echo.
echo Make sure everything is OK before you run 01A or 01B.
pause