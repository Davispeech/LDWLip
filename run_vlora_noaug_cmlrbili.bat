@echo off
setlocal
rem Start this file by hand. One click runs CMLR, then CVSR-Bili.
rem It calls run_vlora_noaug_cmlr.bat and run_vlora_noaug_bili.bat.
rem Those files now run V-LoRA +Sem only. Ours (w/o aug) calls are commented.
rem Those two files do not pause when started from here.
rem ldw_main.py still prints straight to this window.
rem It does not change ldwLip_wjc.

set NO_PAUSE=1
call "%~dp0run_vlora_noaug_cmlr.bat"
if errorlevel 1 goto :fail
call "%~dp0run_vlora_noaug_bili.bat"
if errorlevel 1 goto :fail

echo cmlr and bili runs finished
pause
exit /b 0

:fail
echo stopped because a run failed
pause
exit /b 1
