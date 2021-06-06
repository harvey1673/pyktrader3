set PYTHON_INCLUDE=C:\Anaconda3\include
set PYTHON_LIB=C:\Anaconda3\libs\python38.lib
set PYTHONPATH=.;C:\dev\pyktrader3;C:\dev\akshare;C:\dev\vnpy;C:\dev\pytdx;
chcp 936
%windir%\System32\cmd.exe "/c C:\Anaconda3\Scripts\activate.bat C:\Anaconda3 && python C:\\dev\\pyktrader3\\misc_scripts\\daily_update_job.py && deactivate"
pause
