set PYTHON_INCLUDE=C:\Anaconda3\include
set PYTHON_LIB=C:\Anaconda3\libs\python38.lib
set PYTHONPATH=.;C:\dev\wtpy;C:\dev\pyktrader3;C:\dev\akshare;
chcp 936
%windir%\System32\cmd.exe "/K" C:\Anaconda3\Scripts\activate.bat C:\Anaconda3" & cd C:\dev\pyktrader3 && python C:\dev\pyktrader3\misc_scripts\auto_update_data_xl.py
EXIT /b 0