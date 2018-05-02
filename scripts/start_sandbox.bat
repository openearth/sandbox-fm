set PATH=%PATH%;d:\software\dflowfm-x64-1.1.269.53512;D:\software\Anaconda3\envs\main\Scripts\
start mmi runner dflowfm ../models/Waal_schematic/waal_with_side.mdu --port 62000 --pause -o s1 -o bl -o ucx -o ucy -o zk
sandbox-fm run --mmi tcp://localhost:62000 ../models/Waal_schematic/waal_with_side.mdu
pause



REM move sandbox to correct screen
REM
REM list all windows
REM wmctrl -l
REM pick the with the sandbox, store in $win
REM
REM wmctrl -r sandbox-fm -b remove,fullscreen
REM
REM Select window for actions
REM wmctrl -r $win
REM Switch to desktop 2
REM wmctrl -s 2
REM move window to current desktop
REM wmctrl -R $win
REM wmctrl -a
