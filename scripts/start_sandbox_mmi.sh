#!/bin/bash
MPLBACKEND=qt5agg

source ~/Envs/sandbox/bin/activate

model=../models/Lent/FlowFM.mdu

cd ~/sandbox-fm/scripts/

xrandr --listactivemonitors
sleep 30
xrandr --listactivemonitors
sleep 30
xrandr --listactivemonitors

mmi-runner dflowfm $model --port 62000 --pause -o s1 -o bl -o ucx -o ucy -o zk --interval=20 &
# mmi-runner dflowfm $model --port 62000 --pause -o s1 --interval 10 &

sleep 5

sandbox-fm run --mmi tcp://localhost:62000  $model &

sleep 5

bps.py &

sleep 5

wmctrl -r 'BPS' -b toggle,fullscreen

wmctrl -r 'Sandbox_figure' -e 0,1920,0,640,480
wmctrl -r 'Sandbox_figure' -b toggle,fullscreen


read
