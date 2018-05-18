#!/bin/bash
MPLBACKEND=qt5agg

source ~/Envs/sandbox/bin/activate

model=../models/Lent/FlowFM.mdu

cd ~/sandbox-fm/scripts/
echo 'in sandbox folder' > autostart.log

mmi-runner dflowfm $model --port 62000 --pause -o s1 -o bl -o ucx -o ucy -o zk &

sleep 5

sandbox-fm run --mmi tcp://localhost:62000  $model &

sleep 5

bps.py &

sleep 5

wmctrl -r 'Sandbox_figure' -e 0,1920,0,640,480
wmctrl -r 'Sandbox_figure' -b toggle,fullscreen
wmctrl -r 'BPS' -b toggle,fullscreen

echo 'done' > autostart2.log

read
