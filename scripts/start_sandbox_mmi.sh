#!/bin/bash
MPLBACKEND=qt5agg

source ~/Envs/sandbox/bin/activate

model=../models/Lent/FlowFM.mdu

cd ~/sandbox-fm/scripts/

sandbox-fm run $model &

sleep 5

bps.py &

sleep 5

wmctrl -r 'Sandbox_figure' -e 0,1920,0,640,480
wmctrl -r 'Sandbox_figure' -b toggle,fullscreen
wmctrl -r 'BPS' -b toggle,fullscreen


read
