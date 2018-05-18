#!/bin/bash
MPLBACKEND=qt5agg

model=../models/Lent/FlowFM.mdu

workon sandbox
cd ~/sandbox_fm/scripts/

mmi-runner dflowfm $model --port 62000 --pause -o s1 -o bl -o ucx -o ucy -o zk &
sleep 5
sandbox-fm run --mmi tcp://localhost:62000  $model &
bps.py &

wmctrl -r 'Sandbox_figure' -e 0,1920,0,640,480
wmctrl -r 'Sandbox_figure' -b toggle,fullscreen
wmctrl -r 'BPS' -b toggle,fullscreen
# move sandbox to correct screen

# list all windows
# wmctrl -l
# pick the with the sandbox, store in $win

# wmctrl -r sandbox-fm -b remove,fullscreen

# Select window for actions
# wmctrl -r $win
# Switch to desktop 2
# wmctrl -s 2
# move window to current desktop
# wmctrl -R $win
# wmctrl -a
