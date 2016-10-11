export DFLOWFM=../executable/lnx64/dflowfm
export LD_LIBRARY_PATH=$DFLOWFM/lib:$LD_LIBRARY_PATH
export PATH=$DFLOWFM/bin:$PATH
dflowfm --autostartstop zm_tide.mdu