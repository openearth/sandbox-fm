def dflowfm_compute(data):
    data['s1'] > data['bl']

dflowfm = {
    "initial_vars": [
        'xzw',
        'yzw',
        'xk',
        'yk',
        'zk',
        'ndx',
        'ndxi',             # number of internal points (no boundaries)
        'numk',
        'flowelemnode'
    ],
    "vars": ['bl', 'ucx', 'ucy', 's1', 'zk'],
    "mapping": dict(
        X="xzw",
        Y="yzw",
        DEPTH="zk",
        WATERLEVEL="s1",
        U="u1",
        V="v1"
    ),
    "compute": dflowfm_compute
}

def xbeach_compute(data):
    pass

xbeach = {
    "initial_vars": [
        'x',
        'y',
        'xz',
        'yz',
        'H'
    ],
    "vars": ['zb', 'zs', 'H', 'u', 'v', 'cgx', 'cgy'],
    "mapping": dict(
        X="x",
        Y="y",
        DEPTH="zb",
        WATERLEVEL="zs",
        U="u",
        V="v",
        H="H",
        WAVE_U="cgx",
        WAVE_V="cgy"
    ),
    "compute": xbeach_compute
}
