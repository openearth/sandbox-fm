def dflowfm_compute(data):
    data['is_wet'] = data['s1'] > data['bl']

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
        X_NODES="xk",
        Y_NODES="yk",
        X_CELLS="xzw",
        Y_CELLS="yzw",
        DEPTH_NODES="zk",
        DEPTH_CELLS="bl",
        WATERLEVEL="s1",
        U="ucx",
        V="ucy"
    ),
    "compute": dflowfm_compute
}

def xbeach_compute(data):
    # wetz -> is_wet
    pass


xbeach = {
    "initial_vars": [
        'x',
        'y',
        'xz',
        'yz',
        'H'
    ],
    "vars": ['zb', 'zs', 'H', 'u', 'v', 'cgx', 'cgy', "sedero"],
    "mapping": dict(
        X_NODES="x",
        Y_NODES="y",
        X_CELLS="x",
        Y_CELLS="y",
        DEPTH_NODES="zb",
        DEPTH_CELLS="zb",
        WATERLEVEL="zs",
        U="u",
        V="v",
        H="H",
        WAVE_U="cgx",
        WAVE_V="cgy",
        EROSION="sedero"
    ),
    "compute": xbeach_compute
}
