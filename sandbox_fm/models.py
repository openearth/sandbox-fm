dflowfm_vars = ['bl', 'ucx', 'ucy', 's1', 'zk']

def dflowfm_compute(data):
    """compute variables that are missing/buggy/not available"""
    data['is_wet'] = data['s1'] > data['bl']
    numk = data['zk'].shape[0]
    data['numk'] = numk
    # fix shapes
    for var_name in dflowfm_vars:
        arr = data[var_name]
        if arr.shape[0] == data['numk']:
            data[var_name] = arr[:data['numk']]
        elif arr.shape[0] == data['ndx']:
            "should be of shape ndx"
            # ndxi:ndx are the boundary points (See  netcdf write code in unstruc)
            data[var_name] = arr[:data['ndxi']]
            # data should be off consistent shape now
        elif arr.shape[0] == data['ndxi']:
            # this is ok
            pass
        else:
            raise ValueError("unexpected data shape %s for variable %s" % (arr.shape, var_name))


def update_height_dflowfm(idx, height_nodes_copy, data, model):
    for i in np.where(idx)[0]:
        if data['HEIGHT_NODES'][i] != height_nodes_copy[i]:
            # TODO: bug in zk
            model.set_var_slice('zk', [i + 1], [1], height_nodes_copy[i:i + 1])

dflowfm = {
    "initial_vars": [
        'xzw',
        'yzw',
        'xk',
        'yk',
        'zk',
        'ndx',
        'ndxi',             # number of internal points (no boundaries)
        'flowelemnode'
    ],
    "vars": dflowfm_vars,
    "mapping": dict(
        X_NODES="xk",
        Y_NODES="yk",
        X_CELLS="xzw",
        Y_CELLS="yzw",
        HEIGHT_NODES="zk",
        HEIGHT_CELLS="bl",
        WATERLEVEL="s1",
        U="ucx",
        V="ucy"
    ),
    "compute": dflowfm_compute,
    "update_height": update_height_dflowfm
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
        HEIGHT_NODES="zb",
        HEIGHT_CELLS="zb",
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
