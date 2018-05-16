import numpy as np
import logging

logger = logging.getLogger(__name__)

dflowfm_vars = ['bl', 'ucx', 'ucy', 's1', 'zk']

def dflowfm_compute(data):
    """compute variables that are missing/buggy/not available"""
    numk = data['zk'].shape[0]
    data['numk'] = numk
    # fix shapes
    for var_name in dflowfm_vars:
        arr = data[var_name]
        if arr.shape[0] == data['numk']:
            data[var_name] = arr[:data['numk']]
        elif arr.shape[0] == data['ndx']:
            "should be of shape ndx"
            # ndxi:ndx are the boundary points
            # (See  netcdf write code in unstruc)
            data[var_name] = arr[:data['ndxi']]
            # data should be off consistent shape now
        elif arr.shape[0] == data['ndxi']:
            # this is ok
            pass
        else:
            msg = "unexpected data shape %s for variable %s" % (
                arr.shape,
                var_name
            )
            raise ValueError(msg)
        # compute derivitave variables, should be consistent shape now.
    data['is_wet'] = data['s1'] > data['bl']

    

def update_height_dflowfm(idx, height_nodes_copy, data, model):
    for i in np.where(idx)[0]:
        # Only update model where the bed level changed (by compute_delta_height)
        if data['HEIGHT_NODES'][i] != height_nodes_copy[i]:
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
    "update_nodes": update_height_dflowfm
}


def xbeach_compute(data):
    # rotate velocties with grid angle
    data['u'] = (
        data['uu'] * np.cos(data['alfaz']) -
        data['vv'] * np.sin(data['alfaz'])
    )
    data['v'] = (
        data['uu'] * np.sin(data['alfaz']) +
        data['vv'] * np.cos(data['alfaz'])
    )
    data['cgu'] = (
        data['cgx'] * np.cos(data['alfaz']) -
        data['cgy'] * np.sin(data['alfaz'])
    )
    data['cgv'] = (
        data['cgx'] * np.sin(data['alfaz']) +
        data['cgy'] * np.cos(data['alfaz'])
    )


def update_height_xbeach(idx, height_nodes_copy, data, model):
    data['HEIGHT_NODES'].ravel()[idx] = height_nodes_copy.ravel()[idx]

def update_structure_height_xbeach(idx, height_nodes_copy, data, model):
    delta_height = height_nodes_copy - data['HEIGHT_NODES']
    data['STRUCTURE_HEIGHT'].ravel()[idx] = delta_height.ravel()[idx]

    
xbeach = {
    "initial_vars": [
        'x',
        'y',
        'xz',
        'yz',
        'H',
        'alfaz'
    ],

    "vars": ['zb', 'zs', 'H', 'D', 'cgx', 'cgy', "sedero", 'uu', 'vv', 'structdepth'],
    "mapping": dict(
        X_NODES="xz",
        Y_NODES="yz",
        X_CELLS="xz",
        Y_CELLS="yz",
        HEIGHT_NODES="zb",
        HEIGHT_CELLS="zb",
        WATERLEVEL="zs",
        U="u",
        V="v",
        H="H",
        WAVE_U="cgu",
        WAVE_V="cgv",
        WAVE_HEIGHT='H',
        WAVE_DISSIPATION='D',
        EROSION="sedero",
        STRUCTURE_HEIGHT="structdepth"
    ),
    "compute": xbeach_compute,
    "update_nodes": update_height_xbeach,
    "update_structures": update_structure_height_xbeach
}

dflowfm["reverse_mapping"] = {value: key for key, value in dflowfm["mapping"].items()}
xbeach["reverse_mapping"] = {value: key for key, value in xbeach["mapping"].items()}
available = {
    "xbeach": xbeach,
    "dflowfm": dflowfm

}
