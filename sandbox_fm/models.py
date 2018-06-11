import numpy as np
import logging


logger = logging.getLogger(__name__)


# Temporary copied from sandbox_fm.variables, because importing fails
def update_vars(data, model):
    """get the variables from the model and put them in the data dictionary"""
    meta = available[model.engine]
    for name in meta['vars']:
        data[name] = model.get_var(name)
    # do some stuff per model
    meta["compute"](data)
    for key, val in meta["mapping"].items():
        data[key] = data[val]

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
    # Arbitrary theshhold of 0.1 m for dry cells
    data['is_wet'] = (data['s1'] - data['bl']) < 0.1

# # Disable the FMCustomWrapper. The method to update with set_var did not
# # work, so we are now using set_var_slice, which is working correctly in BMIW
# # and MMI.
#
# import ctypes
# import numpy as np
# import pathlib
# import bmi.wrapper
# import time
#
# # The set_var does not work in current versions of FM
# class FMCustomWrapper(bmi.wrapper.BMIWrapper):
#     def __init__(self, *args, **kwargs):
#         super(self.__class__, self).__init__(*args, **kwargs)
#         self.library.update_land.argtypes = [
#             ctypes.POINTER(ctypes.c_int),
#             ctypes.POINTER(ctypes.c_double)
#         ]
#
#     def set_var(self, name, arr):
#         if name == 'zk':
#             zk_old = self.get_var('zk').copy()
#         logger.info('Updating layer {}'.format(name))
#         super(self.__class__, self).set_var(name, arr)
#         if name == 'zk':
#             zk_new = self.get_var('zk')
#             # get indices of changed bathymetries
#             indices, = np.where(zk_old != zk_new)
#             # see implementation in unstruc_bmi
#             # workaround for missing bathy updates
#             for idx in indices:
#                 self.library.update_land(
#                     ctypes.byref(ctypes.c_int(idx + 1)),
#                     ctypes.byref(ctypes.c_double(zk_new[idx]))
#                 )
#             self.library.on_land_change()



def update_height_dflowfm(idx, height_nodes_new, data, model):
    # nn = 0
    # for i in np.where(idx)[0]:
    #     # Only update model where the bed level changed (by compute_delta_height)
    #     if True:  # height_nodes_new[i] < data['bedlevel_update_maximum'] and np.abs(height_nodes_new[i] - data['HEIGHT_NODES'][i]) > data['bedlevel_update_threshold']:
    #         nn += 1
    #         model.set_var_slice('zk', [int(i+1)], [1], height_nodes_new[i:i + 1])
    # print('Total bed level updates', nn)
    drycells = (data['s1'] - data['bl']) < 0.1
    model.set_var_slice('zk', [1], [len(height_nodes_new)], height_nodes_new)  # This is quick!
    # model.set_var('zk', height_nodes_new)

    # If the cell was dry before, keep it dry by lowering the water level
    update_vars(data, model)
    bl = data['bl'].copy()
    s1 = data['s1'].copy()
    s1[drycells] = bl[drycells]
    model.set_var_slice('s1', [1], [len(s1)], s1)
    # model.set_var('s1', s1)  # Does not work?



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
