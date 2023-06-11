import os
import logging
import numpy as np
from astropy.table import Table, vstack
from LSS.tabulated_cosmo import TabulatedDESI
import matplotlib.pyplot as plt


logger = logging.getLogger('io_tools')


def get_z_cutsky(tracer):
    if tracer.startswith('LRG'):
        return [0.4, 1.1]
    elif tracer.startswith('ELG'):
        return [0.8, 1.6]
    elif tracer.startswith('QSO'):
        return [0.8, 2.1]
    elif tracer.startswith('BGS'):
        return [0.1, 0.4]

def get_z_cubicbox(tracer):
    if tracer.startswith('LRG'):
        return 0.8
    elif tracer.startswith('ELG'):
        return 1.1
    elif tracer.startswith('QSO'):
        return 1.4
    elif tracer.startswith('BGS'):
        return 0.2

def catalog_dir(tracer, mock_type='cutsky', phase_idx=0, cosmo_idx=0, 
    base_dir='/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/AbacusSummit/'):
    if tracer.startswith('BGS'): tracer += '_v2'
    redshift = get_z_cubicbox(tracer)
    if mock_type == 'cubicbox':
        return os.path.join(base_dir, 'CubicBox', tracer, f'z{redshift:.3f}',
                            f'AbacusSummit_base_c{cosmo_idx:03}_p{phase_idx:03}')
    return os.path.join(base_dir, 'CutSky', tracer, f'z{redshift:.3f}')


def catalog_fn(tracer='LRG', mock_type='cutsky', cat_dir=None, phase=0,
    name='data', nrandoms=4, rec_type=None, region=None, **kwargs):
    if cat_dir is None:
        cat_dir = catalog_dir(tracer=tracer, mock_type=mock_type, **kwargs)
    redshift = get_z_cubicbox(tracer)

    if mock_type == 'cutsky':
        if rec_type:
            if name == 'data':
                return os.path.join(cat_dir, f'cutsky_{tracer}_{region}_z{redshift:.3f}_AbacusSummit_base_c000_ph{phase:03}_{rec_type}.fits')
            return [os.path.join(cat_dir, f'cutsky_{tracer}_{region}_random_S{i*100}_1X_{rec_type}.fits') for i in range(1, nrandoms + 1)]
        if name == 'data':
            return os.path.join(cat_dir, f'cutsky_{tracer}_z{redshift:.3f}_AbacusSummit_base_c000_ph{phase:03}.fits')
        if tracer.startswith('BGS'):
            return [os.path.join(cat_dir, f'random_S{i*100}_1X.fits') for i in range(1, nrandoms + 1)]
        return [os.path.join(cat_dir, f'cutsky_{tracer}_random_S{i*100}_1X.fits') for i in range(1, nrandoms + 1)]
    else:
        raise NotImplementedError(f'catalog_fn not implemented for mock_type={mock_type}')

def read_positions_weights_cubicbox(filename, boxsize, hubble, az, los='z'):
    data = Table.read(filename)
    vx = data['vx']
    vy = data['vy']
    vz = data['vz']
    x = data['x']
    y = data['y']
    z = data['z']
    x_rsd = x + vx / (hubble * az)
    y_rsd = y + vy / (hubble * az)
    z_rsd = z + vz / (hubble * az)
    x_rsd = x_rsd % boxsize
    y_rsd = y_rsd % boxsize
    z_rsd = z_rsd % boxsize
    weights = np.ones_like(x)
    if los == 'x':
        return x_rsd, y, z, weights
    elif los == 'y':
        return x, y_rsd, z, weights
    elif los == 'z':
        return x, y, z_rsd, weights

def read_positions_weights_cutsky(filename, zlim=None, region='NGC', abs_maglim=None, app_maglim=None,
    weight_type='FKP', return_mask=False):
    if not isinstance(filename, (tuple, list)):
        filename = [filename]
    positions, weights, mask = [], [], []
    for fn in filename:
        data = Table.read(fn)
        if 'LRG' in fn:
            mask_bits = get_desi_mask(main=1, Y5=1)
        else:
            mask_bits = get_desi_mask(nz=1, Y5=0)
        desi_mask = ((data['STATUS'] & (mask_bits)) == mask_bits)
        redshift_mask = (data['Z'] > zlim[0]) & (data['Z'] < zlim[1])
        if region == 'NGC':
            region_mask = (data['RA'] > 80) & (data['RA'] < 300)
        else:
            region_mask = (data['RA'] < 80) | (data['RA'] > 300)
        _mask = desi_mask & redshift_mask & region_mask
        _weights = np.ones_like(data[_mask]['RA'])
        if 'FKP' in weight_type:
            if 'BGS' in fn:
                raise NotImplementedError('FKP weights not implemented for BGS.')
            nz = data[_mask]['NZ']
            _weights /= (1 + nz * 1e4)
        positions.append([data[_mask]['RA'], data[_mask]['DEC'], data[_mask]['Z']])
        weights.append(_weights)
        mask.append(_mask)
    positions = np.concatenate(positions, axis=1)
    weights = np.concatenate(weights)
    mask = np.concatenate(mask)
    if return_mask:
        return positions, weights, mask
    return positions, weights

def get_desi_mask(main=0, nz=0, Y5=0, sv3=0):
    return main * (2**3) + sv3 * (2**2) + Y5 * (2**1) + nz * (2**0)


# def _format_bitweights(bitweights):
#     if bitweights.ndim == 2: return list(bitweights.T)
#     return [bitweights]


# def get_clustering_positions_weights(catalog, distance, zlim=(0., np.inf),maglim=None, weight_type='default', name='data', return_mask=False, option=None):

#     if maglim is None:
#         mask = (catalog['Z'] >= zlim[0]) & (catalog['Z'] < zlim[1])
#     if maglim is not None:
#         mask = (catalog['Z'] >= zlim[0]) & (catalog['Z'] < zlim[1]) & (catalog['ABSMAG_R'] >= maglim[0]) & (catalog['ABSMAG_R'] < maglim[1])

#     logger.info('Using {:d} rows for {}.'.format(mask.sum(), name))
#     positions = [catalog['RA'][mask], catalog['DEC'][mask], distance(catalog['Z'][mask])]
#     weights = np.ones_like(positions[0])

#     if name == 'data':
#         if 'zfail' in weight_type:
#             weights *= catalog['WEIGHT_ZFAIL'][mask]
#         if 'default' in weight_type and 'bitwise' not in weight_type:
#             weights *= catalog['WEIGHT'][mask]
#         if 'RF' in weight_type:
#             weights *= catalog['WEIGHT_RF'][mask]*catalog['WEIGHT_COMP'][mask]
#         if 'completeness_only' in weight_type:
#             weights = catalog['WEIGHT_COMP'][mask]
#         if 'EB' in weight_type:
#             weights *=  catalog['WEIGHT_SYSEB'][mask]*catalog['WEIGHT_COMP'][mask]   
#         if 'FKP' in weight_type:
#             weights *= catalog['WEIGHT_FKP'][mask]
#         if 'bitwise' in weight_type:
#             weights = _format_bitweights(catalog['BITWEIGHTS'][mask]) + [weights]

#     if name == 'randoms':
#         if 'default' in weight_type:
#             weights *= catalog['WEIGHT'][mask]
#         if 'RF' in weight_type:
#             weights *= catalog['WEIGHT_RF'][mask]*catalog['WEIGHT_COMP'][mask]
#         if 'zfail' in weight_type:
#             weights *= catalog['WEIGHT_ZFAIL'][mask]
#         if 'completeness_only' in weight_type:
#             weights = catalog['WEIGHT_COMP'][mask]
#         if 'EB' in weight_type:
#             weights *=  catalog['WEIGHT_SYSEB'][mask]*catalog['WEIGHT_COMP'][mask]   
#         if 'FKP' in weight_type:
#             weights *= catalog['WEIGHT_FKP'][mask]

#     if return_mask:
#         return positions, weights, mask
#     return positions, weights


# def _concatenate(arrays):
#     if isinstance(arrays[0], (tuple, list)):  # e.g., list of bitwise weights for first catalog
#         array = [np.concatenate([arr[iarr] for arr in arrays], axis=0) for iarr in range(len(arrays[0]))]
#     else:
#         array = np.concatenate(arrays, axis=0)  # e.g. individual weights for first catalog
#     return array


# def read_clustering_positions_weights(distance, zlim =(0., np.inf), maglim=None, weight_type='default', name='data', concatenate=False, option=None, region=None, cat_read=None, dat_cat=None, ran_cat=None, **kwargs):
#     #print(kwargs)
#     if 'GC' in region:
#         region = [region]
    
#     if cat_read == None:
#         def read_positions_weights(name):
#             positions, weights = [], []
#             for reg in region:
#                 cat_fns = catalog_fn(ctype='clustering', name=name, region=reg, **kwargs)
#                 logger.info('Loading {}.'.format(cat_fns))
#                 isscalar = not isinstance(cat_fns, (tuple, list))
   
                
#                 if isscalar:
#                     cat_fns = [cat_fns]
#                 positions_weights = [get_clustering_positions_weights(Table.read(cat_fn), distance, zlim=zlim, maglim=maglim, weight_type=weight_type, name=name, option=option) for cat_fn in cat_fns]
                
#                 if isscalar:
#                     positions.append(positions_weights[0][0])
#                     weights.append(positions_weights[0][1])
#                 else:
#                     p, w = [tmp[0] for tmp in positions_weights], [tmp[1] for tmp in positions_weights]
#                     if concatenate:
#                         p, w = _concatenate(p), _concatenate(w)
#                     positions.append(p)
#                     weights.append(w)
            
#             return positions, weights

#     if cat_read != None:
#         def read_positions_weights(name):
#             positions, weights = [], []
#             for reg in region:
#                 logger.info('Using arrays.')
                
#                 if name == 'data':
#                     cat_read = dat_cat
#                 if name == 'randoms':
#                     cat_read = ran_cat
                   
                    
#                 positions_weights = [get_clustering_positions_weights(cat_read, distance, zlim=zlim, maglim=maglim, weight_type=weight_type, name=name, option=option)]
#                 if name == 'data':
#                     positions.append(positions_weights[0][0])
#                     weights.append(positions_weights[0][1])
                
#                 if name == 'randoms':
#                     p, w = [tmp[0] for tmp in positions_weights], [tmp[1] for tmp in positions_weights]
#                     positions.append(p)
#                     weights.append(w)
            
#             return positions, weights
        
    
#     if isinstance(name, (tuple, list)):
#         return [read_positions_weights(n) for n in name]
#     return read_positions_weights(name)


# def get_full_positions_weights(catalog, name='data', weight_type='default', fibered=False, region='', return_mask=False, weight_attrs=None):
    
#     from pycorr.twopoint_counter import get_inverse_probability_weight
#     if weight_attrs is None: weight_attrs = {}
#     mask = np.ones(len(catalog), dtype='?')
#     if region in ['DS', 'DN']:
#         mask &= select_region(catalog['RA'], catalog['DEC'], region)
#     elif region:
#         mask &= catalog['PHOTSYS'] == region.strip('_')

#     if fibered: mask &= catalog['LOCATION_ASSIGNED']
#     positions = [catalog['RA'][mask], catalog['DEC'][mask], catalog['DEC'][mask]]
#     if name == 'data' and fibered:
#         if 'default' in weight_type or 'completeness' in weight_type:
#             weights = get_inverse_probability_weight(_format_bitweights(catalog['BITWEIGHTS'][mask]), **weight_attrs)
#         if 'bitwise' in weight_type:
#             weights = _format_bitweights(catalog['BITWEIGHTS'][mask])
#     else: weights = np.ones_like(positions[0])
#     if return_mask:
#         return positions, weights, mask
#     return positions, weights


# def read_full_positions_weights(name='data', weight_type='default', fibered=False, region='', weight_attrs=None, **kwargs):

#     def read_positions_weights(name):
#         positions, weights = [], []
#         for reg in region:
#             cat_fn = catalog_fn(ctype='full', name=name, **kwargs)
#             logger.info('Loading {}.'.format(cat_fn))
#             if isinstance(cat_fn, (tuple, list)):
#                 catalog = vstack([Table.read(fn) for fn in cat_fn])
#             else:
#                 catalog = Table.read(cat_fn)
#             p, w = get_full_positions_weights(catalog, name=name, weight_type=weight_type, fibered=fibered, region=reg, weight_attrs=weight_attrs)
#             positions.append(p)
#             weights.append(w)
#         return positions, weights

#     if isinstance(name, (tuple, list)):
#         return [read_positions_weights(n) for n in name]
#     return read_positions_weights(name)


# def normalize_data_randoms_weights(data_weights, randoms_weights, weight_attrs=None):
#     # Renormalize randoms / data for each input catalogs
#     # data_weights should be a list (for each N/S catalogs) of weights
#     import inspect
#     from pycorr.twopoint_counter import _format_weights, get_inverse_probability_weight
#     if weight_attrs is None: weight_attrs = {}
#     weight_attrs = {k: v for k, v in weight_attrs.items() if k in inspect.getargspec(get_inverse_probability_weight).args}
#     wsums, weights = {}, {}
#     for name, catalog_weights in zip(['data', 'randoms'], [data_weights, randoms_weights]):
#         wsums[name], weights[name] = [], []
#         for w in catalog_weights:
#             w, nbits = _format_weights(w, copy=True)  # this will sort bitwise weights first, then single individual weight
#             iip = get_inverse_probability_weight(w[:nbits], **weight_attrs) if nbits else 1.
#             iip = iip * w[nbits]
#             wsums[name].append(iip.sum())
#             weights[name].append(w)
#     wsum_data, wsum_randoms = sum(wsums['data']), sum(wsums['randoms'])
#     for icat, w in enumerate(weights['randoms']):
#         factor = wsums['data'][icat] / wsums['randoms'][icat] * wsum_randoms / wsum_data
#         w[-1] *= factor
#         logger.info('Rescaling randoms weights of catalog {:d} by {:.4f}.'.format(icat, factor))
#     return weights['data'], weights['randoms']


# def concatenate_data_randoms(data, randoms=None, **kwargs):

#     if randoms is None:
#         positions, weights = data
#         return _concatenate(positions), _concatenate(weights)

#     positions, weights = {}, {}
#     for name in ['data', 'randoms']:
#         positions[name], weights[name] = locals()[name]
#     for name in positions:
#         concatenated = not isinstance(positions[name][0][0], (tuple, list))  # first catalog, unconcatenated [RA, DEC, distance] (False) or concatenated RA (True)?
#         if concatenated:
#             positions[name] = _concatenate(positions[name])
#         else: 
#             positions[name] = [_concatenate([p[i] for p in positions[name]]) for i in range(len(positions['randoms'][0]))]
#     data_weights, randoms_weights = [], []
#     if concatenated:
#         wd, wr = normalize_data_randoms_weights(weights['data'], weights['randoms'], weight_attrs=kwargs.get('weight_attrs', None))
#         weights['data'], weights['randoms'] = _concatenate(wd), _concatenate(wr)
#     else:
#         for i in range(len(weights['randoms'][0])):
#             wd, wr = normalize_data_randoms_weights(weights['data'], [w[i] for w in weights['randoms']], weight_attrs=kwargs.get('weight_attrs', None))
#             data_weights.append(_concatenate(wd))
#             randoms_weights.append(_concatenate(wr))
#         weights['data'] = data_weights[0]
#         for wd in data_weights[1:]:
#             for w0, w in zip(weights['data'], wd): assert np.all(w == w0)
#         weights['randoms'] = randoms_weights
#     return [(positions[name], weights[name]) for name in ['data', 'randoms']] 
